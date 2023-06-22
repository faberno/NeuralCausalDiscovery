import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import networkx as nx
import math
from ncd.metrics import evaluate_graph
from ncd.utils import is_dag

def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return lr

def stau(w, tau):
    w1 = F.relu(torch.abs(w) - tau)
    return torch.sign(w) * w1


class MLPEncoder(nn.Module):
    """MLP encoder module."""

    def __init__(self, n_xdims, hidden_layers, n_out, adj_A, mask_A):
        super(MLPEncoder, self).__init__()

        self.adj_A = nn.Parameter(adj_A, requires_grad=True)
        self.register_buffer('mask_A', mask_A.clone())

        self.layers = nn.ModuleList([nn.Linear(n_xdims, hidden_layers[0], bias=True)])
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1], bias=True))
        self.layers.append(nn.Linear(hidden_layers[-1], n_out, bias=True))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs, A, Wa):
        device = self.adj_A.device

        # I - A^T
        adj_norm = (torch.eye(A.shape[0], device=device) - (A.transpose(0, 1)))

        x = inputs
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        x = self.layers[-1](x)
        logits = torch.matmul(adj_norm, x + Wa) - Wa

        return logits


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_z, n_out, hidden_layers):
        super(MLPDecoder, self).__init__()

        self.layers = nn.ModuleList([nn.Linear(n_in_z, hidden_layers[0], bias=True)])
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1], bias=True))
        self.layers.append(nn.Linear(hidden_layers[-1], n_out, bias=True))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_z, origin_A, Wa):
        device = origin_A.device

        # (I-A^T)^(-1)
        adj_norm = torch.inverse(torch.eye(origin_A.shape[0], device=device) - origin_A.transpose(0, 1))
        mat_z = torch.matmul(adj_norm, input_z + Wa) - Wa

        x = mat_z
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        pred = self.layers[-1](x)

        return pred


class DAG_GNN(pl.LightningModule):
    def __init__(self,
                 d,
                 encoder_hidden_layers,
                 decoder_hidden_layers,
                 G=None,
                 adj_start=None,
                 adj_mask=None,
                 lr=3e-3,
                 graph_threshold=0.3,
                 tau_A=0.0,
                 lambda_A=0.0,
                 c_A=1.0,
                 seed=None):
        super().__init__()
        self.d = d
        self.graph_threshold = graph_threshold
        self.tau_A = tau_A
        self.initial_lambda_A = lambda_A
        self.current_lambda_A = lambda_A
        self.initial_c_A = c_A
        self.current_c_A = c_A
        self.initial_lr = lr
        self.current_lr = lr
        self.current_h = torch.inf
        self.new_h = np.inf
        self.seed = seed
        self.G = G

        self.Wa = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.current_graph = None

        self.h_tol = 1e-8

        if adj_start is None:
            adj_start = torch.zeros((d, d))
        if adj_mask is None:
            adj_mask = torch.ones((d, d)) - torch.eye(d)

        self.encoder = MLPEncoder(1, encoder_hidden_layers, 1, adj_start, mask_A=adj_mask)

        self.decoder = MLPDecoder(1, 1, hidden_layers=decoder_hidden_layers)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-3)
        #scheduler = StepLR(optimizer, step_size=200, gamma=1.0)
        return [optimizer]#, [scheduler]


    def on_train_epoch_start(self) -> None:
        optimizer = self.optimizers()
        self.current_lr = update_optimizer(optimizer, self.initial_lr, self.current_c_A)

    def forward(self, X):
        # to amplify the value of A and accelerate convergence.
        origin_A = torch.sinh(3. * self.encoder.adj_A * self.encoder.mask_A)  # adjacency matrix
        edges = self.encoder(X, origin_A, self.Wa)
        predictions = self.decoder(edges, origin_A, self.Wa)
        return edges, origin_A, predictions

    def step(self, batch, batch_idx):
        targets = batch

        if len(targets.shape) == 2:
            targets = torch.unsqueeze(targets, 2)

        edges, origin_A, predictions = self(targets)
        self.current_graph = origin_A
        h_A = self.calc_h()

        variance = 0.0
        nll_loss = self.nll_gaussian(predictions, targets, variance)
        kl_loss = self.kl_gaussian_sem(edges)
        sparse_loss = self.tau_A * torch.sum(torch.abs(origin_A))
        lagrangian_loss = self.current_lambda_A * h_A + 0.5 * self.current_c_A * h_A * h_A + 100. * torch.trace(
            origin_A * origin_A)

        loss = nll_loss + kl_loss + sparse_loss + lagrangian_loss

        self.log_dict({
            'h': h_A.item(),
            'nll_loss': nll_loss.item(),
            'kl_loss': kl_loss.item(),
            'sparse_loss': sparse_loss.item(),
            'lagrangian_loss': lagrangian_loss.item(),
            'loss': loss.item()
        })

        self.current_graph = origin_A

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        loss.backward()

    def optimizer_step(self,epoch, batch_idx, optimizer, optimizer_idx=0, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)
        self.encoder.adj_A.data = stau(self.encoder.adj_A.data, self.tau_A * self.current_lr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        acc = self.evaluate() if self.G is not None else dict()
        self.log_dict(acc)

    def get_graph(self):
        return self.current_graph.detach().cpu().clone().numpy()

    def evaluate(self):
        if self.G is None:
            return dict()
        G_ = self.get_graph()
        G_ = np.abs(G_) > self.graph_threshold
        acc = evaluate_graph(self.G, G_)
        return acc

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

        if self.current_epoch % 200 == 0 and self.current_epoch > 0:
            new_h = self.calc_h().item()
            if new_h > 0.25 * self.current_h:
                self.current_c_A *= 10  # if h didn't decrease enough, increase penalty
            else:
                self.current_h = new_h
                self.current_lambda_A += self.current_c_A * new_h

            if new_h <= self.h_tol or self.current_c_A > 1e20:
                self.trainer.should_stop()

    def matrix_poly(self, matrix):
        x = torch.eye(self.d, device=matrix.device) + torch.div(matrix, self.d)
        return torch.matrix_power(x, self.d)

    def calc_h(self):
        A = self.current_graph
        expm_A = self.matrix_poly(A * A)
        h_A = torch.trace(expm_A) - self.d
        return h_A

    def nll_gaussian(self, preds, target, variance, add_const=False):
        mean1 = preds
        mean2 = target
        neg_log_p = variance + torch.div(torch.pow(mean1 - mean2, 2), 2. * np.exp(2. * variance))
        if add_const:
            const = 0.5 * torch.log(2 * torch.from_numpy(np.pi) * variance)
            neg_log_p += const
        return neg_log_p.sum() / (target.size(0))

    def kl_gaussian_sem(self, preds):
        mu = preds
        kl_div = mu * mu
        kl_sum = kl_div.sum()
        return (kl_sum / (preds.size(0))) * 0.5

