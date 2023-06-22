from ncd.models.dag_gnn.model import DAG_GNN
from ncd.data import sachs
import pytorch_lightning as pl
from torch.utils.data import DataLoader

pl.seed_everything(1)
G, X = sachs(as_df=False)


model = DAG_GNN(11, [64], [64], G=G)
train_dataloader = DataLoader(X, batch_size=100, shuffle=True)

trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1000)
trainer.fit(model, train_dataloader)

G_ = model.get_graph()
acc = model.evaluate()
print(acc)