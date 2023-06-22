import torch.nn as nn
import torch
import math

class OffDiagLinear(nn.Module):

    num_vars: int
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, num_vars: int, in_features: int, out_features: int, bias: bool = True, first: bool = True, random_init: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OffDiagLinear, self).__init__()
        self.num_vars = num_vars
        self.in_features = in_features
        self.out_features = out_features
        self.first = first
        self.weight = nn.Parameter(torch.zeros(num_vars, out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.num_vars, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        # self.adjacency
        self.register_buffer('adjacency', torch.ones((self.num_vars, self.num_vars), **factory_kwargs) - torch.eye(self.num_vars, **factory_kwargs))

        if random_init:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.first:
            adj = self.adjacency.unsqueeze(0)
            x = torch.einsum("tij,ljt,bj->bti", self.weight, adj, input)
        else:
            x = torch.einsum("tij,btj->bti", self.weight, input)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
