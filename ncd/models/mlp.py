import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .utils import OffDiagLinear
class SimpleAdjacencyMLP(nn.Module):
    def __init__(self, input_dim:int, layers: List[int], activation):
        super().__init__()
        self.layers = []
        for i in range(len(layers) - 1):
            in_dim = layers[i] if i != 0 else layers[i] * input_dim
            out_dim = layers[i+1]
            layer = OffDiagLinear(input_dim, in_dim, out_dim, first=i==0)
            self.layers.append(layer)
            if i != len(layers) - 2:
                self.layers.append(activation)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, input):
        return self.layers(input)
