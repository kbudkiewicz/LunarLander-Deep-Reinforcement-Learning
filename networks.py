import torch
import torch.nn as nn
from torch import Tensor


class LinearBlock(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 device: torch.device,
                 activation: nn.Module = nn.ReLU,
                 normalization: nn.Module = None,
                 ):
        super().__init__()
        self.norm = normalization
        self.activation = activation
        self.block = nn.Sequential(
            normalization if normalization else nn.Identity(),
            nn.Linear(in_dim, out_dim),
            activation() if activation else nn.Identity(),
        )
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeedForwardNetwork(nn.Module):
    def __init__(self, *dims,
                 activation: nn.Module = nn.ReLU(),
                 regularizer: nn.Module = nn.Dropout(p=0.1)):
        super().__init__()
        self.dims = dims
        self.module_list = nn.ModuleList()
        self.activation = activation
        self.regularizer = regularizer

        for idx in range(len(self.dims) - 2):
            self.module_list.append(nn.Linear(self.dims[idx], self.dims[idx + 1]))
            self.module_list.append(self.activation)
            self.module_list.append(self.regularizer)
        self.module_list.append(nn.Linear(dims[-2], dims[-1]))  # last layer without activation
        self.module_list.append(self.regularizer)

        self.net = nn.Sequential(*self.module_list)

    def forward(self, state):
        return self.net(state)