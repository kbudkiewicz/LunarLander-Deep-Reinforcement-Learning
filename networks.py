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
    def __init__(self,
                 *dims,
                 activation: nn.Module = nn.ReLU,
                 normalization: nn.Module = nn.Dropout(p=0.1),
                 device=None):
        super().__init__()
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.net = nn.Sequential()
        for idx in range(len(dims) - 1):
            self.net.append(
                LinearBlock(
                    dims[idx], dims[idx + 1],
                    activation=activation,
                    normalization=normalization,
                    device=self.device,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

        self.net = nn.Sequential(*self.module_list)

    def forward(self, state):
        return self.net(state)