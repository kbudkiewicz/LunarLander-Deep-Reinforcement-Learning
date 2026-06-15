import torch
import torch.nn as nn

from torch import Tensor


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
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

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        *dims: int,
        device: torch.device,
        activation: nn.Module = nn.ReLU,
        normalization: nn.Module = nn.Dropout(p=0.1)
    ):
        super().__init__()
        self.device = device
        self.net = nn.Sequential()
        for idx in range(len(dims) - 1):
            if idx == len(dims) - 2:
                activation = nn.Identity
            self.net.append(
                LinearBlock(
                    dims[idx], dims[idx + 1],
                    activation=activation,
                    normalization=normalization,
                )
            )
        self.to(self.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @property
    def model_type(self) -> str:
        return self.__class__.__name__

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())
