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
            if idx == len(dims) - 2:
                activation = nn.Identity
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

    @NotImplementedError
    def backprop(self, x0: Tensor, x1: Tensor, loss_func: nn.Module = nn.MSELoss, do_return: bool = True) -> Tensor:
        self.optimizer.zero_grad()
        loss = loss_func(x0, x1)
        loss.backward()
        self.optimizer.step()

        if do_return:
            return loss
