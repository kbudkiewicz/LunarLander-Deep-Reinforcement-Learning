import torch
import torch.nn as nn

from typing import Tuple
from itertools import pairwise
from torch import Tensor


__all__ = [
    'LinearBlock',
    'FeedForwardNetwork',
    'DuelingQNetwork',
]


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: nn.Module,
        normalization: nn.Module,
    ):
        super().__init__()
        self.norm = normalization
        self.activation = activation
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            normalization(out_dim) if normalization else nn.Identity(),
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
        normalization: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("Need at least 2 dimensions to build a minimal model.")
        if any(d < 1 for d in dims):
            raise ValueError("Model dimensions must be strictly positive.")

        self.device = device
        self.net = nn.Sequential()
        for idx, (in_dim, out_dim) in enumerate(pairwise(dims)):
            if idx == len(dims) - 2:
                activation = nn.Identity
                normalization = nn.Identity
            self.net.append(
                LinearBlock(in_dim, out_dim, activation=activation, normalization=normalization)
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


class DuelingQNetwork(nn.Module):
    def __init__(
        self,
        *dims: int,
        activation: nn.Module = nn.ReLU,
        normalization: nn.Module = nn.LayerNorm,
        device: torch.device,
        encoder_depth: int = 2,
    ):
        super().__init__()
        if len(dims) < 3:
            raise ValueError("Need at least 3 dimensions to build a minimal model.")
        if any(d < 1 for d in dims):
            raise ValueError("Model dimensions must be strictly positive.")
        if not isinstance(encoder_depth, int) or encoder_depth < 1:
            raise ValueError("Encoder depth must be a strictly positive integer.")

        self.device = device
        self.encoder_depth = encoder_depth

        encoder, value_approximator, advantage_approximator = self._build_modules(
            *dims, activation=activation, normalization=normalization
        )
        self.encoder = encoder
        self.value_approximator = value_approximator
        self.advantage_approximator = advantage_approximator

    def forward(self, x: Tensor) -> Tensor:
        """Calculate the Q-value based on the approximated value of the current state and the approximated advantage,
        using the following equation:

        .. math::
            Q(s,a) = V(s) + (A(s,a) - \\frac{1}{|A|} \\sum_{a'} A(s,a'))
        """
        features = self.encoder(x)
        state_value = self.value_approximator(features)
        advantage = self.advantage_approximator(features)
        q_value = state_value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_value

    def _derive_model_dimensions(self, *dims: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        state_space, d, action_space = dims[0], dims[1:-1], dims[-1]
        setattr(self, 'encoder_dim', d[0])
        encoder_dims = (state_space, *[self.encoder_dim] * self.encoder_depth)
        value_dims = (*d, 1)
        advantage_dims = (*d, action_space)
        return encoder_dims, value_dims, advantage_dims

    def _build_modules(
        self,
        *dims: int,
        activation: nn.Module,
        normalization: nn.Module,
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        encoder_dims, value_dims, advantage_dims = self._derive_model_dimensions(*dims)

        encoder = FeedForwardNetwork(
            *encoder_dims, activation=activation, normalization=normalization, device=self.device
        )
        value_approximator = FeedForwardNetwork(
            *value_dims, activation=activation, normalization=normalization, device=self.device
        )
        advantage_approximator = FeedForwardNetwork(
            *advantage_dims, activation=activation, normalization=normalization, device=self.device
        )

        return encoder, value_approximator, advantage_approximator

    @property
    def model_type(self) -> str:
        return self.__class__.__name__

    @property
    def parameter_count(self) -> int:
        return sum(
            net.parameter_count for net in (self.encoder, self.value_approximator, self.advantage_approximator)
        )
