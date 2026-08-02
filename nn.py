import torch
import torch.nn as nn

from typing import Tuple, Optional, Union
from itertools import pairwise
from torch import Tensor
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


__all__ = [
    'LinearBlock',
    'FeedForwardNetwork',
    'DuelingQNetwork',
    'PolicyNetwork',
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
        output_activation: Optional[type[nn.Module]] = None,
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
                activation = nn.Identity if output_activation is None else output_activation
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


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        *dims,
        activation: nn.Module = nn.ReLU,
        normalization: nn.Module = nn.LayerNorm,
        device: torch.device,
        output_activation: type[nn.Module] = nn.Tanh,
        categorical: bool,
        deterministic: bool = False,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("Need at least 2 dimensions to build a minimal model.")
        if any(d < 1 for d in dims):
            raise ValueError("Model dimensions must be strictly positive.")

        self.net = FeedForwardNetwork(
            *dims, activation=activation, normalization=normalization, device=device, output_activation=output_activation
        )
        self.device = device
        self.categorical = categorical
        self.deterministic = deterministic

        if not categorical:
            _action_space = dims[-1]
            self.log_std = torch.nn.Parameter(-0.5 * torch.ones(_action_space, dtype=torch.float32, device=device))
            del _action_space

    def forward(
        self,
        x: Tensor,
        probs: bool = False,
        action: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Approximate action logits given an environment observation.

        If the network is categorical (as defined by ``self.categorical``), it approximates the logits of each action,
        and the action is subsequently sampled from a multimodal distribution using those logits. Otherwise, the mean of
        a Gaussian is approximated and the action sampled from the Gaussian distribution.

        .. Args::
            - x (Tensor): A batch of observations.
            - probs (bool): If ``True``, the action logits are probabilities.
            - action (Tensor, optional): A batch of actions.
        """
        if self.categorical:
            logits = self.net(x)
            pi = Categorical(logits=logits)
            if action is None:
                action = pi.rsample()
            if probs:
                return action, pi.log_prob(action)
            return action.unsqueeze(-1)
        else:
            mu = self.net(x)
            if self.deterministic:
                return mu
            std = torch.exp(self.log_std)
            pi = Normal(mu, std)
            if action is None:
                action = pi.rsample()
            if probs:
                return action, pi.log_prob(action).sum(dim=-1)
            return action

    @property
    def model_type(self) -> str:
        return self.__class__.__name__

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters())


class DoubledNetwork(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.net_1 = base
        self.net_2 = base

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.net_1(x), self.net_2(x)

    def q1(self, x: Tensor) -> Tensor:
        return self.net_1(x)

    @property
    def model_type(self) -> str:
        return self.__class__.__name__

    def parameter_count(self) -> int:
        return sum(
            net.parameter_count for net in (self.net_1, self.net_2)
        )
