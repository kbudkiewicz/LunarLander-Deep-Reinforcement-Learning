import os
import random
import copy

import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Tuple
from torch import Tensor
from collections import deque, namedtuple


__all__ = [
    'ReplayMemory',
    'DeepQNetwork',
    'DoubleDQN',
]


memory = namedtuple('Memory', ('s', 'a', 'r', 'next_s', 'term'))


class ReplayMemory(object):
    """Replay Memory class as described in `Human-level control through deep reinforcement learning
    <https://www.nature.com/articles/nature14236>`_."""
    def __init__(self, memory_size: int, batch_size: int):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def remember(self, *args) -> None:
        self.memory.append(memory(*args))

    def get_sample(self, device) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_new, flags = map(np.array, zip(*samples))
        states = torch.tensor(states, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        states_new = torch.tensor(states_new, device=device, dtype=torch.float)
        flags = torch.tensor(flags, device=device, dtype=torch.long)
        actions = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)

        return states, actions, rewards, states_new, flags

    def __len__(self) -> int:
        return len(self.memory)


class Agent(ABC):
    def __init__(
        self,
        local: torch.nn.Module,
        target: torch.nn.Module,
        device: torch.device,
        action_space: int,
        criterion: torch.nn.Module,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_norm: float = 10.,
        inference_only: bool = False,
        net_update_freq: int = 6,
        tau: float = 2.5e-3,        # soft parameter update constant
        gamma: float = 0.99,        # (reward) discount factor
        eps_start: float = 0.7,     # starting epsilon value
        eps_end: float = 0.05,      # final epsilon value
        eps_term: int = 300,        # episode at which eps_end is reached
        replay_memory_size: int = int(1e5),
    ):
        super().__init__()
        self.local = local
        self.target = target.eval()
        self.device = device
        self.batch_size = batch_size
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(replay_memory_size, batch_size)
        self.criterion = criterion
        self.action_space = action_space
        self.net_update_freq = net_update_freq
        self.tau = tau
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_term = eps_term
        self.epsilon = self.eps_start
        self.t_step = 0

        if not isinstance(replay_memory_size, int) or not (0 < replay_memory_size):
            raise ValueError(f'The replay memory size must be an integer greater than 0.')
        if not (0 < action_space):
            raise ValueError(f'The action space must be a strictly positive number.')
        if not isinstance(net_update_freq, int) or not (0 < net_update_freq):
            raise ValueError(f'The network update frequency must be an integer greater than 0.')
        if not (0 < tau < 1):
            raise ValueError(f'Tau must be between 0 and 1.')
        if not (eps_end <= eps_start):
            raise ValueError(f'The final epsilon value must be smaller or equal to the initial one.')
        if not isinstance(eps_term, int) or not (0 < eps_term):
            raise ValueError(f'The epsilon termination episode must be an integer greater than 0.')

        if not inference_only:
            self.optimizer = torch.optim.Adam(self.local.parameters(), lr=lr, weight_decay=weight_decay)
            self.max_norm = max_norm
            self.memory = ReplayMemory(replay_memory_size, batch_size)

    def __call__(self, observation: np.array) -> np.array:
        """Select an action from the actor based on the current state.

        Given a state, use :math:`\epsilon`-greedy action selection to return a random action or the one approximated by
        the local network (see `Epsilon-Greedy action selection`_).

        The forward pass is performed with no_grad().

        Args:
            observation (np.array): Current state of the environment.
        Return:
            action (np.array): Action chosen by the agent.

        .. _Epsilon-Greedy action selection: https://en.wikipedia.org/wiki/Multi-armed_bandit
        """

        if random.random() > self.epsilon:
            with torch.no_grad():
                observation = torch.from_numpy(observation).to(self.device)
                action_values = self.local(observation)
                return torch.argmax(action_values).item()
        else:
            return random.randint(0, self.action_space - 1)

    def memorize(self, *args) -> Union[Tuple[float, float], Tuple[None, None]]:
        """
        Save the states, action, rewards and environment flags to agent's ``ReplayMemory``.

        Args:
            *args: A tuple (s, a, r, next_s, terminated)
        """
        self.memory.remember(*args)
        self.t_step += 1
        if (self.t_step % self.net_update_freq == 0) and (self.memory.__len__() >= self.batch_size):
            loss, grad_norm = self.update_net()
            return loss, grad_norm
        return None, None

    def update_weights(self) -> None:
        """Perform a soft parameter update of network weights."""
        for target_param, local_param in zip(self.target.parameters(), self.local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)

    def backpropagate(self, input: Tensor, target: Tensor) -> Tuple[float, float]:
        """Calculate the loss based on the chosen criterion and backpropagate them throught the network"""
        self.optimizer.zero_grad()
        loss = self.criterion(input, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.local.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        self.update_weights()
        return loss.item(), grad_norm.item()

    @abstractmethod
    def update_net(self) -> Tuple[float, float]:
        """Subclasses must implement their own target computation and call self.backpropagate(). Must return the loss
        and the gradient norm."""

    def update_epsilon(self, epoch: int) -> None:
        r"""Calculates a new :math:`\epsilon` for each successive epoch following a predefined rate schedule.

        .. Args::
            current_episode: current training episode

        .. Return::
            loss (float)
        """
        slope = (self.eps_end - self.eps_start) / self.eps_term
        new_eps = slope * epoch + self.eps_start
        self.epsilon = max(self.eps_end, new_eps)

    def zero_epsilon(self) -> None:
        self.epsilon = 0.
        self.eps_start = 0.
        self.eps_end = 0.
        self.eps_term = 0.

    def save_state_dict(self, path_to_dir: Union[str, os.PathLike], get_path: bool = False) -> Union[str, None]:
        path_local = os.path.join(path_to_dir, 'local.pt')
        torch.save(self.local.state_dict(), path_local)
        path_target = os.path.join(path_to_dir, 'target.pt')
        torch.save(self.target.state_dict(), path_target)

        if get_path:
            return os.path.dirname(os.path.abspath(path_local))

    def load_state_dict(self, path_local: str, path_target: str) -> None:
        self.local.load_state_dict(torch.load(path_local))
        self.target.load_state_dict(torch.load(path_target))

    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

    def get_config(self) -> dict:
        raise NotImplementedError

    def get_optimizer_config(self) -> dict:
        return {
            'optimizer.type': self.optimizer.__class__.__name__,
            'optimizer.lr': self.optimizer.defaults['lr'],
            'optimizer.betas': self.optimizer.defaults['betas'],
            'optimizer.weight_decay': self.optimizer.defaults['weight_decay'],
            'optimizer.max_norm': self.max_norm,
        }


class DeepQNetwork(Agent):
    """A Deep Q-learning Network (DQN) agent based on `Human-level control through deep reinforcement learning
    <https://www.nature.com/articles/nature14236>`_."""
    def __init__(
        self,
        local: torch.nn.Module,
        target: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        **kwargs,
    ):
        super().__init__(
            local=local,
            target=target,
            device=device,
            criterion=criterion,
            action_space=action_space,
            **kwargs
        )

    def update_net(self) -> Tuple[float, float]:
        state, action, reward, state_new, flags = self.memory.get_sample(device=self.device)

        with torch.no_grad():
            q_target = reward + self.gamma * torch.max(self.target(state_new), dim=1)[0] * (1 - flags)
        q_local = self.local(state).gather(1, action).squeeze()
        loss, grad_norm = self.backpropagate(q_local, q_target)

        return loss, grad_norm


class DoubleDQN(Agent):
    """Double Deep Q-Learning Network based on `Deep Reinforcement Learning with Double Q-learning
    <https://arxiv.org/abs/1509.06461>`_."""

    def __init__(
        self,
        local: torch.nn.Module,
        target: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        **kwargs,
    ):
        super().__init__(
            local=local,
            target=target,
            device=device,
            criterion=criterion,
            action_space=action_space,
            **kwargs
        )

    def update_net(self) -> Tuple[float, float]:
        """Perform soft parameter update based on a sample from replay memory.

        The Q-value is approximated via a local and a target network parametrized by respectively :math:`\Theta_t` and
        :math:`\Theta_t'`, and is defined as:

        .. math::

            Q(s,a) = R_{t+1} + \gamma Q(S_{t+1}, \\argmax_a Q(S_{t+1}, a), \Theta_t; \Theta_t')
        """
        state, action, reward, state_new, flags = self.memory.get_sample(device=self.device)

        with torch.no_grad():
            argmax_a = torch.argmax(self.local(state_new), dim=1).unsqueeze(1)  # B, 1
            q_target = reward + self.gamma * self.target(state_new).gather(1, argmax_a).squeeze(1) * (1 - flags)
        q_local = self.local(state).gather(1, action).squeeze()
        loss, grad_norm = self.backpropagate(q_local, q_target)

        return loss, grad_norm

