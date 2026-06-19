import os
import random
import copy
import torch
import torch.nn as nn
import numpy as np

from abc import abstractmethod
from typing import Union, Tuple
from torch import Tensor
from collections import deque, namedtuple
from nn import FeedForwardNetwork

# defining memory instance
memory = namedtuple('Memory', ('s', 'a', 'r', 'next_s', 'term'))


class ReplayMemory(object):
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


class AgentConfig:
    """
    Hyperparameters for the RL agent. Contains both agent, as well as network hyperparameters.
    """
    state_space: int = 8
    action_space: int = 4
    memory_size: int = 100_000
    t_step: int = 0
    batch_size: int = 64
    tau: float = 2.5e-3     # soft parameter update constant
    gamma: float = 0.99     # discount factor
    lr: float = 1e-3
    net_update_freq: int = 6
    eps_start: float = 0.7  # starting epsilon value
    eps_end: float = 0.05   # final epsilon value
    eps_term: int = 200     # episode # at which eps_end is reached4
    loss: float = float('inf')


class Agent(AgentConfig):
    def __init__(
        self,
        *dims,
        device,
    ):
        super().__init__()
        self.eps = self.eps_start
        # FIXME: replacing local and target with module and its deepcopy breaks the agent
        self.qnet_local = FeedForwardNetwork(*dims, device=device)
        self.qnet_target = FeedForwardNetwork(*dims, device=device).eval()
        self.device = device
        self.optimizer = torch.optim.Adam(self.qnet_local.parameters(), self.lr)
        self.memory = ReplayMemory(self.memory_size, self.batch_size)

    def __call__(self, observation: np.array) -> np.array:
        """
        Return the best or a random action from the environment given some observation. The probability of getting
        the best vs. a random action is given by the value of :math:`\epsilon` (see `Epsilon-Greedy action selection`_).

        The forward pass is performed with no_grad().

        :param observation: a vector describing the current state of the environment
        :return:    Action [int]

        .. _Epsilon-Greedy action selection: https://en.wikipedia.org/wiki/Multi-armed_bandit
        """

        if random.random() > self.eps:
            with torch.no_grad():
                observation = torch.from_numpy(observation).to(self.device)
                action_values = self.qnet_local(observation)
                return torch.argmax(action_values).item()
        else:
            return random.randint(0, 2)

    def memorize(self, *args) -> Union[float, None]:
        """
        Save SARS to agent's ``ReplayMemory``.

        Args:
            *args: A vector containing (s, a, r, next_s, terminated).
        """
        self.memory.remember(*args)
        self.t_step += 1
        if (self.t_step % self.net_update_freq == 0) and (self.memory.__len__() >= self.batch_size):
            loss = self.update_net()
            return loss
        return None

    def backprop(
        self,
        predicted: Tensor,
        target: Tensor,
        criterion: torch.nn.Module = torch.nn.MSELoss(),
    ) -> Union[float]:
        """
        Perform a backpropagation step.

        Args:
            predicted (Tensor): Predicted values
            target (Tensor): Actual values
            criterion (Callable): Loss function
            do_return (bool): Return the loss value. Default is True.
        """
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(self.qnet_local.parameters(), max_norm=1.)
        loss = criterion(predicted, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @abstractmethod
    def policy_update(
        self,
        sars: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """User-defined function to calculate the Q-values.

        .. Args::
            - sars (Tuple of Tensor): Tuple of batched State, Action, Reward, State'
        """

    def update_net(self) -> float:
        state, action, reward, state_new, flags = self.memory.get_sample(device=self.device)

        # Use Bellman equation to calculate the Q-values
        q = self.qnet_target(state_new)
        q_target = reward + self.gamma * torch.max(q, dim=1)[0] * (1 - flags)  # q_target
        q_local = self.qnet_local(state).gather(1, action).squeeze()  # current q
        loss = self.backprop(q_local, q_target)

        # soft parameter update
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)

        return loss

    def update_epsilon(self, epoch: int) -> None:
        r"""Calculates a new :math:`\epsilon` for each successive epoch following a predefined rate schedule.

        .. Args::
            current_episode: current training episode

        .. Return::
            loss (float)
        """
        slope = (self.eps_end - self.eps_start) / self.eps_term
        new_eps = slope * epoch + self.eps_start
        self.eps = max(self.eps_end, new_eps)

    def save_state_dict(self, path_to_dir: os.PathLike = './model_params'):
        path = os.path.join(path_to_dir, "local.pt")
        torch.save(self.qnet_local.state_dict(), path)
        path = os.path.join(path_to_dir, "target.pt")
        torch.save(self.qnet_target.state_dict(), path)

    def load_state_dict(self, path_local, path_target):
        self.qnet_local.load_state_dict(torch.load(path_local))
        self.qnet_target.load_state_dict(torch.load(path_target))

    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

