import os
import random
import copy

import torch
import torch.nn as nn
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler, ExponentialLR
from collections import deque, namedtuple
from nn import DuelingQNetwork, PolicyNetwork

__all__ = [
    'ReplayMemory',
    'DeepQNetwork',
    'DoubleDQN',
    'DuelingDQN',
    'ValueAgent',
    'PolicyAgent',
    'DDPG',
]


memory = namedtuple('Memory', ('s', 'a', 'r', 'next_s', 'term'))


class ReplayMemory(object):
    """Replay Memory class as described in `Human-level control through deep reinforcement learning
    <https://www.nature.com/articles/nature14236>`_."""
    def __init__(self, memory_size: int, batch_size: int, continuous: bool = False):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.continuous = continuous

    def remember(self, *args) -> None:
        self.memory.append(memory(*args))

    def get_sample(self, device) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_new, flags = map(np.array, zip(*samples))
        states = torch.tensor(states, device=device, dtype=torch.float)
        rewards = torch.tensor(rewards, device=device, dtype=torch.float)
        states_new = torch.tensor(states_new, device=device, dtype=torch.float)
        flags = torch.tensor(flags, device=device, dtype=torch.long)
        if self.continuous:
            actions = torch.tensor(actions, device=device, dtype=torch.float)
        else:
            actions = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(1)

        return states, actions, rewards, states_new, flags

    def __len__(self) -> int:
        return len(self.memory)


class Agent(ABC):
    """Root agent class that must be subclassed by higher-level agents.

    Attributes:
        batch_size (int): Batch size
        lr (float): Learning rate
        weight_decay (float): Weight decay coefficient
        max_norm (float): Maximal grad norm. Values exceeding it will be clipped before being passed to the optimizer
        net_update_freq (int): Frequency of soft (Polyakof) parameter update between the local and target networks
        tau (float): Coefficient of the soft parameter update
        gamma (float): Reward discount factor
        warmup (int): Number of episodes before the network update will be performed for the first time.
    """
    def __init__(
        self,
        device: torch.device,
        action_space: int,
        criterion: torch.nn.Module,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_norm: float = 10.,
        net_update_freq: int = 6,
        tau: float = 2.5e-3,
        gamma: float = 0.99,
        warmup: int = 1,
    ):
        super().__init__()
        self.device = device
        self.action_space = action_space
        self.criterion = criterion
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_norm = max_norm
        self.net_update_freq = net_update_freq
        self.tau = tau
        self.gamma = gamma
        self.warmup = warmup
        self.t_step = 0

        if not (0 < action_space):
            raise ValueError(f'The action space must be a strictly positive number.')
        if not isinstance(net_update_freq, int) or not (0 < net_update_freq):
            raise ValueError(f'The network update frequency must be an integer greater than 0.')
        if not (0 < tau < 1):
            raise ValueError(f'Tau must be between 0 and 1.')

    @abstractmethod
    def __call__(self, observation: np.array) -> np.array:
        """Sample an action given some observation sampled from the environment."""
        raise NotImplementedError

    @abstractmethod
    def update_weights(self) -> None:
        """Perform a soft parameter update of network weights."""
        raise NotImplementedError

    @abstractmethod
    def backpropagate(self, *args) -> Tuple[float, float]:
        """Calculate the loss based on the chosen criterion and backpropagate them throught the network"""
        raise NotImplementedError

    @abstractmethod
    def update_net(self) -> Tuple[float, float, float]:
        """Subclasses must implement their own target computation and call self.backpropagate(). Must return the loss
        and the gradient norm."""
        raise NotImplementedError

    @abstractmethod
    def get_optimizer_config(self) -> dict:
        raise NotImplementedError

    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

    def get_config(self) -> dict:
        raise NotImplementedError


class ValueAgent(Agent):
    """Base class for value-based reinforcement learning agents.

    Attributes:
        batch_size (int): Batch size
        lr (float): Learning rate
        weight_decay (float): Weight decay coefficient
        max_norm (float): Maximal grad norm. Values exceeding it will be clipped before being passed to the optimizer
        inference_only (bool): Flag . If True, the optimizer and replay memory are not
            instantiated within the ``Agent``.
        net_update_freq (int): Frequency of soft (Polyakof) parameter update between the local and target networks
        tau (float): Coefficient of the soft parameter update
        gamma (float): Reward discount factor
        eps_start (float): Initial epsilon value
        eps_end (float): Final epsilon value
        eps_term (int): The episode number at which the final epsilon value is reached
        replay_memory_size (int): The size of replay memory
    """
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        action_space: int,
        criterion: torch.nn.Module,
        replay_memory_size: int = int(1e5),
        eps_start: float = 0.7,  # starting epsilon value
        eps_end: float = 0.05,  # final epsilon value
        eps_term: int = 300,  # episode at which eps_end is reached
        inference_only: bool = False,
        **kwargs
    ):
        super().__init__(
            device=device,
            action_space=action_space,
            criterion=criterion,
            **kwargs
        )
        self.local = model
        self.target = copy.deepcopy(model).eval()
        self.device = device
        self.action_space = action_space
        self.criterion = criterion
        self.replay_memory_size = replay_memory_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_term = eps_term
        self.epsilon = self.eps_start
        self.inference_only = inference_only

        if not isinstance(replay_memory_size, int) or not (0 < replay_memory_size):
            raise ValueError(f'The replay memory size must be an integer greater than 0.')
        if not (eps_end <= eps_start):
            raise ValueError(f'The final epsilon value must be smaller or equal to the initial one.')
        if not isinstance(eps_term, int) or not (0 < eps_term):
            raise ValueError(f'The epsilon termination episode must be an integer greater than 0.')
        if not isinstance(self.warmup, int) or not (0 < self.warmup):
            raise ValueError(f'The warmup value must be an integer greater than 0.')

        if not self.inference_only:
            self.optimizer = torch.optim.Adam(self.local.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.memory = ReplayMemory(self.replay_memory_size, self.batch_size)

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

    def memorize(self, *args) -> Union[Tuple[float, float, float], Tuple[None, None, None]]:
        """
        Save the states, action, rewards and environment flags to agent's ``ReplayMemory``.

        Args:
            *args: A tuple (s, a, r, next_s, terminated)
        """
        self.memory.remember(*args)
        self.t_step += 1
        if (
            self.t_step >= self.warmup
            and self.t_step % self.net_update_freq == 0
            and self.memory.__len__() >= self.batch_size
        ):
            loss, grad_norm, q_value = self.update_net()
            return loss, grad_norm, q_value
        return None, None, None

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

    @abstractmethod
    def update_net(self) -> Tuple[float, float, float]:
        """Update the local and target network parameters and return the loss, gradient norm and the average Q-value
        estimate."""
        raise NotImplementedError

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

    def get_optimizer_config(self) -> dict:
        return {
            'optimizer.type': self.optimizer.__class__.__name__,
            'optimizer.lr': self.optimizer.defaults['lr'],
            'optimizer.betas': self.optimizer.defaults['betas'],
            'optimizer.weight_decay': self.optimizer.defaults['weight_decay'],
            'optimizer.max_norm': self.max_norm,
        }


class PolicyAgent(Agent):
    """Base class for policy-based reinforcement learning agents. Consist of an actor and a critic networks with
    individual optimizers."""
    def __init__(
        self,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        action_range: np.ndarray,
        categorical: bool,
        inference_only: bool = False,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        scheduler: Optional[type[LRScheduler]] = None,
        replay_memory_size: Optional[int] = None,
        continuous: Optional[bool] = None,
        **kwargs
    ):
        super().__init__(
            device=device,
            action_space=action_space,
            criterion=criterion,
            **kwargs
        )
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr = lr_critic     # TODO: temporary fix

        if not isinstance(replay_memory_size, int) or not (0 < replay_memory_size):
            raise ValueError(f'The replay memory size must be an integer greater than 0.')
        if not (0 < action_space):
            raise ValueError(f'The action space must be a strictly positive number.')

        self.action_space = action_space
        self.action_range = action_range
        self.use_entropy = False
        self.categorical = categorical

        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(actor).requires_grad_(False)
        self.critic_target = copy.deepcopy(critic).requires_grad_(False)
        self.device = device
        self.criterion = criterion
        self.actor_scheduler = None
        self.critic_scheduler = None

        if not inference_only:
            if replay_memory_size is not None:
                assert continuous is not None
                self.replay_memory_size = replay_memory_size
                self.memory = ReplayMemory(replay_memory_size, self.batch_size, continuous=continuous)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
            if scheduler is not None and issubclass(scheduler, LRScheduler):
                self.actor_scheduler = ExponentialLR(self.actor_optimizer, gamma=0.9999)
                self.critic_scheduler = ExponentialLR(self.critic_optimizer, gamma=0.9999)

    @abstractmethod
    def __call__(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update_epsilon(self, epoch: int) -> None:
        pass

    def zero_epsilon(self) -> None:
        pass

    def _rescale_action(self, action: Tensor, is_tensor: bool = False) -> Union[np.ndarray, Tensor]:
        """Clamp continuous action values to be within the allowed action range."""
        action = torch.clamp(action, -1, 1)
        lows, highs = self.action_range

        if is_tensor:
            lows = torch.as_tensor(lows, device=self.device, dtype=action.dtype)
            highs = torch.as_tensor(highs, device=self.device, dtype=action.dtype)
        else:
            action = action.cpu().detach().numpy()

        action = lows + (action + 1.) * 0.5 * (highs - lows)
        return action

    def sample_critic(self, observation: np.ndarray) -> np.ndarray:
        """Return the predicted value of the critic network."""
        with torch.no_grad():
            observation = torch.from_numpy(observation).to(self.device)
            return self.critic(observation)

    def sample_actor(self, observation: np.ndarray) -> np.ndarray:
        """Return the predicted value of the actor network."""
        with torch.no_grad():
            observation = torch.from_numpy(observation).to(self.device)
            return self.actor(observation)

    @abstractmethod
    def update_net(self) -> Tuple[float, float, float]:
        raise NotImplementedError

    def update_weights(self) -> None:
        """Perform a soft parameter update of network weights."""
        for target_param, local_param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)
        for target_param, local_param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)

    def critic_loss(self, input: Tensor, target: Tensor) -> Tuple[float, float]:
        """Calculate the loss and the gradient norm for the critic."""
        self.critic_optimizer.zero_grad()
        loss = self.criterion(input, target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.max_norm)
        self.critic_optimizer.step()
        if isinstance(self.critic_scheduler, LRScheduler):
            self.critic_scheduler.step()
            self.lr_critic = self.critic_scheduler.get_last_lr()[0]
        return loss.item(), grad_norm.item()

    @abstractmethod
    def actor_loss(self, *args, **kwargs) -> Tuple[float, float]:
        """Calculate the loss and the gradient norm for the actor."""
        raise NotImplementedError

    def backpropagate(self, value: Tensor, value_target: Tensor, state: Tensor) -> Tuple[float, float, float, float]:
        """Calculate the loss and backpropagate it through the actor and critic."""
        critic_loss, critic_grad_norm = self.critic_loss(value, value_target)
        actor_loss, actor_grad_norm = self.actor_loss(state)
        self.update_weights()
        return critic_loss, critic_grad_norm, actor_loss, actor_grad_norm

    @property
    def agent_type(self) -> str:
        return self.__class__.__name__

    def get_config(self) -> dict:
        raise NotImplementedError

    def get_optimizer_config(self) -> dict:
        return {
            'optimizer.actor.type': self.actor_optimizer.__class__.__name__,
            'optimizer.actor.lr': self.actor_optimizer.defaults['lr'],
            'optimizer.actor.betas': self.actor_optimizer.defaults['betas'],
            'optimizer.actor.weight_decay': self.actor_optimizer.defaults['weight_decay'],
            'optimizer.actor.max_norm': self.max_norm,
            'optimizer.critic.type': self.critic_optimizer.__class__.__name__,
            'optimizer.critic.lr': self.critic_optimizer.defaults['lr'],
            'optimizer.critic.betas': self.critic_optimizer.defaults['betas'],
            'optimizer.critic.weight_decay': self.critic_optimizer.defaults['weight_decay'],
            'optimizer.critic.max_norm': self.max_norm,
        }


class DeepQNetwork(ValueAgent):
    """A Deep Q-learning Network (DQN) agent based on `Human-level control through deep reinforcement learning
    <https://www.nature.com/articles/nature14236>`_."""
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        **kwargs,
    ):
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            action_space=action_space,
            **kwargs
        )

    def update_net(self) -> Tuple[float, float, float]:
        state, action, reward, state_new, flags = self.memory.get_sample(device=self.device)

        with torch.no_grad():
            q_target = reward + self.gamma * torch.max(self.target(state_new), dim=1)[0] * (1 - flags)
        q_local = self.local(state).gather(1, action).squeeze()
        loss, grad_norm = self.backpropagate(q_local, q_target)

        return loss, grad_norm, q_local.mean().item()


class DoubleDQN(ValueAgent):
    """Double Deep Q-Learning Network based on `Deep Reinforcement Learning with Double Q-learning
    <https://arxiv.org/abs/1509.06461>`_."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        **kwargs,
    ):
        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            action_space=action_space,
            **kwargs
        )

    def update_net(self) -> Tuple[float, float, float]:
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

        return loss, grad_norm, q_local.mean().item()


class DuelingDQN(ValueAgent):
    """Dueling Network based on `Dueling Network Architectures for Deep Reinforcement Learning
    <https://arxiv.org/abs/1511.06581>`_.

    The Q-value target can be defined as a maximum of the approximated Q-value (DQN) or the argmax of the approximated
    Q-value across actions (Double DQN) via the ``use_double`` attribute.
    """
    def __init__(
        self,
        model: DuelingQNetwork,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        use_double: bool = True,
        **kwargs,
    ):
        if not isinstance(model, DuelingQNetwork):
            raise AttributeError(
                f"Value and advantage networks must be a DuelingQNetwork class, but are {type(model)}."
            )

        super().__init__(
            model=model,
            device=device,
            criterion=criterion,
            action_space=action_space,
            **kwargs
        )
        self.use_double = use_double

    def update_net(self) -> Tuple[float, float, float]:
        """Perform soft parameter update based on a sample from replay memory.

        If ``use_double`` is ``True``, then the Q-value target is defined as:

        .. math::
            Q(s,a) = R_{t+1} + \gamma Q(S_{t+1}, \argmax_a Q(S_{t+1}, a), \Theta_t; \Theta_t').

        Otherwise, the standard DQN target is used:

        .. math::
            Q(s,a) = R_{t+1} + \gamma \max_a Q(S_{t+1}, a), \Theta_t; \Theta_t')
        """
        state, action, reward, state_new, flags = self.memory.get_sample(device=self.device)

        with torch.no_grad():
            if self.use_double:
                argmax_a = torch.argmax(self.local(state_new), dim=1).unsqueeze(1)  # B, 1
                q_target = reward + self.gamma * self.target(state_new).gather(1, argmax_a).squeeze(1) * (1 - flags)
            else:
                q_target = reward + self.gamma * torch.max(self.target(state_new), dim=1)[0] * (1 - flags)
        q_local = self.local(state).gather(1, action).squeeze()
        loss, grad_norm = self.backpropagate(q_local, q_target)

        return loss, grad_norm, q_local.mean().item()


class DDPG(PolicyAgent):
    """Deep Deterministic Policy Gradient (DDPG) Agent based on
    `Continuous Control with Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_."""
    def __init__(
        self,
        actor: PolicyNetwork,
        critic: torch.nn.Module,
        device: torch.device,
        criterion: torch.nn.Module,
        action_space: int,
        replay_memory_size: int = int(1e6),
        noise_range: Tuple[float, float] = (0.20, 0.005),
        noise_term: int = 400,
        **kwargs,
    ):
        self.noise_start, self.noise_end = noise_range
        self.noise_scale = self.noise_start
        self.noise_term = noise_term

        super().__init__(
            actor=actor,
            critic=critic,
            device=device,
            criterion=criterion,
            action_space=action_space,
            replay_memory_size=replay_memory_size,
            continuous=True,
            **kwargs
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            observation = torch.from_numpy(observation).to(self.device)
            action = self.actor(observation)
            if self.noise_scale > 0:
                action += self.noise_scale * torch.randn_like(action)
            action = self._rescale_action(action)
            return action

    def memorize(self, *args) -> Union[Tuple[float, float, float], Tuple[None, None, None]]:
        """Save the states, action, rewards and environment flags to agent's ``ReplayMemory``.

        Args:
            *args: A tuple (s, a, r, next_s, terminated)
        """
        self.memory.remember(*args)
        self.t_step += 1
        if (
            self.t_step >= self.warmup
            and self.t_step % self.net_update_freq == 0
            and self.memory.__len__() >= self.batch_size
        ):
            loss, grad_norm, q_value = self.update_net()
            return loss, grad_norm, q_value
        return None, None, None

    def update_noise(self, epoch: int) -> None:
        r"""Calculates a new noise gain for each successive epoch following a predefined rate schedule."""
        slope = (self.noise_end - self.noise_start) / self.noise_term
        noise_new = slope * epoch + self.noise_start
        self.noise_scale = max(self.noise_end, noise_new)

    def actor_loss(self, state: Tensor) -> Tuple[float, float]:
        self.actor_optimizer.zero_grad()
        actions = self.actor(state)
        actions = self._rescale_action(actions, is_tensor=True)
        state_action_pair = torch.hstack((state, actions))

        self.critic.requires_grad_(False)
        loss = -self.critic(state_action_pair).mean()
        loss.backward()
        self.critic.requires_grad_(True)

        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_norm)
        self.actor_optimizer.step()
        return loss.item(), grad_norm.item()

    def update_net(self) -> Tuple[float, float, float]:
        state, action, reward, state_new, flags = self.memory.get_sample(device=self.device)

        with torch.no_grad():
            action_new = self.actor_target(state_new)
            action_new = self._rescale_action(action_new, is_tensor=True)
            state_action_pair_new = torch.hstack((state_new, action_new))

            q = self.critic_target(state_action_pair_new).squeeze(-1)
            q_target = reward + self.gamma * q * (1 - flags)

        state_action_pair = torch.hstack((state, action))
        q_local = self.critic(state_action_pair).squeeze(-1)
        critic_loss, critic_grad_norm, actor_loss, actor_grad_norm = self.backpropagate(
            value=q_local, value_target=q_target, state=state
        )

        return critic_loss, critic_grad_norm, q_local.mean().item()
