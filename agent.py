import os
import random
import numpy as np
import torch
from torch import Tensor
from configs import AgentConfig
from collections import deque, namedtuple
from networks import FeedForwardNetwork

# defining memory instance
memory = namedtuple('Memory', ('s', 'a', 'r', 'next_s', 'term'))


class ReplayMemory(object):
    def __init__(self, memory_size: int, batch_size: int):
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def remember(self, *args):
        self.memory.append(memory(*args))

    def get_sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(AgentConfig):
    def __init__(self):
        super().__init__()
        self.eps = self.eps_start
        self.qnet_local = FeedForwardNetwork(8,64,64,4)
        self.qnet_target = FeedForwardNetwork(8,64,64,4).eval()
        self.optimizer = optim.Adam(self.qnet_local.parameters(), self.lr)
        self.memory = ReplayMemory(self.memory_size, self.batch_size)
        self.s_tens = torch.tensor(np.zeros((self.batch_size, 8))).float()
        self.a_tens = torch.tensor(range(self.batch_size)).unsqueeze(1).long()
        self.r_tens = torch.tensor(range(self.batch_size)).float()
        self.s_next_tens = torch.tensor(np.zeros((self.batch_size, 8))).float()
        self.term_tens = torch.tensor(range(self.batch_size)).long()

        # pre-allocation
        self.s_tens = torch.zeros([self.batch_size, 8], device=self.device).float()
        self.s_next_tens = torch.zeros_like(self.s_tens, device=self.device)
        self.a_tens = torch.tensor(range(self.batch_size), device=self.device).unsqueeze(1).long()
        self.r_tens = torch.tensor(range(self.batch_size), device=self.device)
        self.term_tens = torch.tensor(range(self.batch_size), device=self.device)

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
            observation = torch.from_numpy(observation).requires_grad_(False).to(self.device)
            with torch.no_grad():
                action_values = self.qnet_local.forward(observation)
            return torch.argmax(action_values).cpu().detach().numpy()
        else:
            return torch.randint(size=[1], low=0, high=3).item()

    def memorize(self, *args):
        """
        Save SARS to agent's ``ReplayMemory``.

        Args:
            *args: A vector containing (s, a, r, next_s, terminated).
        """
        self.memory.remember(*args)
        self.t_step += 1
        if (self.t_step % self.net_update_freq == 0) and (self.memory.__len__() >= self.batch_size):
            self.update_net()

    def backprop(self, x0: Tensor, x1: Tensor,
                 criterion: torch.nn.Module = torch.nn.MSELoss(),
                 do_return: bool = True) -> Tensor | None:
        """
        Perform a backpropagation step.

    def update_net(self, exp: namedtuple):
        # unpack memories into a tensor/vector with states, actions, or rewards
        # attach an argument of named tuple from each memory
        for i in range(len(exp)):
            self.s_tens[i] = torch.tensor(exp[i].s, device=self.device)
            self.a_tens[i] = torch.tensor(exp[i].a, device=self.device)
            self.r_tens[i] = torch.tensor(exp[i].r, device=self.device)
            self.s_next_tens[i] = torch.tensor(exp[i].next_s, device=self.device)
            self.term_tens[i] = torch.tensor(exp[i].term, device=self.device)

        # Bellman equation. Calculating q_target and and current q_value
        q = self.qnet_target(self.s_next_tens)  # get q_values of next states
        q_target = self.r_tens + self.gamma * torch.max(q, dim=1)[0] * (1 - self.term_tens)  # q_target
        q_local = self.qnet_local(self.s_tens).gather(1, self.a_tens).squeeze()  # current q

        self.loss = self.backprop(q_local, q_target)

        # update network parameters
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)

    def update_epsilon(self, current_episode: int) -> None:
        """
        Calculates the new :math:`\epsilon` value for each successive :code:`episode`. This function is equivalent to
        a learning rate scheduler.

        Args:
            current_episode: current training episode
        """
        slope = (self.eps_end - self.eps_start) / self.eps_term
        new_eps = slope * current_episode + self.eps_start
        self.eps = max(self.eps_end, new_eps)

    def save_state_dict(self, path_to_dir: os.PathLike = './model_params'):
        path = os.path.join(path_to_dir, "local.pt")
        torch.save(self.qnet_local.state_dict(), path)
        path = os.path.join(path_to_dir, "target.pt")
        torch.save(self.qnet_target.state_dict(), path)

    def load_state_dict(self, path_local, path_target):
        self.qnet_local.load_state_dict(torch.load(path_local))
        self.qnet_target.load_state_dict(torch.load(path_target))
