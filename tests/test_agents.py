import pytest
import torch
import gymnasium as gym
import numpy as np
import copy

from typing import Tuple
from torch.nn import Module
from nn import FeedForwardNetwork, DuelingQNetwork
from agent import Agent, DeepQNetwork, DoubleDQN, DuelingDQN


class BaseAgentTest:
    @pytest.fixture(params=[0, 1, 42, 12345])
    def seed(self, request) -> int:
        """Set seed for test reproducibility."""
        return request.param

    @pytest.fixture(autouse=True)
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    @pytest.fixture(scope='module')
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @pytest.fixture
    def env_id(self) -> str:
        return 'LunarLander-v3'

    @pytest.fixture
    def max_episode_steps(self) -> int:
        return 20

    @pytest.fixture
    def env(self, env_id, max_episode_steps) -> gym.Env:
        """Build a ``gymnasium`` environment."""
        environment = gym.make(id=env_id, max_episode_steps=max_episode_steps)
        yield environment
        environment.close()

    @pytest.fixture
    def state_space(self, env: gym.Env) -> Tuple[int, ...]:
        return env.observation_space.shape

    @pytest.fixture
    def action_space(self, env: gym.Env) -> int:
        return env.action_space.n

    @pytest.fixture
    def dims(self, state_space, action_space) -> tuple[int, ...]:
        return *state_space, np.random.randint(2, 32), np.random.randint(2, 32), action_space

    @pytest.fixture
    def criterion(self) -> Module:
        # assert criterion_name in torch.nn.modules.loss
        return torch.nn.MSELoss()

    @pytest.fixture
    def network(self, dims, device: torch.device) -> Module:
        """Build a neural network that will be used for initializing the Agent."""
        raise NotImplementedError

    @pytest.fixture
    def agent(self, *args, **kwargs) -> Agent:
        """Initialize the Agent."""
        raise NotImplementedError

    @staticmethod
    def assert_action_is_valid(action, action_space):
        """Assert the agent's action is within a discrete action space."""
        if isinstance(action, np.ndarray):
            assert np.isfinite(action).all()
            assert action.shape[0] == action_space
            assert np.all((0 <= action) & (action < action_space))
        elif isinstance(action, (int, np.integer)):
            assert 0 <= action < action_space
        else:
            raise ValueError(f'Unexpected action type: {type(action)}.')

    @staticmethod
    def assert_replay_memory_within_range(agent, current_step: int):
        """Check if agent saves experiences in replay memory."""
        expected_len = min(current_step + 1, agent.memory.memory.maxlen)
        assert len(agent.memory) == expected_len

    def test_interact(self, seed: int, agent: Agent, env: gym.Env, action_space: int) -> None:
        state, _ = env.reset(seed=seed)
        max_episode_steps = env.spec.max_episode_steps

        for step in range(max_episode_steps):
            action = agent(state)
            self.assert_action_is_valid(action, action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            _, _ = agent.memorize(state, action, reward, next_state, terminated)
            self.assert_replay_memory_within_range(agent, step)
            state = next_state
            if terminated or truncated:
                break


class TestDQN(BaseAgentTest):
    @pytest.fixture
    def network(self, dims, device) -> FeedForwardNetwork:
        return FeedForwardNetwork(*dims, device=device)

    @pytest.fixture
    def agent(self, network, criterion, action_space, device):
        model = network
        return DeepQNetwork(
            model=model,
            criterion=criterion,
            action_space=action_space,
            device=device,
        )


class TestDoubleDQN(BaseAgentTest):
    @pytest.fixture
    def network(self, dims, device) -> FeedForwardNetwork:
        return FeedForwardNetwork(*dims, device=device)

    @pytest.fixture
    def agent(self, network, criterion, action_space, device):
        model = network
        return DoubleDQN(
            model=model,
            criterion=criterion,
            action_space=action_space,
            device=device,
        )


class TestDuelingDQN(BaseAgentTest):
    @pytest.fixture
    def network(self, dims, device: torch.device) -> DuelingQNetwork:
        return DuelingQNetwork(*dims, device=device)

    @pytest.fixture
    def agent(self, network, criterion, action_space, device) -> DuelingDQN:
        model = network
        return DuelingDQN(
            model=model,
            criterion=criterion,
            action_space=action_space,
            device=device,
        )
