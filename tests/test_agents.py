import pytest
import torch
import gymnasium as gym
import numpy as np

from typing import Tuple, Union
from torch.nn import Module
from nn import FeedForwardNetwork, DuelingQNetwork, PolicyNetwork
from agent import Agent, DeepQNetwork, DoubleDQN, DuelingDQN, DDPG, TD3


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
    def state_space(self, env: gym.Env) -> int:
        return env.observation_space.shape[0]

    @pytest.fixture
    def action_space(self, env: gym.Env) -> int:
        return env.action_space.n

    @pytest.fixture
    def dims(self, state_space, action_space) -> tuple[int, ...]:
        return state_space, np.random.randint(2, 32), np.random.randint(2, 32), action_space

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
    def assert_discrete_action_is_valid(action: Union[np.ndarray, int, np.int64], action_space: int):
        """Assert the agent's action is within a discrete action space."""
        if isinstance(action, np.ndarray):
            assert np.isfinite(action).all()
            assert action.shape[0] == action_space
            assert np.all((0 <= action) & (action < action_space))
        elif isinstance(action, (int, np.int64)):
            assert 0 <= action < action_space
        else:
            raise ValueError(f'Unexpected action type: {type(action)}.')

    def assert_continuous_action_is_valid(
        self,
        action: np.ndarray,
        action_space: int,
        categorical: bool,
        action_range: Tuple[np.ndarray,np.ndarray]
    ):
        """Assert the agent's action is within a continuous action space."""
        if categorical:
            self.assert_discrete_action_is_valid(action, action_space)
        else:
            min_action, max_action = action_range
            assert action.ndim >= 1 and action.shape[0] == action_space
            assert np.isfinite(action).all()
            assert np.all((min_action <= action) & (action <= max_action))
            if isinstance(action, np.ndarray):
                assert action.shape[0] == action_space
            else:
                raise ValueError(f'Unexpected action type: {type(action)}.')

    @staticmethod
    def assert_replay_memory_within_range(agent, current_step: int):
        """Check if agent saves experiences in replay memory."""
        assert isinstance(agent.memory.memory.maxlen, int)
        expected_len = min(current_step + 1, agent.memory.memory.maxlen)
        assert len(agent.memory) == expected_len

    def test_interact(self, seed: int, agent: Agent, env: gym.Env, action_space: int) -> None:
        state, _ = env.reset(seed=seed)
        max_episode_steps = env.spec.max_episode_steps

        for step in range(max_episode_steps):
            action = agent(state)
            self.assert_discrete_action_is_valid(action, action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            _ = agent.memorize(state, action, reward, next_state, terminated)
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


class PolicyAgentTest(BaseAgentTest):
    @pytest.fixture(
        params=[True, False], ids=['categorical', 'continuous'],
    )
    def action_space_type(self, request):
        return request.param

    @pytest.fixture
    def env(self, env_id, max_episode_steps) -> gym.Env:
        """Build a ``gymnasium`` environment."""
        environment = gym.make(id=env_id, max_episode_steps=max_episode_steps, continuous=True)
        yield environment
        environment.close()

    @pytest.fixture
    def action_space(self, env: gym.Env) -> int:
        return env.action_space.shape[0]

    @pytest.fixture
    def action_range(self, env: gym.Env) -> Tuple[float, float]:
        return env.action_space.low, env.action_space.high

    @pytest.fixture(params=[True, False], ids=['categorical', 'continuous'])
    def categorical(self, request) -> bool:
        raise NotImplementedError

    def test_interact(self, seed, agent, env, action_space, categorical, action_range) -> None:
        state, _ = env.reset(seed=seed)
        max_episode_steps = env.spec.max_episode_steps

        for step in range(max_episode_steps):
            action = agent(state)
            self.assert_continuous_action_is_valid(action, action_space, categorical=categorical, action_range=action_range)
            next_state, reward, terminated, truncated, _ = env.step(action)
            _ = agent.memorize(state, action, reward, next_state, terminated)
            self.assert_replay_memory_within_range(agent, step)
            state = next_state
            if terminated or truncated:
                break


class TestDDPG(PolicyAgentTest):
    @pytest.fixture
    def categorical(self, request) -> bool:
        return False

    @pytest.fixture
    def replay_memory_size(self) -> int:
        return int(1e5)

    @pytest.fixture
    def actor_dims(self, state_space, action_space) -> Tuple[int, ...]:
        return state_space, np.random.randint(2, 32), action_space

    @pytest.fixture
    def critic_dims(self, state_space, action_space) -> Tuple[int, ...]:
        return state_space + action_space, np.random.randint(2, 32), 1

    @pytest.fixture
    def actor(self, actor_dims, device, categorical):
        return PolicyNetwork(*actor_dims, device=device, categorical=categorical)

    @pytest.fixture
    def output_activation(self) -> type[torch.nn.Module]:
        return torch.nn.Tanh

    @pytest.fixture
    def critic(self, critic_dims, device, output_activation):
        return FeedForwardNetwork(*critic_dims, device=device, output_activation=output_activation)

    @pytest.fixture
    def agent(self, actor, critic, criterion, action_space, device, categorical, replay_memory_size, action_range):
        return DDPG(
            actor=actor,
            critic=critic,
            criterion=criterion,
            action_space=action_space,
            action_range=action_range,
            device=device,
            categorical=categorical,
            replay_memory_size=replay_memory_size,
        )


class TestTD3(TestDDPG):
    @pytest.fixture(params=[0.1, 1.1, 1e8], ids=["below_one", "over_one", "big_float"])
    def noise_clip(self, request) -> float:
        return request.param

    @pytest.fixture(params=[-0.1, 0., 2, -3], ids=["negative_float", "zero", "positive_int", "negative_int"])
    def noise_clip_invalid(self, request) -> Union[float, int]:
        return request.param

    @pytest.fixture()
    def agent(
        self, actor, critic, criterion, action_space, device, categorical, replay_memory_size, action_range, noise_clip
    ):
        return TD3(
            actor=actor,
            critic=critic,
            criterion=criterion,
            action_space=action_space,
            action_range=action_range,
            device=device,
            categorical=categorical,
            replay_memory_size=replay_memory_size,
            noise_clip=noise_clip,
        )

    def test_build_invalid_agent(
        self, actor, critic, criterion, action_space, device, categorical, replay_memory_size, action_range,
        noise_clip_invalid
    ):
        with pytest.raises(ValueError):
            return TD3(
                actor=actor,
                critic=critic,
                criterion=criterion,
                action_space=action_space,
                action_range=action_range,
                device=device,
                categorical=categorical,
                replay_memory_size=replay_memory_size,
                noise_clip=noise_clip_invalid,
            )

    def test_noise_is_in_range(self, agent: TD3, action_space, noise_clip) -> None:
        action = torch.randn(action_space)
        noise = agent.sample_noise(action)
        assert isinstance(noise, torch.Tensor)
        assert action.shape == noise.shape, "Action and noise shape do not match."
        assert torch.all((-noise_clip <= noise) & (noise <= noise_clip))

