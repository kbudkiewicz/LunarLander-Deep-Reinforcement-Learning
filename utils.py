import pynvml
import platform
import torch
import gymnasium as gym
import agent as _agent_module

from typing import Tuple
from git import Repo
from torch.nn import Module

from agent import *
from nn import FeedForwardNetwork, DuelingQNetwork, PolicyNetwork


def get_nvml_info() -> dict:
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {
            'cuda.version': pynvml.nvmlSystemGetCudaDriverVersion_v2(),
            'cuda.driver': pynvml.nvmlSystemGetDriverVersion(),
            'gpu.count': pynvml.nvmlDeviceGetCount(),
            'gpu.name': pynvml.nvmlDeviceGetName(handle),
            'gpu.vram_mb': int(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1e6),
            'gpu.multiprocessor_count': pynvml.nvmlDeviceGetNumGpuCores(handle),
        }
    except pynvml.NVMLError as e:
        print(f"[WARNING] NVMLError: {e}")
        return {}
    finally:
        pynvml.nvmlShutdown()


def get_module_info() -> dict:
    return {
        'python.version': platform.python_version(),
        'gym.version': gym.__version__,
    }


def get_git_info() -> dict:
    with Repo('.').config_reader() as cfg:
        return {
            'git.user': cfg.get_value('user', 'name'),
            'git.email': cfg.get_value('user', 'email'),
        }


def get_agent_class(agent_type: str) -> type[ValueAgent, PolicyAgent]:
    """Import a chosen agent type module."""
    if not hasattr(_agent_module, agent_type):
        raise ImportError(f"Agent class {agent_type!r} is not a valid class. Choose from {_agent_module.__all__}")
    return getattr(_agent_module, agent_type)


def get_environment_dimensions(env: gym.Env) -> Tuple[int, int]:
    """Derive the observation and action space dimensions of the environment."""
    if not isinstance(env, gym.Env):
        raise ValueError("Expected a gym environment.")

    if isinstance(env.action_space, gym.spaces.Box):
        return env.observation_space.shape[0], env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.Discrete):
        return env.observation_space.shape[0], int(env.action_space.n)
    else:
        raise ValueError(f"Action space {env.action_space!r} is not a valid for this setup.")


def build_agent(
    agent: str,
    model_dims: Tuple[int],
    device: torch.device,
    criterion: Module,
    environment: gym.Env,
) -> Tuple[Agent, int]:
    """Build an Agent."""
    if not isinstance(agent, str):
        raise ValueError("Agent must be a string.")
    else:
        agent_class = get_agent_class(agent)
        observation_space, action_space = get_environment_dimensions(environment)
        dims = (observation_space, *model_dims, action_space)

        if issubclass(agent_class, (DDPG, TD3)):
            action_range = environment.action_space.low, environment.action_space.high
            actor = PolicyNetwork(*dims, device=device, categorical=False, deterministic=True)
            dims = (observation_space + action_space, *model_dims, 1)
            critic = FeedForwardNetwork(*dims, device=device)

            if issubclass(agent_class, DDPG):
                agent_kwargs = dict(
                    actor=actor, critic=critic, device=device, action_space=action_space, action_range=action_range,
                    criterion=criterion, categorical=False
                )
            elif issubclass(agent_class, TD3):
                agent_kwargs = dict(
                    actor=actor, critic=critic, device=device, action_space=action_space, action_range=action_range,
                    criterion=criterion, categorical=False, noise_clip=0.5
                )

            agent = agent_class(**agent_kwargs)
        elif issubclass(agent_class, DuelingDQN):
            agent = agent_class(
                model=DuelingQNetwork(*dims, device=device),
                action_space=action_space, device=device, criterion=criterion
            )
        else:
            agent = agent_class(
                model=FeedForwardNetwork(*dims, device=device),
                device=device, action_space=action_space, criterion=criterion,
            )

    return agent, action_space
