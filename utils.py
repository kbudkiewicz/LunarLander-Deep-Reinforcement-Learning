import pynvml
import platform
import gymnasium as gym
import agent as _agent_module

from git import Repo
from agent import Agent


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


def get_agent_class(agent_type: str) -> type[Agent]:
    """Import a chosen agent type module."""
    if not hasattr(_agent_module, agent_type):
        raise ImportError(f"Agent class {agent_type!r} is not a valid class. Choose from {_agent_module.__all__}")
    return getattr(_agent_module, agent_type)
