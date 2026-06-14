import gym
import pynvml
import platform

from git import Repo


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
