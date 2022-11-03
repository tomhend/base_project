from pathlib import Path
from typing import Callable
from functools import wraps
import time
import yaml


def parse_cfg(cfg_path: Path) -> dict:
    with open(cfg_path, "r") as cfgyaml:
        cfg = yaml.safe_load(cfgyaml)
    return cfg


def time_func(func: Callable) -> Callable:
    @wraps(func)
    def time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return time_wrapper
