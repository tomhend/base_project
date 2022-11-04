"""
File containing some utility functions
"""

from pathlib import Path
from typing import Callable
from functools import wraps
import time
import yaml


def parse_cfg(cfg_path: Path) -> dict[any, any]:
    """
    Parses the yaml file to a dictionary

    Args:
        cfg_path (Path): Path instance that points to the configuration yaml file

    Returns:
        dict[any, any]: parsed yaml file
    """
    with open(cfg_path, "r", encoding="utf8") as cfgyaml:
        cfg = yaml.safe_load(cfgyaml)
    return cfg


def time_func(func: Callable) -> Callable:
    """
    Decorator that times the execution time of a function or method

    Args:
        func (Callable): the function or method to be decorated

    Returns:
        Callable: the decorated function
    """

    @wraps(func)
    def time_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return time_wrapper
