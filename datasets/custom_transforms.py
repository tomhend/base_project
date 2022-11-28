import numpy as np
from scipy.ndimage import zoom
from typing import Tuple


class CustomResize:
    def __init__(self, scale: Tuple[float, float, float]) -> None:
        self.scale = scale

    def __call__(self, np_array: np.array) -> np.array:
        return zoom(np_array, self.scale)


class CustomMinMaxNormalize:
    def __init__(self, min: float, max: float) -> None:
        self.min = min
        self.max = max

    def __call__(self, np_array: np.array) -> np.array:
        return (np_array - self.min) / (self.max - self.min)


class CustomClip:
    def __init__(self, min: float, max: float) -> None:
        self.min = min
        self.max = max

    def __call__(self, np_array: np.array) -> np.array:
        return np.clip(np_array, self.min, self.max)
