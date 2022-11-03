import torchvision
import torch
import numpy as np
from typing import Tuple
from torch.utils.data import Dataset


class MockImageDataset(Dataset):
    def __init__(
        self,
        size: int,
        image_size: list[int],
        classes: int,
        _transforms: torchvision.transforms = None,
    ) -> None:
        super().__init__()

        self.size = size
        self.image_size = image_size
        self._transforms = _transforms
        self.img_cache = np.zeros((size, *image_size), dtype=np.float32)
        self.outcomes = np.eye(classes)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        mock_img = self.img_cache[index]
        if not np.any(mock_img):
            mock_img = np.random.rand(*self.image_size)
            self.img_cache[index] = mock_img
        return torch.Tensor(mock_img), torch.Tensor(self.outcomes[index])
