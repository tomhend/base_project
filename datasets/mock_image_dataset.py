"""
This file contains the definition of the MockImageDataset
"""

from typing import Tuple

import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset


class MockImageDataset(Dataset):
    """
    Class that creates a randomly initialized image dataset of a given size and image size, and
    returns the random image and a one-hot outcome vector as tensors when indexed. Special case of
    classes = 1 gives a 0 or 1 as tensor.

    Attributes:
        size (int): the size of the dataset
        image_size (list[int]): the size of the image (channels, height, width)
        _transforms (torchvision.transforms.Compose): composition of the transforms
        img_cache (np.array): a cache of the generated images
        outcomes (np.array): when classes = 1, a 2x1 array with 0, 1 as values, else the identity
            matrix with size = classes
        outcome_cache (dict[int, np.array]): a cache of the generated outcomes
    """

    def __init__(
        self,
        size: int,
        image_size: list[int],
        classes: int,
        _transforms: torchvision.transforms.Compose = None,
    ) -> None:
        """
        Initializes the dataset wit the given parameters.

        Args:
            size (int): number of image-outcome pairs in the dataset
            image_size (list[int]): size of the image (channels, height, width)
            classes (int): numbers of classes in outcome
            _transforms (torchvision.transforms.Compose, optional): Transform to apply to the
                generated image. Defaults to None.
        """
        super().__init__()

        self.size = size
        self.image_size = image_size
        self._transforms = _transforms
        self.img_cache = np.zeros((size, *image_size), dtype=np.float32)
        self.outcomes = np.eye(classes) if classes > 1 else np.array([[0], [1]])
        self.outcome_cache = {}

    def __len__(self) -> int:
        """
        Returns number of instances in dataset

        Returns:
            int: size of the dataset
        """
        return self.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image-outcome pair as tensors from cache. If not available, generates a new img,
        or randomly selects an outcome.

        Args:
            index (int): the index of the desired dataset entry

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the image-outcome pair as a tuple of tensors
        """
        mock_img = self.img_cache[index]
        if not np.any(mock_img):
            mock_img = np.random.rand(*self.image_size)
            self.img_cache[index] = mock_img

        if index in self.outcome_cache:
            outcome = self.outcome_cache[index]
        else:
            outcome = self.outcomes[np.random.randint(0, self.outcomes.shape[0])]

        return torch.Tensor(mock_img), torch.Tensor(outcome)
