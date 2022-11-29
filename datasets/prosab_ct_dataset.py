"""
File containing the NiftiImageDataset class.
"""

import torch
import SimpleITK as sitk
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from datasets.custom_transforms import CustomClip, CustomMinMaxNormalize, CustomResize
from torchvision import transforms


class NiftiImageDataset(Dataset):
    """
    Dataset for nifti images
    """
    def __init__(self, index_file_path: Path) -> None:
        """
        Initializes the dataset via an index_file_path. This index file should be a csv with an
        outcomes and nifti_paths columns.

        Args:
            index_file_path (Path): the path to the index file
        """
        super().__init__()

        self.index_df = pd.read_csv(index_file_path)
        self._transforms = transforms.Compose(
            [
                CustomResize((1, 0.125, 0.125)),
                CustomClip(0, 100),
                CustomMinMaxNormalize(0, 100),
            ]
        )
        # TODO: create builder for transforms to allow selection from cfg file

    def __len__(self) -> int:
        """
        Returns number of instances in dataset

        Returns:
            int: size of the dataset
        """
        return len(self.index_df)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image-outcome pair as tensors.

        Args:
            index (int): the index of the desired dataset entry

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Returns the image-outcome pair as a tuple of tensors
        """
        scan_path = self.index_df.loc[index, "nifti_paths"]
        outcome = self.index_df.loc[index, "outcomes"]
        scan_nifti = sitk.ReadImage(str(Path(scan_path)))
        scan_array = sitk.GetArrayFromImage(scan_nifti)
        if self._transforms:
            scan_array = self._transforms(scan_array)

        return torch.tensor(scan_array, dtype=torch.float).unsqueeze(0), torch.tensor(
            [outcome], dtype=torch.float
        )
