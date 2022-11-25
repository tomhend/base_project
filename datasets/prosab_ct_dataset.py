import torch
import SimpleITK as sitk
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
from datasets.custom_transforms import CustomClip, CustomMinMaxNormalize, CustomResize
from torchvision import transforms


class ProsabCTDataset(Dataset):
    def __init__(self, index_file_path: Path) -> None:
        super().__init__()

        self.index_df = pd.read_csv(index_file_path)
        self._transforms = transforms.Compose(
            [
                CustomResize((1, 0.5, 0.5)),
                CustomClip(-100, 500),
                CustomMinMaxNormalize(-100, 500),
            ]
        )
        # TODO: create builder for transforms to allow selection from cfg file

    def __len__(self) -> int:
        return len(self.index_df)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        scan_path = self.index_df.loc[index, "nifti_paths"]
        outcome = self.index_df.loc[index, "outcomes"]
        scan_nifti = sitk.ReadImage(str(Path(scan_path)))
        scan_array = sitk.GetArrayFromImage(scan_nifti)
        if self._transforms:
            scan_array = self._transforms(scan_array)

        return torch.tensor(scan_array, dtype=torch.float).unsqueeze(0), torch.tensor([outcome], dtype=torch.float)
