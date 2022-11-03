from datasets.mock_image_dataset import MockImageDataset
from torch.utils.data import Dataset

from datasets.prosab_ct_dataset import ProsabCTDataset
from pathlib import Path

def build_dataset(dataset_name: str, **kwargs) -> Dataset:
    dataset = DATASET_CONSTRUCTORS[dataset_name](**kwargs)
    return dataset


def mock_image(**kwargs) -> MockImageDataset:
    return MockImageDataset(**kwargs)


<<<<<<< HEAD
def prosab_ct(**kwargs) -> ProsabCTDataset:
    index_file_path = Path(kwargs['index_file_path'])
    return ProsabCTDataset(index_file_path=index_file_path)


DATASET_CONSTRUCTORS = {
    'mock_image': mock_image,
    'prosab_ct': prosab_ct
}
=======
DATASET_CONSTRUCTORS = {"mock_image": mock_image}
>>>>>>> main
