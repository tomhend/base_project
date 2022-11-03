from datasets.mock_image_dataset import MockImageDataset
from torch.utils.data import Dataset


def build_dataset(dataset_name: str, **kwargs) -> Dataset:
    dataset = DATASET_CONSTRUCTORS[dataset_name](**kwargs)
    return dataset


def mock_image(**kwargs) -> MockImageDataset:
    return MockImageDataset(**kwargs)


DATASET_CONSTRUCTORS = {"mock_image": mock_image}
