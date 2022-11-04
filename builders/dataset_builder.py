"""
Builder file for datasets, the build_dataset function is called from 'run_manager'
to create the dataset.
"""

from torch.utils.data import Dataset
from datasets.mock_image_dataset import MockImageDataset


def build_dataset(dataset_name: str, **kwargs: dict[str, any]) -> Dataset:
    """
    Main function for building the dataset. Uses the dataset_name to select the correct builder
    function from DATASET_CONSTRUCTORS, and passes the kwargs on to this function.

    Args:
        dataset_name (str): name of the dataset, should be in DATASET_CONSTRUCTORS
        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function

    Returns:
        Dataset: the built dataset
    """
    dataset = DATASET_CONSTRUCTORS[dataset_name](**kwargs)
    return dataset


def mock_image(**kwargs: dict[str, any]) -> MockImageDataset:
    """
    Builds the mock image dataset from datasets/mock_image_dataset.py, with the specifications
    found in the kwargs.

    Args:
        kwargs(dict[str, any]): keyword arguments, should contain: size, image_size, classes

    Returns:
        MockImageDataset: an instance of the MockImageDataset
    """
    return MockImageDataset(**kwargs)


DATASET_CONSTRUCTORS = {"mock_image": mock_image}
