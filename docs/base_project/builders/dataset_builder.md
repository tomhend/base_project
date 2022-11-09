Module base_project.builders.dataset_builder
============================================
Builder file for datasets, the build_dataset function is called from 'run_manager'
to create the dataset.

Functions
---------

    
`build_dataset(dataset_name: str, **kwargs: dict[str, any]) -> torch.utils.data.dataset.Dataset`
:   Main function for building the dataset. Uses the dataset_name to select the correct builder
    function from DATASET_CONSTRUCTORS, and passes the kwargs on to this function.
    
    Args:
        dataset_name (str): name of the dataset, should be in DATASET_CONSTRUCTORS
        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function
    
    Returns:
        Dataset: the built dataset

    
`mock_image(**kwargs: dict[str, any]) -> datasets.mock_image_dataset.MockImageDataset`
:   Builds the mock image dataset from datasets/mock_image_dataset.py, with the specifications
    found in the kwargs.
    
    Args:
        kwargs(dict[str, any]): keyword arguments, should contain: size, image_size, classes
    
    Returns:
        MockImageDataset: an instance of the MockImageDataset