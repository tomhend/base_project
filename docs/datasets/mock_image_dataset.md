Module base_project.datasets.mock_image_dataset
===============================================
This file contains the definition of the MockImageDataset

Classes
-------

`MockImageDataset(size: int, image_size: list[int], classes: int)`
:   Class that creates a randomly initialized image dataset of a given size and image size, and
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
    
    Initializes the dataset wit the given parameters.
    
    Args:
        size (int): number of image-outcome pairs in the dataset
        image_size (list[int]): size of the image (channels, height, width)
        classes (int): numbers of classes in outcome
        _transforms (torchvision.transforms.Compose, optional): Transform to apply to the
            generated image. Defaults to None.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic