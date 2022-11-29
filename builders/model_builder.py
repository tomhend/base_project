"""
Builder file for models, the build_model function is called from 'run_manager'
to create the model.
"""

import torch
import torchvision

from models.architectures.medical_net import MedicalNet10, MedicalNet50

from models.architectures import layered_3dconvnet


def build_model(model_name: str, **kwargs: dict[str, any]) -> torch.nn.Module:
    """
    Main function for building the model. Uses the model_name to select the correct builder
    function from MODEL_CONSTRUCTORS, and passes the kwargs on to this function

    Args:
        model_name (str): name of the model, should be in MODEL_CONSTRUCTORS
        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function

    Returns:
        torch.nn.Module: returns the built model
    """
    model = MODEL_CONSTRUCTORS[model_name](**kwargs)
    return model


def test_resnet(**kwargs: dict[str, any]) -> torch.nn.Module:
    """
    Builds a torchvision resnet18 with the given kwargs

    Args:
        kwargs (dict[str, any]):
            see https://pytorch.org/vision/stable/models/resnet.html for available kwargs

    Returns:
        torch.nn.Module: the created resnet18
    """
    model = torchvision.models.resnet18(**kwargs)
    return model


def medical_net10(**kwargs) -> MedicalNet10:
    """
    Builds a medicalnet 10 from models.architectures.medical_net with the given kwargs.
    See: https://github.com/Tencent/MedicalNet

    Args:
        kwargs (dict[str, any]): kwargs used to construct the model.

    Returns:
        torch.nn.Module: the created MedicalNet10
    """
    model = MedicalNet10(**kwargs)
    return model


def medical_net50(**kwargs) -> MedicalNet50:
     """
    Builds a medicalnet 50 from models.architectures.medical_net with the given kwargs.
    See: https://github.com/Tencent/MedicalNet

    Args:
        kwargs (dict[str, any]): kwargs used to construct the model.

    Returns:
        torch.nn.Module: the created MedicalNet50
    """
    model = MedicalNet50(**kwargs)
    return model


def convnet_3d(**kwargs: dict[str, any]) -> torch.nn.Module:
     """
    Builds a layered convnet3D from models.architectures.layered_3dconvnet with the given kwargs.

    Args:
        kwargs (dict[str, any]): kwargs used to construct the model.

    Returns:
        torch.nn.Module: the created MedicalNet10
    """
    model = layered_3dconvnet.ConvNet3D(**kwargs)
    return model


MODEL_CONSTRUCTORS = {
    "test_resnet": test_resnet,
    "convnet_3d": convnet_3d,
    "medical_net10": medical_net10,
    "medical_net50": medical_net50,
}
