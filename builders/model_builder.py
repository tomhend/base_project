"""
Builder file for models, the build_model function is called from 'run_manager'
to create the model.
"""

import torch
import torchvision

from models.architectures.medical_net import MedicalNet10, MedicalNet50


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
    model = MedicalNet10(**kwargs)
    return model

def medical_net50(**kwargs) -> MedicalNet50:
    model = MedicalNet50(**kwargs)
    return model

MODEL_CONSTRUCTORS = {
    'test_resnet': test_resnet,
    'medical_net10': medical_net10,
    'medical_net50': medical_net50
}
