import torch
import torchvision

from models.architectures.medical_net import MedicalNet10, MedicalNet50


def build_model(model_name: str, **kwargs) -> torch.nn.Module:
    model = MODEL_CONSTRUCTORS[model_name](**kwargs)
    return model

def test_resnet(**kwargs) -> torch.nn.Module:
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