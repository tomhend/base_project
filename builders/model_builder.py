import torch
import torchvision


def build_model(model_name: str, **kwargs) -> torch.nn.Module:
    model = MODEL_CONSTRUCTORS[model_name](**kwargs)
    return model


def test_resnet(**kwargs) -> torch.nn.Module:
    model = torchvision.models.resnet18(**kwargs)
    return model


MODEL_CONSTRUCTORS = {
    'test_resnet': test_resnet
}
