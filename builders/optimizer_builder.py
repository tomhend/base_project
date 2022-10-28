from typing import Generator
import torch
import torchvision


def build_optimizer(optimizer_name: str, model_parameters: Generator[torch.Tensor, None, None], **kwargs) -> torch.nn.Module:
    optimizer = MODEL_CONSTRUCTORS[optimizer_name](model_parameters, **kwargs)
    return optimizer

def adam(model_parameters, **kwargs) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model_parameters, **kwargs)
    return optimizer
    

MODEL_CONSTRUCTORS = {
    'adam': adam
}