import torch

def build_loss_function(loss_name: str, **kwargs) -> torch.nn.modules.loss._Loss:
    loss_fn = LOSS_CONSTRUCTORS[loss_name](**kwargs)
    return loss_fn

def bce_logits(**kwargs) -> torch.nn.BCEWithLogitsLoss:
    loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
    return loss_fn
    
LOSS_CONSTRUCTORS = {
    'bce_logits': bce_logits
}