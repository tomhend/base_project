"""
Builder file for loss functions, the build_loss_function function is called from 'run_manager'
to create the loss_function.
"""

import torch


def build_loss_function(
    loss_name: str, **kwargs: dict[str, any]
) -> torch.nn.modules.loss._Loss:
    """
    Main function for building the loss function. Uses the loss_name to select the correct builder
    function from LOSS_CONSTRUCTORS, and passes the kwargs on to this function

    Args:
        loss_name (str): name of the loss_function, should be in LOSS_CONSTRUCTORS
        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function

    Returns:
        torch.nn.modules.loss._Loss: the built loss function
    """
    loss_fn = LOSS_CONSTRUCTORS[loss_name](**kwargs)
    return loss_fn


def bce_logits(**kwargs: dict[str, any]) -> torch.nn.BCEWithLogitsLoss:
    """
    Builds a Binary Cross Entropy with logits loss function with the given kwargs

    Args:
        kwargs (dict[str, any]):
            see https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html for
            available kwargs

    Returns:
        torch.nn.BCEWithLogitsLoss: BCEWithLogitsLoss object initialized with kwargs
    """
    loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
    return loss_fn


def ce_logits(**kwargs: dict[str, any]) -> torch.nn.CrossEntropyLoss:
    """
    Builds a Cross Entropy with logits loss function with the given kwargs

    Args:
        kwargs (dict[str, any]):
            see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html for
            available kwargs

    Returns:
        torch.nn.CrossEntropyLoss: CrossEntropyLoss object initialized with kwargs
    """
    loss_fn = torch.nn.CrossEntropyLoss(**kwargs)
    return loss_fn


LOSS_CONSTRUCTORS = {"bce_logits": bce_logits, "ce_logits": ce_logits}
