"""
Builder file for trainers, the build_trainer function is called from 'run_manager'
to create the trainer.
"""
from __future__ import annotations
from trainers.base_trainer import BaseTrainer


def build_trainer(
    trainer_name: str,
    model: "torch.nn.Module",
    loss_fn: "torch.nn.modules.loss._Loss",
    optimizer: "torch.optim.optimizer.Optimizer",
    device: str,
    metrics: list[str],
    **kwargs: dict[str, any],
) -> BaseTrainer:
    """
    Main function for building the trainer. Uses the trainer_name to select the correct builder
    function from TRAINER_CONSTRUCTORS, and passes the kwargs on to this function

    Args:
        trainer_name (str): name of the trainer, should be in TRAINER_CONSTRUCTORS

        kwargs (dict[str, any]): the keyword arguments to be passed onto the builder function

    Returns:
        BaseTrainer: returns the built trainer
    """
    trainer = TRAINER_CONSTRUCTORS[trainer_name](
        model, loss_fn, optimizer, device, metrics, **kwargs
    )
    return trainer


def base_trainer(
    model: "torch.nn.Module",
    loss_fn: "torch.nn.modules.loss._Loss",
    optimizer: "torch.optim.optimizer.Optimizer",
    device: str,
    metrics: list[str],
    **_: dict[str, any],
) -> BaseTrainer:
    """
    Builds a BaseTrainer instance with the given parameters.

    Args:
        model (torch.nn.Module): the model to train
        loss_fn (torch.nn.modules.loss._Loss): the loss function to use
        optimizer (torch.optim.optimizer.Optimizer): the optimizer to use
        device (str): device to train on, should be 'cpu' or a cuda device
        metrics (list[str]): list of metrics to be calculated
        _ (dict[str, any]): not used

    Returns:
        BaseTrainer: a BaseTrainer instance
    """
    trainer = BaseTrainer(model, loss_fn, optimizer, device, metrics)
    return trainer


TRAINER_CONSTRUCTORS = {"base_trainer": base_trainer}
