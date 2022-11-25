"""
The file containing the BaseTrainer class, which runs a basic pytorch training loop.
"""

import torch
import numpy as np
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

from run_files.run_logger import RunLogger
from run_files.metrics import Metrics, Moments


class BaseTrainer:
    """
    Class that handles the training loop and optionally sends information to the logger.

    Attributes:
        model (nn.Module): the pytorch model that will be trained
        optimizer (torch.optim.Optimizer): optimizer that will be used
        loss_fn (torch.nn.modules.loss._Loss): loss function that will be used
        device (str): name of the device to train on
        metrics (Metrics): instance of Metrics that handles the metric calculation
        run_logger (RunLogger, optional): the logger that handles the logging of training
        and validation info
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        device: str,
        metrics: Metrics,
    ) -> None:
        """
        Initializes an instance of BaseTrainer.

        Args:
            model (torch.nn.Module): model to train
            loss_fn (torch.nn.modules.loss._Loss): loss function to use
            optimizer (torch.optim.Optimizer): the optimizer to use
            device (str): the device to train on
            metrics (Metrics): the Metric instance to calculate the metrics
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.metrics = metrics
        self.run_logger = None

    def set_run_logger(self, run_logger: RunLogger) -> None:
        """
        Sets the run_logger.

        Args:
            run_logger (RunLogger): the RunLogger instance to use
        """
        self.run_logger = run_logger

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, any]:
        """
        Runs a training epoch for the model with the given dataloader.

        Args:
            dataloader (DataLoader): the dataloader containing the model input and labels
            epoch (int): the epoch number

        Returns:
            dict[str, any]: the metrics calculated at the end of the epoch
        """
        self.model.train()

        # _inputs = []
        labels = []
        outputs = []
        losses = []

        for i, (_input, label) in enumerate(tqdm(dataloader)):
            step = epoch * len(dataloader) + i

<<<<<<< HEAD
            _input = _input.to(self.device)
=======
            # _input = _input.to(self.device) may cause memory issues with large datasets
>>>>>>> origin/main
            label = label.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(_input)
            loss = self.loss_fn(output, label)

            loss.backward()
            self.optimizer.step()

            step_metrics = self.metrics.calculate_metrics(
                moment=Moments.TRAIN_STEP,
                # _input=_input.detach(), may cause memory issues with large datasets
                label=label.detach(),
                output=output.detach(),
                loss=loss.item(),
            )
            step_metrics.update({Moments.TRAIN_STEP: step})

            # _inputs.append(_input) may cause memory issues with large datasets
            labels.append(label.detach())
            outputs.append(output.detach())
            losses.append(loss.item())

            if self.run_logger:
                self.run_logger.log_metrics(step_metrics)

        epoch_loss = np.array(losses).mean()
        epoch_metrics = self.metrics.calculate_metrics(
            moment=Moments.TRAIN_EPOCH,
            # _inputs=_inputs, may cause memory issues with large datasets
            labels=labels,
            outputs=outputs,
            loss=epoch_loss,
        )
        epoch_metrics.update({"epoch": epoch})

        if self.run_logger:
            self.run_logger.log_metrics(epoch_metrics)

        return epoch_metrics

    def val_epoch(self, dataloader: DataLoader, epoch: int) -> dict[str, any]:
        """
        Runs a validation epoch for the model with the given dataloader.

        Args:
            dataloader (DataLoader): the dataloader containing the model input and labels
            epoch (int): the epoch number

        Returns:
            dict[str, any]: the metrics calculated at the end of the epoch
        """
        self.model.eval()

        _inputs = []
        labels = []
        outputs = []
        losses = []

        with torch.no_grad():
            for _input, label in tqdm(dataloader):
                _input = _input.to(self.device)
                label = label.to(self.device)

                output = self.model(_input)
                loss = self.loss_fn(output, label)

                step_metrics = self.metrics.calculate_metrics(
                    moment=Moments.VAL_STEP,
                    _input=_input,
                    label=label,
                    output=output,
                    loss=loss.item(),
                )

                _inputs.append(_input)
                labels.append(label)
                outputs.append(output)
                losses.append(loss.item())

                if self.run_logger:
                    self.run_logger.log_metrics(step_metrics)

            epoch_loss = np.array(losses).mean()
            epoch_metrics = self.metrics.calculate_metrics(
                moment=Moments.VAL_EPOCH,
                _inputs=_inputs,
                labels=labels,
                outputs=outputs,
                loss=epoch_loss,
            )
            epoch_metrics.update({"epoch": epoch})

            if self.run_logger:
                self.run_logger.log_metrics(epoch_metrics)

        return epoch_metrics
