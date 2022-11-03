import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from run_files.run_logger import RunLogger
from run_files.metrics import Metrics


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer,
        device: str,
        metrics: Metrics,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.metrics = metrics
        self.selection_metric = None
        self.run_logger = None

    def set_run_logger(self, run_logger: RunLogger) -> None:
        self.run_logger = run_logger

    def set_run_metrics(self, metrics: Metrics) -> None:
        self.metrics = metrics

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        self.model.train()

        _inputs = []
        labels = []
        outputs = []
        losses = []

        for i, (_input, label) in enumerate(tqdm(dataloader)):
            step = epoch * len(dataloader) + i

            _input = _input.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(_input)
            loss = self.loss_fn(output, label)

            loss.backward()
            self.optimizer.step()

            step_metrics = self.metrics.calculate_metrics(
                moment="train_step",
                _input=_input,
                label=label,
                output=output,
                loss=loss.item(),
            )
            step_metrics.update({"train_step": step})

            _inputs.append(_input)
            labels.append(label)
            outputs.append(output)
            losses.append(loss.item())

            if self.run_logger:
                self.run_logger.log_metrics(step_metrics)

        epoch_loss = np.array(losses).mean()
        epoch_metrics = self.metrics.calculate_metrics(
            moment="train_epoch",
            _inputs=_inputs,
            labels=labels,
            outputs=outputs,
            loss=epoch_loss,
        )
        epoch_metrics.update({"epoch": epoch})

        if self.run_logger:
            self.run_logger.log_metrics(epoch_metrics)

        return epoch_metrics

    def val_epoch(self, dataloader: DataLoader, epoch: int) -> None:
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
                    moment="val_step",
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
                moment="val_epoch",
                _inputs=_inputs,
                labels=labels,
                outputs=outputs,
                loss=epoch_loss,
            )
            epoch_metrics.update({"epoch": epoch})

            if self.run_logger:
                self.run_logger.log_metrics(epoch_metrics)

        return epoch_metrics
