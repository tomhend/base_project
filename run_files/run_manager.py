"""
File containing the RunManager class which handles the initaliazation of all components necessary
for a run and progresses the run.
"""

import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from run_files.metrics import Metrics
from run_files.run_logger import RunLogger
from builders import (
    dataset_builder,
    loss_builder,
    model_builder,
    optimizer_builder,
    trainer_builder,
)


class RunManager:
    """
    This class initalizes all components necessary to execute the run, this is done following the
    configuration dictionary passed as an argument to the init function. The config file options and
    necessities are defined in a seperate file TODO: add file name.

    Attributes:
        cfg (dict[str, any]): configuration dictionary of the run
        logger (RunLogger, optional): logger that is used
        train_dataloader (DataLoader): dataloader containing the training data
        val_dataloader (DataLoader): dataloader containing the validation data
        model (nn.Module): the pytorch model that will be trained
        optimizer (torch.optim.Optimizer): optimizer that will be used
        loss_fn (torch.nn.modules.loss._Loss): loss function that will be used
        device (str): name of the device to train on
        metrics (Metrics): instance of Metrics that handles the metric calculation
        selection_metric (str): name of the metric to select the best model on
        goal (str): string defining if the selection metric should be maximized or minimized
        trainer (trainers.base_trainer.BaseTrainer): trainer handeling the training of the model
        epochs (int): the number of epochs to run
    """

    def __init__(self, cfg: dict[str, any]) -> None:
        """
        Initializes the RunManager given the cfg parameter.

        Args:
            cfg (dict[str, any]): the configuration dictionary for the run
        """

        self.cfg = cfg
        self.epochs = self.cfg["session_cfg"]["epochs"]
        self.logger = None

        # If log_cfg exists, build a logger
        if "log_cfg" in cfg:
            self.logger = RunLogger(cfg)
            # When doing wandb sweeps, the configuration is changed and stored in wandb.config
            # so it should be used in runs instead of cfg
            self.cfg = wandb.config

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info("Running on %s", self.device)

        self.train_dataloader, self.val_dataloader = self._create_dataloaders(self.cfg)

        model_cfg = self.cfg["model_cfg"]
        self.model = model_builder.build_model(model_cfg["name"], **model_cfg["kwargs"])

        optimizer_cfg = self.cfg["optimizer_cfg"]
        self.optimizer = optimizer_builder.build_optimizer(
            optimizer_cfg["name"], self.model.parameters(), **optimizer_cfg["kwargs"]
        )

        loss_cfg = self.cfg["loss_cfg"]
        self.loss_fn = loss_builder.build_loss_function(
            loss_cfg["name"], **loss_cfg.get("kwargs", {})
        )

        # Ensure that loss_train_epoch and loss_val_epoch are in the metric list as these are
        # required for training
        metric_list = list(
            set(
                ["loss_train_epoch", "loss_val_epoch"]
                + self.cfg["session_cfg"].get("metrics", [])
            )
        )
        self.metrics = Metrics(metric_list)
        self.selection_metric = self.cfg["session_cfg"].get(
            "selection_metric", "loss_val_epoch"
        )
        self.goal = self.cfg["session_cfg"].get("goal", "minimize")

        trainer_cfg = self.cfg["trainer_cfg"]
        self.trainer = trainer_builder.build_trainer(
            trainer_cfg["name"],
            self.model,
            self.loss_fn,
            self.optimizer,
            self.device,
            self.metrics,
            **trainer_cfg.get("kwargs", {}),
        )
        if self.logger:
            self.trainer.set_run_logger(self.logger)

    def start_training(self) -> None:
        """
        Starts the training and keeps running for the number of epochs in self.epochs, this method
        also handles the logic for saving the best model. NOTE: possibly seperate model saving into
        another method
        """
        best_metric_value = np.Inf
        if self.goal == "maximize":
            best_metric_value = -np.Inf

        for i in range(self.epochs):
            logging.info("starting training epoch %s", i)
            metrics_dict = self.trainer.train_epoch(self.train_dataloader, i)
            logging.info("starting validation epoch %s", i)
            validation_metrics_dict = self.trainer.val_epoch(self.val_dataloader, i)

            metrics_dict.update(validation_metrics_dict)
            model_name = (
                wandb.run.name
                if self.logger
                else f'{self.cfg["model_cfg"]["name"]}_best'
            )

            if self.goal == "maximize":
                if metrics_dict[self.selection_metric] > best_metric_value:
                    logging.info(
                        "Best value for %s, %s",
                        self.selection_metric,
                        metrics_dict[self.selection_metric],
                    )
                    torch.save(
                        self.model.state_dict(),
                        f"models/trained_models/{model_name}.pt",
                    )
                continue

            if metrics_dict[self.selection_metric] < best_metric_value:
                best_metric_value = metrics_dict[self.selection_metric]
                logging.info(
                    "Best value for %s, %s",
                    self.selection_metric,
                    metrics_dict[self.selection_metric],
                )
                torch.save(
                    self.model.state_dict(),
                    f"models/trained_models/{model_name}.pt",
                )

    @staticmethod
    def _create_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
        """
        Helper method that creates the dataloaders (as this is quite verbose)

        Args:
            cfg (dict): configuration file containing the train and validation set configurations

        Returns:
            Tuple[DataLoader, DataLoader]: the train and validation dataloaders
        """
        train_dataset_cfg = cfg["train_dataset_cfg"]
        train_dataset = dataset_builder.build_dataset(
            train_dataset_cfg["name"], **train_dataset_cfg["kwargs"]
        )

        val_dataset_cfg = cfg["val_dataset_cfg"]
        val_dataset = dataset_builder.build_dataset(
            val_dataset_cfg["name"], **val_dataset_cfg["kwargs"]
        )

        train_dataloader_cfg = cfg["train_dataloader_cfg"]
        train_dataloader = DataLoader(train_dataset, **train_dataloader_cfg)

        val_dataloader_cfg = cfg["val_dataloader_cfg"]
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        return train_dataloader, val_dataloader
