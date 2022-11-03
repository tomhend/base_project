from typing import Tuple
from torch.utils.data import DataLoader
import torch
import logging
import wandb
import numpy as np

from builders import dataset_builder, model_builder, trainer_builder, optimizer_builder, loss_builder
from run_files.run_logger import RunLogger
from run_files.metrics import Metrics


class RunManager:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.logger = None
        if 'log_cfg' in cfg.keys():
            self.logger = RunLogger(cfg)
            self.cfg = wandb.config

        self.train_dataloader, self.val_dataloader = self._create_dataloaders(
            self.cfg)

        model_cfg = self.cfg['model_cfg']
        self.model = model_builder.build_model(
            model_cfg['name'], **model_cfg['kwargs'])

        optimizer_cfg = self.cfg['optimizer_cfg']
        self.optimizer = optimizer_builder.build_optimizer(
            optimizer_cfg['name'], self.model.parameters(), **optimizer_cfg['kwargs'])

        loss_cfg = self.cfg['loss_cfg']
        self.loss_fn = loss_builder.build_loss_function(
            loss_cfg['name'], **loss_cfg.get('kwargs', {}))

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Running on {self.device}')

        # Ensure that loss_train_epoch and loss_val_epoch are in the metric list
        metric_list = list(set(
            ['loss_train_epoch', 'loss_val_epoch'] + self.cfg['session_cfg'].get('metrics', [])))
        self.metrics = Metrics(metric_list)
        self.selection_metric = self.cfg['session_cfg'].get(
            'selection_metric', 'loss_val_epoch')
        self.goal = self.cfg['session_cfg'].get('goal', 'minimize')

        trainer_cfg = self.cfg['trainer_cfg']
        self.trainer = trainer_builder.build_trainer(
            trainer_cfg['name'], self.model, self.loss_fn, self.optimizer, self.device, self.metrics, **trainer_cfg.get('kwargs', {}))
        if self.logger:
            self.trainer.set_run_logger(self.logger)

        self.epochs = self.cfg['session_cfg']['epochs']

    def start_training(self) -> None:
        best_metric_value = np.Inf
        if self.goal == 'maximize':
            best_metric_value = -np.Inf

        for i in range(self.epochs):
            logging.info(f'starting training epoch {i}')
            metrics_dict = self.trainer.train_epoch(self.train_dataloader, i)
            logging.info(f'starting validation epoch {i}')
            validation_metrics_dict = self.trainer.val_epoch(
                self.val_dataloader, i)

            metrics_dict.update(validation_metrics_dict)
            model_name = wandb.run.name if self.logger else f'{self.cfg["model_cfg"]["name"]}_best'

            if self.goal == 'maximize':
                if metrics_dict[self.selection_metric] > best_metric_value:
                    logging.info(
                        f'Best value for {self.selection_metric}, {metrics_dict[self.selection_metric]}')
                    torch.save(self.model.state_dict(
                    ), f'models/saved_models/trained_models/{model_name}.pt')
                continue

            if metrics_dict[self.selection_metric] < best_metric_value:
                logging.info(
                    f'Best value for {self.selection_metric}, {metrics_dict[self.selection_metric]}')
                torch.save(self.model.state_dict(),
                           f'models/saved_models/trained_models/{model_name}.pt')

    @staticmethod
    def _create_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
        train_dataset_cfg = cfg['train_dataset_cfg']
        train_dataset = dataset_builder.build_dataset(
            train_dataset_cfg['name'], **train_dataset_cfg['kwargs'])

        val_dataset_cfg = cfg['val_dataset_cfg']
        val_dataset = dataset_builder.build_dataset(
            val_dataset_cfg['name'], **val_dataset_cfg['kwargs'])

        train_dataloader_cfg = cfg['train_dataloader_cfg']
        train_dataloader = DataLoader(train_dataset, **train_dataloader_cfg)

        val_dataloader_cfg = cfg['val_dataloader_cfg']
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)

        return train_dataloader, val_dataloader
