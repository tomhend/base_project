from builders import dataset_builder, model_builder, trainer_builder, optimizer_builder, loss_builder
from run_logger import RunLogger
from typing import Tuple
from torch.utils.data import DataLoader
import torch


class RunManager:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        
        self.train_dataloader, self.val_dataloader = self._create_dataloaders(cfg)
    
        model_cfg = cfg['model_cfg']
        self.model = model_builder.build_model(model_cfg['name'], **model_cfg['kwargs'])
        
        optimizer_cfg = cfg['optimizer_cfg']
        self.optimizer = optimizer_builder.build_optimizer(optimizer_cfg['name'], self.model.parameters(), **optimizer_cfg['kwargs'])

        loss_cfg = cfg['loss_cfg']
        self.loss_fn = loss_builder.build_loss_function(loss_cfg['name'], **loss_cfg.get('kwargs', {}))
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        trainer_cfg = cfg['trainer_cfg']
        self.trainer = trainer_builder.build_trainer(trainer_cfg['name'], self.model, self.loss_fn, self.optimizer, self.device, **trainer_cfg.get('kwargs', {}))
        
        self.epochs = cfg['session_cfg']['epochs']
        
        log_cfg = cfg.get('log_cfg', None)
        self.logger = RunLogger(log_cfg) if log_cfg else None
    
    def start_training(self) -> None:
        for i in range(self.epochs):
            print(f'starting epoch {i}')
            self.trainer.train_epoch(self.train_dataloader)
            self.trainer.val_epoch(self.val_dataloader)
            
            if self.logger:
                self.logger.log(train_loss=5.0)
            
    
    @staticmethod
    def _create_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
        train_dataset_cfg = cfg['train_dataset_cfg']
        train_dataset = dataset_builder.build_dataset(train_dataset_cfg['name'], **train_dataset_cfg['kwargs'])
        
        val_dataset_cfg = cfg['val_dataset_cfg']
        val_dataset = dataset_builder.build_dataset(val_dataset_cfg['name'], **val_dataset_cfg['kwargs'])
        
        train_dataloader_cfg = cfg['train_dataloader_cfg']
        train_dataloader = DataLoader(train_dataset, **train_dataloader_cfg)
        
        val_dataloader_cfg = cfg['val_dataloader_cfg']
        val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)
        
        return train_dataloader, val_dataloader