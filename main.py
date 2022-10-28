from typing import Tuple
import utils
import torch
from builders import dataset_builder, model_builder, trainer_builder, optimizer_builder, loss_builder
from torch.utils.data import DataLoader

def run():
    cfg = utils.parse_cfg('config.yaml')
    
    train_dataloader, val_dataloader = create_dataloaders(cfg)
    
    model_cfg = cfg['model_cfg']
    model = model_builder.build_model(model_cfg['name'], **model_cfg['kwargs'])
    
    optimizer_cfg = cfg['optimizer_cfg']
    optimizer = optimizer_builder.build_optimizer(optimizer_cfg['name'], model.parameters(), **optimizer_cfg['kwargs'])

    loss_cfg = cfg['loss_cfg']
    loss_fn = loss_builder.build_loss_function(loss_cfg['name'], **loss_cfg.get('kwargs', {}))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    trainer_cfg = cfg['trainer_cfg']
    trainer = trainer_builder.build_trainer(trainer_cfg['name'], model, loss_fn, optimizer, device, **trainer_cfg.get('kwargs', {}))
    
    trainer.train_epoch(train_dataloader)
    trainer.val_epoch(val_dataloader)

def create_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    train_dataset_cfg = cfg['train_dataset_cfg']
    train_dataset = dataset_builder.build_dataset(train_dataset_cfg['name'], **train_dataset_cfg['kwargs'])
    
    val_dataset_cfg = cfg['val_dataset_cfg']
    val_dataset = dataset_builder.build_dataset(val_dataset_cfg['name'], **val_dataset_cfg['kwargs'])
    
    train_dataloader_cfg = cfg['train_dataloader_cfg']
    train_dataloader = DataLoader(train_dataset, **train_dataloader_cfg)
    
    val_dataloader_cfg = cfg['val_dataloader_cfg']
    val_dataloader = DataLoader(val_dataset, **val_dataloader_cfg)
    
    return train_dataloader, val_dataloader

if __name__ == '__main__':
    run()