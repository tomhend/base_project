from abc import ABC
import torch
from torch.utils.data import DataLoader
from run_logger import RunLogger

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer, device: str) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.run_logger = None
    
    def set_run_logger(self, run_logger: RunLogger) -> None:
        self.run_logger = run_logger
    
    def train_epoch(self,  dataloader: DataLoader, epoch: int) -> None:
        self.model.train()
        
        _inputs = []
        labels = []
        outputs = []
        losses = []
        
        for _input, label in dataloader:
            _input = _input.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(_input)
            loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.step()
            
            if self.run_logger:
                self.run_logger.log_train_step(_input=_input, label=label, output=output, loss=loss.item())
                
                _inputs.append(_input)
                labels.append(label)
                outputs.append(output)
                losses.append(loss.item())
        
        if self.run_logger:
            self.run_logger.log_train_epoch(_input=_input, label=label, output=output, losses=losses)
    
    def val_epoch(self, dataloader: DataLoader, epoch: int) -> None:
        self.model.eval()
        
        _inputs = []
        labels = []
        outputs = []
        losses = []
        
        with torch.no_grad():
            for _input, label in dataloader:
                _input = _input.to(self.device)
                label = label.to(self.device)
                
                output = self.model(_input)
                loss = self.loss_fn(output, label)
                
                if self.run_logger:
                    self.run_logger.log_val_step(_input=_input, label=label, output=output, loss=loss.item())
                    _inputs.append(_input)
                    labels.append(label)
                    outputs.append(output)
                    losses.append(loss.item())
                
            if self.run_logger:
                self.run_logger.log_val_epoch(_input=_input, label=label, output=output, losses=losses)
            