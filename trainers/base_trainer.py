from abc import ABC
import torch
from torch.utils.data import DataLoader

class BaseTrainer(ABC):
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer, device: str) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def train_epoch(self,  dataloader: DataLoader) -> None:
        self.model.train()
        
        for _input, label in dataloader:
            _input = _input.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(_input)
            loss = self.loss_fn(output, label)
            loss.backward()
            self.optimizer.step()
            print(loss.item())
    
    def val_epoch(self, dataloader: DataLoader) -> None:
        self.model.eval()
        
        with torch.no_grad():
            for _input, label in dataloader:
                _input = _input.to(self.device)
                label = label.to(self.device)
                
                output = self.model(_input)
                loss = self.loss_fn(output, label)
                
                print(f'validation loss: {loss.item()}')
            