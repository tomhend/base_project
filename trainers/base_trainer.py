import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from run_files.run_logger import RunLogger
from run_files.metrics import Metrics

class BaseTrainer:
    def __init__(self, model: torch.nn.Module, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.Optimizer, device: str) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.run_logger = None
        self.metrics = None
    
    def set_run_logger(self, run_logger: RunLogger) -> None:
        self.run_logger = run_logger
    
    def set_run_metrics(self, metrics: Metrics) -> None:
        self.metrics = metrics
    
    def train_epoch(self,  dataloader: DataLoader, epoch: int) -> None:
        self.model.train()
        
        _inputs = []
        labels = []
        outputs = []
        losses = []
        
        for i, (_input, label) in enumerate(tqdm(dataloader)):
            step = epoch*len(dataloader) + i
            
            _input = _input.to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            
            output = self.model(_input)
            loss = self.loss_fn(output, label)
            
            loss.backward()
            self.optimizer.step()
            
            if self.metrics:
                metrics_dict = self.metrics.calculate_metrics(moment='train_step', _input=_input, label=label, output=output, loss=loss.item())
                metrics_dict.update({'train_step': step})
                self.run_logger.log_metrics(metrics_dict)
                
                _inputs.append(_input)
                labels.append(label)
                outputs.append(output)
                losses.append(loss.item())
        
        if self.metrics:
            metrics_dict = self.metrics.calculate_metrics(moment='train_epoch', _inputs=_inputs, labels=labels, outputs=outputs, losses=losses)
            metrics_dict.update({'epoch': epoch})
            self.run_logger.log_metrics(metrics_dict)
    
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
                
                if self.metrics:
                    metrics_dict = self.metrics.calculate_metrics(moment='val_step', _input=_input, label=label, output=output, loss=loss.item())
                    self.run_logger.log_metrics(metrics_dict)
                    
                    _inputs.append(_input)
                    labels.append(label)
                    outputs.append(output)
                    losses.append(loss.item())
                
            if self.metrics:
                metrics_dict = self.metrics.calculate_metrics(moment='val_epoch', _inputs=_inputs, labels=labels, outputs=outputs, losses=losses)
                metrics_dict.update({'epoch': epoch})
                self.run_logger.log_metrics(metrics_dict)
            