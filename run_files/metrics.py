from typing import Callable, Union
import torch
import numpy as np

class Metrics:
    METRIC_FUNCTIONS = {}
    
    def __init__(self, name_list: list[str]) -> None:
        self.selected_metrics = self.select_metrics(name_list)
        self.best_metrics = {}
    
    def select_metrics(self, name_list: list[str]) -> dict[str, Callable]:
        try:
            return {function_name: self.METRIC_FUNCTIONS[function_name] for function_name in name_list}
        except KeyError as e:
            print('Metric not found, available metrics:')
            print('\n'.join(self.METRIC_FUNCTIONS.keys()))
            raise
    
    def calculate_metrics(self, moment: str, **kwargs):
        metrics_dict = {}
        for name, function in self.selected_metrics.items():
            if moment in name:
                metrics_dict.update(function(self, moment, **kwargs))
        
        return metrics_dict
    
    def register_function(name: str, func_dict: dict):
        def decorate(fnc: Callable):
            func_dict[name] = fnc
            return fnc
        return decorate
    
    def update_best(self, name: str, value: str, higher: bool = False) -> bool:
        if name in self.best_metrics.keys():
            if value > self.best_metrics[name]:
                if higher:
                    self.best_metrics[name] = value
                    return True
                return False
            if higher:
                return False
            self.best_metrics[name] = value
            return True
        self.best_metrics[name] = value
        return True
    
    def get_best(self, name: str) -> Union[float, int]:
        return self.best_metrics[name]
    
    @register_function('loss_train_step', METRIC_FUNCTIONS)        
    @register_function('loss_val_step', METRIC_FUNCTIONS)    
    @register_function('loss_train_epoch', METRIC_FUNCTIONS)
    @register_function('loss_val_epoch', METRIC_FUNCTIONS)   
    def loss(self, moment: str, **kwargs):
        loss_name = 'loss' + '_' + moment
        loss_value = kwargs['loss']
        self.update_best(loss_name, loss_value)
        
        return {loss_name: loss_value}
    
    @register_function('acc_train_epoch', METRIC_FUNCTIONS)
    @register_function('acc_val_epoch', METRIC_FUNCTIONS)
    def accuracy(self, moment: str, **kwargs):
        # Check if the output and label is a single tensor or a list of tensors
        # If it's a list, comebine them into a single tensor
        output = torch.cat(kwargs['outputs']) if kwargs.get('outputs', None) else kwargs['output'] 
        label = torch.cat(kwargs['labels']) if kwargs.get('labels', None) else kwargs['output']

        accuracy_name = 'accuracy' + '_' + moment
        n = len(output)
        
        if output[0].dim() > 0:
            # if multi-class output use this:
            accuracy =  (output.argmax() == label.argmax()).sum()/n
        else:
            # if single-class output use this:
            accuracy =  ((output > 0.5) == label).sum()/n
        
        self.update_best(accuracy_name, accuracy, True)
        
        return {accuracy_name: accuracy.item()}