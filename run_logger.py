from select import select
from typing import Callable
import wandb

class RunLogger:
    LOG_FUNCTIONS = {}
    
    def __init__(self, log_cfg: dict) -> None:
        wandb.init(**log_cfg['wandb_init'])
        self.selected_log_functions = self.select_log_functions(log_cfg['log_fns'])
    
    def log(self, **kwargs):
        for function in self.selected_log_functions:
            function(self, **kwargs)
    
    def select_log_functions(self, function_list: list[str]):
        return [self.LOG_FUNCTIONS[function_name] for function_name in function_list]
    
    def register_function(name: str, func_dict: dict):
        def decorate(fnc: Callable):
            func_dict[name] = fnc
            return fnc
        return decorate
            
    @register_function('train_loss', LOG_FUNCTIONS)
    def log_train_loss(self, **kwargs):
        wandb.log({'train_loss': kwargs['train_loss']})
    
    
    
            
   