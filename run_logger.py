from typing import Callable
import wandb
import numpy as np
import torch


class RunLogger:
    LOG_FUNCTIONS = {}

    def __init__(self, cfg: dict) -> None:
        log_cfg = cfg["log_cfg"]
        wandb.init(**log_cfg["wandb_init"], config=cfg)
        self.selected_log_functions = self.select_log_functions(log_cfg["log_fns"])

    def log_train_step(self, **kwargs) -> None:
        log_dict = {}
        for name, function in self.selected_log_functions.items():
            if "train_step" in name:
                log_dict.update(function(self, "train_step", **kwargs))

        wandb.log(log_dict)

    def log_train_epoch(self, **kwargs) -> None:
        log_dict = {}
        for name, function in self.selected_log_functions.items():
            if "train_epoch" in name:
                log_dict.update(function(self, "train_epoch", **kwargs))

        wandb.log(log_dict, commit=False)

    def log_val_step(self, **kwargs) -> None:
        log_dict = {}
        for name, function in self.selected_log_functions.items():
            if "val_step" in name:
                log_dict.update(function(self, "val_step", **kwargs))

        wandb.log(log_dict)

    def log_val_epoch(self, **kwargs) -> None:
        log_dict = {}
        for name, function in self.selected_log_functions.items():
            if "val_epoch" in name:
                log_dict.update(function(self, "val_epoch", **kwargs))

        wandb.log(log_dict, commit=False)

    def select_log_functions(self, function_list: list[str]) -> dict[str, Callable]:
        try:
            return {
                function_name: self.LOG_FUNCTIONS[function_name]
                for function_name in function_list
            }
        except KeyError as e:
            print("Log function not found, available log functions:")
            print("\n".join(self.LOG_FUNCTIONS.keys()))
            raise

    def register_function(name: str, func_dict: dict):
        def decorate(fnc: Callable):
            func_dict[name] = fnc
            return fnc

        return decorate

    @register_function("loss_train_step", LOG_FUNCTIONS)
    @register_function("loss_val_step", LOG_FUNCTIONS)
    def loss(self, moment: str, **kwargs):
        loss_name = "loss" + "_" + moment
        return {loss_name: kwargs["loss"]}

    @register_function("loss_train_epoch", LOG_FUNCTIONS)
    @register_function("loss_val_epoch", LOG_FUNCTIONS)
    def avg_loss(self, moment: str, **kwargs):
        loss_name = "avg_loss" + "_" + moment
        loss_list = kwargs["losses"]
        return {loss_name: np.array(loss_list).mean()}

    def binary_accuracy(self, moment: str, **kwargs):
        accuracy_name = "bi_accuracy" + "_" + moment
        output = kwargs["output"]
        label = kwargs["label"]
        n = len(output)
        accuracy = ((output > 0.5) == label).sum() / n

        return {accuracy_name: accuracy.item()}

    @register_function("bi_acc_train_epoch", LOG_FUNCTIONS)
    @register_function("bi_acc_val_epoch", LOG_FUNCTIONS)
    def epoch_binary_accuracy(self, moment: str, **kwargs):
        outputs = torch.cat(kwargs["outputs"])
        labels = torch.cat(kwargs["labels"])

        return self.binary_accuracy(moment, output=outputs, label=labels)

    def mc_accuracy(self, moment: str, **kwargs):
        accuracy_name = "mc_accuracy" + "_" + moment
        output = kwargs["output"]
        label = kwargs["label"]
        n = len(output)
        accuracy = (output.argmax() == label.argmax()).sum() / n

        return {accuracy_name: accuracy.item()}

    @register_function("mc_acc_train_epoch", LOG_FUNCTIONS)
    @register_function("mc_acc_val_epoch", LOG_FUNCTIONS)
    def epoch_mc_accuracy(self, moment: str, **kwargs):
        outputs = torch.cat(kwargs["outputs"])
        labels = torch.cat(kwargs["labels"])

        return self.mc_accuracy(moment, output=outputs, label=labels)
