"""
File with the Metrics class and the register_function helper function. These handle the calculation
of the desired metrics. Functions that calculate metrics are registered to the METRIC_FUNCTIONS
dictionary, from which they will be selected given the list in the config.yaml file.

When you have written a new function you an register it by adding the @register_function decorator
above the definition. The name determines when they are calculated, and should include 'train_step',
'val_step', 'train_epoch', or 'val_epoch'.
"""
from typing import Callable
import torch


def register_function(name: str, func_dict: dict):
    """
    Registers a function to a dictionary with the given name as key and the function as value.

    Args:
        name (str): name of the function
        func_dict (dict): dictionary the function should be added to
    """

    def decorate(fnc: Callable):
        func_dict[name] = fnc
        return fnc

    return decorate


class Metrics:
    """
    The Metrics class handles the calculation of the metrics. On initialization it selects the
    metrics based on the name_list. The calculate_metrics function is called by the trainer at
    different moments, specified by the moment parameter.

    Attributes:
        selected_metrics (dict[str, Callable]): name-function pairs of the metrics to be calculated
    """

    METRIC_FUNCTIONS = {}

    def __init__(self, name_list: list[str]) -> None:
        self.selected_metrics = self.select_metrics(name_list)

    def select_metrics(self, name_list: list[str]) -> dict[str, Callable]:
        """
        Select the metrics based on their name.

        Args:
            name_list (list[str]): list of names of metrics that should be run

        Returns:
            dict[str, Callable]: dictionary with the name-function pairs of metrics
        """
        try:
            return {
                function_name: self.METRIC_FUNCTIONS[function_name]
                for function_name in name_list
            }
        except KeyError:
            print("Metric not found, available metrics:")
            print("\n".join(self.METRIC_FUNCTIONS.keys()))
            raise

    def calculate_metrics(
        self, moment: str, **kwargs: dict[str, any]
    ) -> dict[str, any]:
        """
        Caculate all the metrics that are found in the self.selected_metrics dictionary if they have
        the moment parameter in their name.

        Args:
            moment (str): string containing the moment to look for in the metric name
            kwargs (dict[str, any]): kwargs to pass onto the metric function

        Returns:
            dict[str,any]: dictionary containing the calculated metrics name and value
        """
        metrics_dict = {}
        for name, function in self.selected_metrics.items():
            if moment in name:
                metrics_dict.update(function(self, moment, **kwargs))

        return metrics_dict

    @register_function("loss_train_step", METRIC_FUNCTIONS)
    @register_function("loss_val_step", METRIC_FUNCTIONS)
    @register_function("loss_train_epoch", METRIC_FUNCTIONS)
    @register_function("loss_val_epoch", METRIC_FUNCTIONS)
    def loss(self, moment: str, loss: float, **_) -> dict[str, float]:
        """
        Return the loss in a dictionary with the moment added to the name. This function only exists
        for consistency in the calculation of metrics, and does not do much.

        Args:
            moment (str): string containing the moment to look for in the metric name 
            loss (float): the loss calculated by the trainer

        Returns:
            dict[str, float]: dictionary with the loss name and value
        """
        loss_name = "loss" + "_" + moment

        return {loss_name: loss}

    @register_function("acc_train_epoch", METRIC_FUNCTIONS)
    @register_function("acc_val_epoch", METRIC_FUNCTIONS)
    def accuracy(
        self,
        moment: str,
        output: torch.Tensor = None,
        outputs: list[torch.Tensor] = None,
        label: torch.Tensor = None,
        labels: list[torch.Tensor] = None,
        **_
    ) -> dict[str, float]:
        """
        Calculates the accuracy of a classification. For binary classification it uses 0.5 as a
        threshold, for multiclass classification the argmax is used to determine the prediction.
        Either <output and label> or <outputs and labels> should have values.
        
        Args:
            moment (str): string containing the moment to look for in the metric name 
            output (torch.Tensor, optional): output tensor of the model. Defaults to None
            outputs (list[torch.Tensor], optional): list of output tensors of the model. 
                Defaults to None
            label (torch.Tensor, optional): label tensor. Defaults to None
            labels (list[torch.Tensor], optional): list of label tensors. Defaults to None.

        Returns:
            dict[str, float]: dictionary containing the accuracy name and value pair
        """
        if outputs:
            output = torch.cat(outputs)
        if labels:
            label = torch.cat(labels)

        accuracy_name = "accuracy" + "_" + moment
        n_outputs = len(output)

        if output[0].dim() > 0:
            # if multi-class output use this:
            accuracy = (output.argmax() == label.argmax()).sum() / n_outputs
        else:
            # if single-class output use this:
            accuracy = ((output > 0.5) == label).sum() / n_outputs

        return {accuracy_name: accuracy.item()}
