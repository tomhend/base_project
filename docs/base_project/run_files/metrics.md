Module base_project.run_files.metrics
=====================================
File with the Metrics class and the register_function helper function. These handle the calculation
of the desired metrics. Functions that calculate metrics are registered to the METRIC_FUNCTIONS
dictionary, from which they will be selected given the list in the config.yaml file.

When you have written a new function you an register it by adding the @register_function decorator
above the definition. The name determines when they are calculated, and should include 'train_step',
'val_step', 'train_epoch', or 'val_epoch'.

Functions
---------

    
`register_function(name: str, moment: str, func_dict: dict)`
:   Registers a function to a dictionary with the given name as key and the function as value.
    
    Args:
        name (str): name of the function
        moment (str): moment the function should be run, has to be in Moments
        func_dict (dict): dictionary the function should be added to

Classes
-------

`Metrics(name_list: list[str])`
:   The Metrics class handles the calculation of the metrics. On initialization it selects the
    metrics based on the name_list. The calculate_metrics function is called by the trainer at
    different moments, specified by the moment parameter.
    
    Attributes:
        selected_metrics (dict[str, Callable]): name-function pairs of the metrics to be calculated

    ### Class variables

    `METRIC_FUNCTIONS`
    :

    ### Methods

    `accuracy(self, moment: str, output: torch.Tensor = None, outputs: list[torch.Tensor] = None, label: torch.Tensor = None, labels: list[torch.Tensor] = None, **_) -> dict[str, float]`
    :   Calculates the accuracy of a classification. For binary classification it uses 0.5 as a
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

    `calculate_metrics(self, moment: str, **kwargs: dict[str, any]) -> dict[str, any]`
    :   Caculate all the metrics that are found in the self.selected_metrics dictionary if they have
        the moment parameter in their name.
        
        Args:
            moment (str): string containing the moment to look for in the metric name
            kwargs (dict[str, any]): kwargs to pass onto the metric function
        
        Returns:
            dict[str,any]: dictionary containing the calculated metrics name and value

    `loss(self, moment: str, loss: float, **_) -> dict[str, float]`
    :   Return the loss in a dictionary with the moment added to the name. This function only exists
        for consistency in the calculation of metrics, and does not do much.
        
        Args:
            moment (str): string containing the moment to look for in the metric name
            loss (float): the loss calculated by the trainer
        
        Returns:
            dict[str, float]: dictionary with the loss name and value

    `select_metrics(self, name_list: list[str]) -> dict[str, typing.Callable]`
    :   Select the metrics based on their name.
        
        Args:
            name_list (list[str]): list of names of metrics that should be run
        
        Returns:
            dict[str, Callable]: dictionary with the name-function pairs of metrics

`Moments()`
:   Class that holds the constants for the moments on which functions are run

    ### Class variables

    `TRAIN_EPOCH`
    :

    `TRAIN_STEP`
    :

    `VAL_EPOCH`
    :

    `VAL_STEP`
    :