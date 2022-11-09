Module base_project.trainers.base_trainer
=========================================
The file containing the BaseTrainer class, which runs a basic pytorch training loop.

Classes
-------

`BaseTrainer(model: torch.nn.modules.module.Module, loss_fn: torch.nn.modules.loss._Loss, optimizer: torch.optim.optimizer.Optimizer, device: str, metrics: run_files.metrics.Metrics)`
:   Class that handles the training loop and optionally sends information to the logger.
    
    Attributes:
        model (nn.Module): the pytorch model that will be trained
        optimizer (torch.optim.Optimizer): optimizer that will be used
        loss_fn (torch.nn.modules.loss._Loss): loss function that will be used
        device (str): name of the device to train on
        metrics (Metrics): instance of Metrics that handles the metric calculation
        run_logger (RunLogger, optional): the logger that handles the logging of training
            and validation info
    
    Initializes an instance of BaseTrainer.
    
    Args:
        model (torch.nn.Module): model to train
        loss_fn (torch.nn.modules.loss._Loss): loss function to use
        optimizer (torch.optim.Optimizer): the optimizer to use
        device (str): the device to train on
        metrics (Metrics): the Metric instance to calculate the metrics

    ### Methods

    `set_run_logger(self, run_logger: run_files.run_logger.RunLogger) -> NoneType`
    :   Sets the run_logger.
        
        Args:
            run_logger (RunLogger): the RunLogger instance to use

    `train_epoch(self, dataloader: torch.utils.data.dataloader.DataLoader, epoch: int) -> dict[str, any]`
    :   Runs a training epoch for the model with the given dataloader.
        
        Args:
            dataloader (DataLoader): the dataloader containing the model input and labels
            epoch (int): the epoch number
        
        Returns:
            dict[str, any]: the metrics calculated at the end of the epoch

    `val_epoch(self, dataloader: torch.utils.data.dataloader.DataLoader, epoch: int) -> dict[str, any]`
    :   Runs a validation epoch for the model with the given dataloader.
        
        Args:
            dataloader (DataLoader): the dataloader containing the model input and labels
            epoch (int): the epoch number
        
        Returns:
            dict[str, any]: the metrics calculated at the end of the epoch