Module base_project.run_files.run_manager
=========================================
File containing the RunManager class which handles the initaliazation of all components necessary
for a run and progresses the run.

Classes
-------

`RunManager(cfg: dict[str, any])`
:   This class initalizes all components necessary to execute the run, this is done following the
    configuration dictionary passed as an argument to the init function. The config file options and
    necessities are defined in a seperate file TODO: add file name.
    
    Attributes:
        cfg (dict[str, any]): configuration dictionary of the run
        logger (RunLogger, optional): logger that is used
        train_dataloader (DataLoader): dataloader containing the training data
        val_dataloader (DataLoader): dataloader containing the validation data
        model (nn.Module): the pytorch model that will be trained
        optimizer (torch.optim.Optimizer): optimizer that will be used
        loss_fn (torch.nn.modules.loss._Loss): loss function that will be used
        device (str): name of the device to train on
        metrics (Metrics): instance of Metrics that handles the metric calculation
        selection_metric (str): name of the metric to select the best model on
        goal (str): string defining if the selection metric should be maximized or minimized
        trainer (trainers.base_trainer.BaseTrainer): trainer handeling the training of the model
        epochs (int): the number of epochs to run
    
    Initializes the RunManager given the cfg parameter.
    
    Args:
        cfg (dict[str, any]): the configuration dictionary for the run

    ### Methods

    `start_training(self) -> NoneType`
    :   Starts the training and keeps running for the number of epochs in self.epochs, this method
        also handles the logic for saving the best model. NOTE: possibly seperate model saving into
        another method