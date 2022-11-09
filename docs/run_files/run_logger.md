Module base_project.run_files.run_logger
========================================
File containing the 'weights and biases'/'wandb' related functions

Classes
-------

`RunLogger(cfg: dict[str, any])`
:   Class that handles the initialization of wandb and logging of metrics. Currently not very useful
    except for decoupling of the logging.
    
    Initialize a RunLogger instance that handles logging for a run
    
    Args:
        cfg (dict[str, any]): configuration file of the run

    ### Class variables

    `LOG_FUNCTIONS`
    :

    ### Methods

    `log_metrics(self, metrics_dict: dict[str, any]) -> NoneType`
    :   Log metrics to wandb.
        
        Args:
            metrics_dict (dict[str, any]): dictionary containing the name-value pairs of the metrics
                to log