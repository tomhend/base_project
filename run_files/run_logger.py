"""
File containing the 'weights and biases'/'wandb' related functions
"""
import wandb


class RunLogger:
    """
    Class that handles the initialization of wandb and logging of metrics. Currently not very useful
    except for decoupling of the logging.
    """

    LOG_FUNCTIONS = {}

    def __init__(self, cfg: dict[str, any]) -> None:
        """
        Initialize a RunLogger instance that handles logging for a run

        Args:
            cfg (dict[str, any]): configuration file of the run
        """
        log_cfg = cfg["log_cfg"]
        wandb.init(**log_cfg["wandb_init"], config=cfg)

    def log_metrics(self, metrics_dict: dict[str, any]) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics_dict (dict[str, any]): dictionary containing the name-value pairs of the metrics
                to log
        """
        wandb.log(metrics_dict)
