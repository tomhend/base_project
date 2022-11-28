"""
File containing the 'weights and biases'/'wandb' related functions
"""
import wandb
import pandas as pd
import torch

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

    def watch_model(self, model: torch.nn.Module) -> None:
        """
        Logs information about the model to wandb

        Args:
            model (torch.nn.Module): the model that should be logged
        """
        wandb.watch(model, log='all', log_freq = 1)
    
    def log_metrics(self, metrics_dict: dict[str, any]) -> None:
        """
        Log metrics to wandb.

        Args:
            metrics_dict (dict[str, any]): dictionary containing the name-value pairs of the metrics
                to log
        """
        wandb.log(metrics_dict)

    def log_image(self, image: torch.Tensor) -> None:
        """
        Logs an image to wandb

        Args:
            image (torch.Tensor): image to log
        """
        image = wandb.Image(image, caption="validation_sample")
        wandb.log({"validation_sample": image})

    def log_out_label(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        Logs outputs and labels to a table in wandb

        Args:
            outputs (torch.Tensor): outputs to log
            labels (torch.Tensor): labels to log
        """
        out_label_df = pd.DataFrame(
            zip(outputs.detach().cpu().numpy(), labels.detach().cpu().numpy()),
            columns=["outputs", "labels"],
        )
        wandb.log({'Outputs and labels': out_label_df})
