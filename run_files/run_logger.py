import wandb


class RunLogger:
    LOG_FUNCTIONS = {}

    def __init__(self, cfg: dict) -> None:
        log_cfg = cfg["log_cfg"]
        wandb.init(**log_cfg["wandb_init"], config=cfg)

    def log_metrics(self, metrics_dict: dict[str, any]) -> None:
        wandb.log(metrics_dict)
