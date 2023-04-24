"""
Main file, is the file that is run from the command line to initialize the run.
"""

import logging

from settings_utils import utils
from run_files.run_manager import RunManager
from pathlib import Path


def run():
    """
    Runs the program
    """
    logging.basicConfig(level=logging.INFO)
    cfg = utils.parse_cfg(Path("settings_utils/config.yaml"))

    run_manager = RunManager(cfg)
    run_manager.start_training()


if __name__ == "__main__":
    run()
