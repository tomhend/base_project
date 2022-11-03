import settings_utils.utils as utils
import logging
from run_files.run_manager import RunManager


def run():
    logging.basicConfig(level=logging.INFO)
    cfg = utils.parse_cfg('settings_utils/config.yaml')

    run_manager = RunManager(cfg)
    run_manager.start_training()


if __name__ == '__main__':
    run()
