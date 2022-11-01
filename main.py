import utils
import logging
from run_manager import RunManager

def run():
    logging.basicConfig(level=logging.INFO)
    cfg = utils.parse_cfg('config.yaml')
    
    run_manager = RunManager(cfg)
    run_manager.start_training()

if __name__ == '__main__':
    run()