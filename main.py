import yaml
import os
import logging
import time

from adversarialcv.train_runner import TrainRunner
from utils.config import load_config, seed_everything

def main() -> None:
    config_dir = "./configs/"
    base_config_path = os.path.join(config_dir, 'config.yaml')
    config = load_config(base_config_path, config_dir)

    seed_everything(config['env']['random_state'])
    os.makedirs(os.path.join(config['paths']['artifacts'], 'logs'), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        filename=os.path.join(config['paths']['artifacts'], 'logs', time.strftime("log_%Y-%m-%d_%H-%M-%S.log")), 
        filemode="w", 
        format="%(asctime)s %(levelname)s %(message)s", 
        encoding='utf-8'
        )
    logging.info(config)

    if(config['mode'] == 'train'):
        runner = TrainRunner(config)
        runner.run()
    else:
        raise Exception(f"Unsupported mode: {config['mode']}")

    logging.info('Complete!')

if __name__ == "__main__":
    main()