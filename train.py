"""
Trainer class and training loop are here
"""
import os
import argparse
from defaults import DEFAULT_ARGS
from typeguard import typechecked

from utils import load_config_from_path


@typechecked
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a MARL model')
    parser.add_argument(
        '--config', required=True, type=str, 
        default=None, help='Path to config file')
    args = parser.parse_args()
    return args


@typechecked
class Trainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = load_config_from_path(args.config)
        self._init_env()

    def _init_env(self) -> None:
        env_config = self.config.env_config
        self.env_name = env_config.env_name
        env_class = env_config.env_class
        self.env = env_class(env_config.__dict__)
        

if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    breakpoint()