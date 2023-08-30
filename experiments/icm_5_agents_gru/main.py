import os
import hydra
from omegaconf import DictConfig

from social_rl.runner.exp_runner import TrainRunner


# version 1.2 doesn't change cwd to output
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(args: DictConfig) -> None:    
    train_runner = TrainRunner(args)
    train_runner.run()


if __name__ == "__main__":
    main()