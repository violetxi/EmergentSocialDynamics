import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from social_rl.runner.exp_runner import TrainRunner


# version 1.2 doesn't change cwd to output
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(args: DictConfig) -> None:
    n_agents = args.environment.base_env_kwargs.num_agents
    svo_values = [args.model.IMPolicy.args.svo for _ in range(n_agents)]
    OmegaConf.update(
        args, 
        'model.IMPolicy.args', 
        {'svo': svo_values}, 
        merge=True
        )
    train_runner = TrainRunner(args)
    train_runner.run()


if __name__ == "__main__":
    main()