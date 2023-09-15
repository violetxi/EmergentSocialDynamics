import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from social_rl.runner.exp_runner import TrainRunner


def generate_population_svo(mean, std, n_agents):
    """Generate population of SVOs for each agent."""
    return np.random.normal(mean, std, n_agents)

# version 1.2 doesn't change cwd to output
@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(args: DictConfig) -> None:
    svo_values = generate_population_svo(
        args.model.IMPolicy.svo_mean, 
        args.model.IMPolicy.svo_std, 
        args.environment.base_env_kwargs.num_agents
        )
    OmegaConf.update(
        args, 
        'model.IMPolicy.args', 
        {'svo': svo_values.tolist()}, 
        merge=True
        )
    train_runner = TrainRunner(args)
    train_runner.run()


if __name__ == "__main__":
    main()