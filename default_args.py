from typeguard import typechecked
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class DefaultGlobalArgs:
    """
    Default global arguments for all algorithms 
    """
    reward_threshold: float = 4500
    seed: int = 1626
    buffer_size: int = 20_000
    lr: float = 1e-4
    gamma: float = 0.99    # discount factor
    epoch: int = 1   #1000
    step_per_epoch: int = 5 #5000
    # number of steps per collect, in collector steps in env is actually
    # divided by number of envs in the vectorized environment
    #step_per_collect: int = 2000
    step_per_collect: int = None
    episode_per_collect: int = 1
    repeat_per_collect: int = 10
    batch_size: int = 2048
    # number of train and test envs in the vectorized environment
    train_env_num: int = 10
    test_env_num: int = 2
    # number of episode used to evaluate trained agents after training is finished
    eval_eps: int = 3
    logdir: str = 'log'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ckpt_dir can be provided when continue to train or only doing evaluation
    ckpt_dir: str = None
    resume_training: bool = False
    eval_only: bool = False
    # wandb logging
    save_interval: int = 1
    project_name: str = 'emergent-social-dynamics'
    
    
