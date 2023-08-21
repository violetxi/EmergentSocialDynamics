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
    epoch: int = 1000
    step_per_epoch: int = 5000    # number of train steps per epoch    
    # divided by number of envs in the vectorized environment
    step_per_collect: int = 2000   # number of transitions collected for all train envs
    episode_per_collect: int = None   # None means use step_per_collect, vice versa    
    repeat_per_collect: int = 10    # number of policy learning per collect
    batch_size: int = 2048
    # number of train and test envs in the vectorized environment
    train_env_num: int = 10
    test_env_num: int = 2
    test_eps: int = 2
    # number of episode used to evaluate trained agents after training is finished
    eval_eps: int = 4    # must be multiply of test_env_num
    logdir: str = 'log'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    # ckpt_dir can be provided when continue to train or only doing evaluation
    ckpt_dir: str = None
    resume_training: bool = False
    eval_only: bool = False
    # wandb logging
    update_interval: int = 1    # number of gradient steps for logging loss
    save_interval: int = 1
    project_name: str = 'emergent-social-dynamics'
    
    
