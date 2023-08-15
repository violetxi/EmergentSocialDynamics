from typeguard import typechecked
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class DefaultGlobalArgs:
    """
    Default global arguments for all algorithms 
    """
    task: str = 'harvest'
    reward_threshold: float = 4500
    seed: int = 1626
    buffer_size: int = 20_000
    lr: float = 1e-4
    gamma: float = 0.99    # discount factor
    epoch: int = 1000
    step_per_epoch: int = 5000
    step_per_collect: int = 2000
    repeat_per_collect: int = 10
    batch_size: int = 2048
    train_num: int = 10
    test_num: int = 2
    logdir: str = 'log'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
