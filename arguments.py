from typing import List
from dataclasses import field, dataclass
from hydra.core.config_store import ConfigStore


@dataclass
class ExpRunArguments:
    """
    Experiment run arguments
    """
    seed: int = field(
        default=1626, 
        metadata={'help': 'random seed'}
        )
