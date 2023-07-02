"""Core modules for all networks with different architectures
"""
from typeguard import typechecked
from typing import Dict, Any, Callable, Optional, Tuple, Union, List

import torch
from torch import nn
from torchrl.modules.models import MLP

@typechecked
class MLPModule(nn.Module):
    def __init__(self, kwargs: Dict) -> None:
        """MLP module for world model
        kwargs: should be instantiated in config file
        """
        super().__init__()        
        self.model = MLP(**kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    