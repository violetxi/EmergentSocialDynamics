"""Base classes for dynamics models of the world and other agents.
"""
from typeguard import typechecked
from abc import ABC, abstractmethod

import torch
from torch import nn


@typechecked
class ForwardDynamicsModelBase(nn.Module, ABC):
    def __init__(
            self,
            backbone: nn.Module,
            obs_head: nn.Module,
            action_head: nn.Module,
        ) -> None:
        super().__init__()
        self.backbone = backbone
        self.obs_head = obs_head
        self.action_head = action_head
    
    @abstractmethod
    def forward_obs_head(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward_action_head(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def forward_rollout(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
        