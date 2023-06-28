""" Default configuration for simple_tag_v3 environment.
"""
import torch.nn as nn
from typeguard import typechecked

from social_rl.environment.petting_zoo_base import PettingZooMPEBase


@typechecked
class EnvConfig:
    def __init__(self) -> None:
        self.env_name = "mpe"
        self.task_name = "simple_tag_v3"
        self.env_class = PettingZooMPEBase
        self.env_kwargs = dict(
            num_good=4, 
            num_adversaries=4,
            num_obstacles=3, 
            max_cycles=25, 
            continuous_actions=False
        )


class WmConfig:
    def __init__(self) -> None:
        self.backbone_kwargs = dict(
            out_features=128,
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.LazyLinear,
            device="cpu",
        )
        self.obs_head_kwargs = dict(
            in_features=128,
            out_features=32,
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            device="cpu",
        )
        self.action_head_kwargs = dict(
            in_features=128,
            out_features=5,
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            device="cpu",
        )


class PolicyConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class ActorConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class ExpConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class Config:
    def __init__(self) -> None:
        self.env_config = EnvConfig()
        self.wm_config = WmConfig()
        self.policy_config = PolicyConfig()
        self.actor_config = ActorConfig()
        self.exp_config = ExpConfig()