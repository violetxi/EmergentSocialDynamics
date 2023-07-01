""" Default configuration for simple_tag_v3 environment. Using .py file because it's easier to 
pass around python objects (e.g. class Config) than json/yaml files.
Storage suggestion: store args information in a json file, along with the config.py file path, combining 
both will provide the full configuration for the experiment.
"""
import argparse
import torch.nn as nn
from typeguard import typechecked

from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from social_rl.models.cores import MLPModule
from social_rl.config.base_config import BaseConfig
from social_rl.models.wm_nets.mlp_dynamics_model import MLPDynamicsModel
from social_rl.environment.petting_zoo_base import PettingZooMPEBase



@typechecked
class EnvConfig(BaseConfig):
    def __init__(self, actor_config: BaseConfig) -> None:
        self.env_name = "mpe"
        self.task_name = "simple_tag_v3"
        self.env_class = PettingZooMPEBase
        self.env_kwargs = dict(
            num_good=actor_config.num_good,
            num_adversaries=actor_config.num_adversaries,
            num_obstacles=3, 
            max_cycles=25, 
            continuous_actions=False
        )


@typechecked
class WmConfig(BaseConfig):
    def __init__(self) -> None:
        self.backbone_kwargs = dict(
            in_features=32,
            out_features=128,
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
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
        self.wm_net_cls =  MLPDynamicsModel


@typechecked
class PolicyConfig(BaseConfig):
    def __init__(self) -> None:
        self.net_module = MLPModule
        self.net_kwargs = dict(
            in_features = 128,
            out_features = 5,    # number of actions
            num_cells = [128, 128],    # number of hidden units in each layer
            activation_class = nn.ReLU,
            dropout = 0.2,
            layer_class = nn.Linear
    )
        self.policy_cls_type = "dict"    # can be either TensorDictPolicy or TensorDictSequentialPolicy


@typechecked
class ValueConfig(BaseConfig):
    def __init__(self) -> None:
        self.net_module = MLPModule
        self.net_kwargs = dict(
            in_features = 128,    # use latent representation from world model
            out_features = 1,    # value for current state
            num_cells = [128, 128],    # number of hidden units in each layer
            activation_class = nn.ReLU,
            dropout = 0.2,
            layer_class = nn.Linear
    )
        self.in_keys = ["latent"]
        self.out_keys = ["actions"]
        


@typechecked
class QValueConfig(BaseConfig):
    def __init__(self) -> None:
        self.net_module = MLPModule
        self.net_kwargs = dict(
            in_features = 128,    # use latent representation from world model
            out_features = 5,    # value for actionss given current state
            num_cells = [128, 128],    # number of hidden units in each layer
            activation_class = nn.ReLU,
            dropout = 0.2,
            layer_class = nn.Linear
        )



@typechecked
class ReplayBufferConfig(BaseConfig):
    def __init__(
            self, 
            exp_config: BaseConfig) -> None:
        """Replay buffer config
            - Using TensorDictReplayBuffer from torchrl, which by default uses a random sampler
            with replacement if none is provided
            - There is an argument for multi-threading or processing (prefecth : int) to prefect 
            n batches
        """
        self.buffer_class = TensorDictReplayBuffer
        self.batch_size = exp_config.batch_size
        storage = LazyMemmapStorage(
            max_size=1e6,
            scratch_dir=exp_config.log_dir,
            device=exp_config.device,
        )
        self.buffer_kwargs = dict(
            batch_size=self.batch_size,
            storage=storage            
        )


@typechecked
class AgentConfig(BaseConfig):
    def __init__(
            self,
            policy_config: BaseConfig,
            value_config: BaseConfig,
            qvalue_config: BaseConfig,
            wm_config: BaseConfig,
            replay_buffer_config: BaseConfig
        ) -> None:
        self.num_good = 4
        self.num_adversaries = 4
        self.num_agents = self.num_good + self.num_adversaries
        self.policy_config = policy_config
        self.value_config = value_config
        self.qvalue_config = qvalue_config
        self.wm_config = wm_config
        self.replay_buffer_config = replay_buffer_config


@typechecked
class ExpConfig(BaseConfig):
    def __init__(
            self, 
            args: argparse.Namespace) -> None:
        for attr in vars(args):
            setattr(self, attr, getattr(args, attr))


@typechecked
class Config(BaseConfig):
    def __init__(
            self,
            args: argparse.Namespace) -> None:
        self.exp_config = ExpConfig(args)
        self.wm_config = WmConfig()
        policy_config = PolicyConfig()
        value_config = ValueConfig()
        qvalue_config = ValueConfig()
        self.agent_config = AgentConfig(
            policy_config,
            value_config,
            qvalue_config
        )
        self.env_config = EnvConfig(self.actor_config)
        
           
        