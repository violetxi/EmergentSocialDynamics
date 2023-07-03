""" Default configuration for simple_tag_v3 environment. Using .py file because it's easier to 
pass around python objects (e.g. class Config) than json/yaml files.
Storage suggestion: store args information in a json file, along with the config.py file path, combining 
both will provide the full configuration for the experiment.
"""
import argparse
import torch.nn as nn
from typeguard import typechecked

from torchrl.data import (
    TensorDictReplayBuffer, 
    LazyMemmapStorage
)
from torchrl.modules.tensordict_module.actors import (
    Actor,
    ValueOperator
)

from social_rl.models.cores import MLPModule
from social_rl.config.base_config import BaseConfig
from social_rl.models.wm_nets.mlp_dynamics_model import (
    MLPDynamicsModel,
    MLPDynamicsTensorDictModel
)
from social_rl.environment.petting_zoo_base import PettingZooMPEBase
from social_rl.agents.vanilla_agent import VanillaAgent



@typechecked
class ExpConfig(BaseConfig):
    def __init__(
            self, 
            args: argparse.Namespace) -> None:
        for attr in vars(args):
            setattr(self, attr, getattr(args, attr))
        # @TODO: this will be changed after moving this to a server
        # identify the device to use in train.py then add it to namespacec args
        self.device = "cpu"


@typechecked
class EnvConfig(BaseConfig):
    def __init__(self, actor_config: BaseConfig) -> None:
        self.env_name = "mpe"
        self.task_name = "simple_tag_v3"
        self.env_class = PettingZooMPEBase
        self.obs_dim = 32
        self.action_dim = 5
        self.env_kwargs = dict(
            num_good=actor_config.num_good,
            num_adversaries=actor_config.num_adversaries,
            num_obstacles=3, 
            max_cycles=25, 
            continuous_actions=False
        )



@typechecked
class ActorConfig(BaseConfig):
    """ Configuration for the actor (policy) network. """
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
        self.in_keys = ["latent"]
        self.out_keys = ["action"] 
        self.wrapper_class = Actor



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
        self.in_keys = ["obs"]
        self.wrapper_class = ValueOperator
        


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
        self.in_keys = ['obs', 'action']
        self.wrapper_class = ValueOperator



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
class WmConfig(BaseConfig):
    def __init__(self) -> None:
        self.backbone_kwargs = dict(
            #in_features=32,
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
        self.wm_module_cls =  MLPDynamicsModel
        self.in_keys = ["obs", "action", "past_actions", "next_actions", "next_obs"]
        self.out_keys = ["latent", "loss_dict"]
        self.wrapper_class = MLPDynamicsTensorDictModel



@typechecked
class AgentConfig(BaseConfig):
    def __init__(
            self,
            actor_config: BaseConfig,
            value_config: BaseConfig,
            qvalue_config: BaseConfig,
            wm_config: BaseConfig,
            replay_buffer_config: BaseConfig
        ) -> None:
        self.num_good = 4
        self.num_adversaries = 4
        self.num_agents = self.num_good + self.num_adversaries
        self.actor_config = actor_config
        self.lr_actor = 1e-3
        self.value_config = value_config
        self.lr_value = 1e-3
        self.qvalue_config = qvalue_config
        self.lr_qvalue = 1e-3
        self.wm_config = wm_config
        self.lr_wm = 1e-3
        self.replay_buffer_config = replay_buffer_config
        self.agent_class = VanillaAgent



@typechecked
class Config(BaseConfig):
    def __init__(
            self,
            args: argparse.Namespace) -> None:
        self.exp_config = ExpConfig(args)
        wm_config = WmConfig()
        actor_config = ActorConfig()
        value_config = ValueConfig()
        qvalue_config = QValueConfig()
        replay_buffer_config = ReplayBufferConfig(self.exp_config)
        self.agent_config = AgentConfig(
            actor_config,
            value_config,
            qvalue_config,
            wm_config,
            replay_buffer_config
        )
        self.env_config = EnvConfig(self.agent_config)