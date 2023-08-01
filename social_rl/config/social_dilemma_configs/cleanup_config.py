""" Default configuration for cleanup environment. Using .py file because it's easier to 
pass around python objects (e.g. class Config) than json/yaml files.
Storage suggestion: store args information in a json file, along with the config.py file path, combining 
both will provide the full configuration for the experiment.
"""
import argparse
import torch.nn as nn
from typeguard import typechecked
from typing import Optional

import torchvision
torchvision.disable_beta_transforms_warning()    # v2 modules are still in beta, disable beta warning
from torchvision.transforms.v2 import (
    Compose, 
    Grayscale, 
    ToTensor
)
from torchrl.data import (
    TensorDictReplayBuffer, 
    LazyMemmapStorage
)
from torchrl.modules.tensordict_module.actors import (
    ProbabilisticActor,
    ValueOperator
)
from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec
from torchrl.modules.distributions.discrete import OneHotCategorical
from torchrl.modules.distributions.continuous import NormalParamWrapper


from social_rl.models.cores import MLPModule
from social_rl.config.base_config import BaseConfig
from social_rl.models.wm_nets.conv_world_model import (
    ConvWorldModel,
    WorldModelTensorDictBase
)
from social_rl.environment.social_dilemma_env import SocialDilemmaEnv
from social_rl.agents.vanilla_agent import VanillaAgent


# global constants for the environment and agents
NUM_AGENTS = 2
ACTION_DIM = 9
# models
LATENT_DIM = 128



@typechecked
class ExpConfig(BaseConfig):
    def __init__(
            self, 
            args: argparse.Namespace
            ) -> None:
        for attr in vars(args):
            setattr(self, attr, getattr(args, attr))            


@typechecked
class EnvConfig(BaseConfig):
    def __init__(self) -> None:
        self.env_name = "social_dilemma"
        self.task_name = "cleanup"
        self.env_class = SocialDilemmaEnv
        self.tranforms = Compose([            
            ToTensor(),
            Grayscale()
        ])
        self.env_kwargs = dict(
            num_agents=NUM_AGENTS,
            use_collective_reward=False,
            inequity_averse_reward=False,
            alpha=0.0,
            beta=0.0,
        )


@typechecked
class WmConfig(BaseConfig):
    def __init__(self) -> None:
        self.obs_encoder_kwargs = dict(
            in_features=1,    # grayscale, but the actual implemenation should include N stacked frames
            num_cells=[32, 32, 32, 32],
            kernel_sizes=[5, 3, 3, 3],
            strides=1,
            activation_class=nn.ReLU,
            )
        self.encoder_latent_net_kwargs = dict(
            in_features=800,    # flattened conv output dim
            out_features=LATENT_DIM,
            num_cells=[LATENT_DIM, LATENT_DIM],
            activation_class=nn.ReLU,
            )
        self.decoder_fc_kwargs = dict(
            in_features=LATENT_DIM + ACTION_DIM * NUM_AGENTS,    # (latent, prev_actions)
            out_features=800,
            num_cells=[LATENT_DIM, LATENT_DIM],
            activation_class=nn.ReLU,
            )                    
        self.obs_decoder_kwargs = dict(
            in_features=32,
            out_features=1,
            num_cells=[32, 32, 32, 32],
            kernel_sizes=[3, 3, 3, 3],           
            strides=1,
            paddings=0,
            activation_class=nn.ReLU,        
            )
        self.action_head_kwargs = dict(
            in_features=LATENT_DIM + ACTION_DIM * NUM_AGENTS,    # (latent, prev_actions)
            out_features=ACTION_DIM * NUM_AGENTS,
            num_cells=[LATENT_DIM, LATENT_DIM],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            )
        self.reward_head_kwargs = dict(
            in_features=LATENT_DIM + ACTION_DIM * NUM_AGENTS,    # (latent, predicted actions)
            out_features=1,
            num_cells=[LATENT_DIM, LATENT_DIM],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            device="cpu",
        )
        self.action_dim = ACTION_DIM
        self.wm_module_cls =  ConvWorldModel
        self.in_keys = [
            "observation", "action", "prev_action", ("next", "observation")]
        self.out_keys = ["latent", "loss_dict"]
        self.wrapper_class = WorldModelTensorDictBase


@typechecked
class ActorConfig(BaseConfig):
    """ Configuration for the actor (policy) network. """
    def __init__(self) -> None:
        self.net_module = MLPModule
        self.net_kwargs = dict(
            in_features = 128,
            out_features = 5 * 2,    # mean, std for action
            num_cells = [128, 128],    # number of hidden units in each layer
            activation_class = nn.ReLU,
            layer_class = nn.Linear
        )
        self.dist_wrapper = NormalParamWrapper 
        self.in_keys = ["latent"]
        self.intermediate_keys = ["logits"]
        self.out_keys = ["action"]
        self.action_spec = OneHotDiscreteTensorSpec(5)
        self.dist_class = OneHotCategorical
        self.wrapper_class = ProbabilisticActor



@typechecked
class QValueConfig(BaseConfig):
    """ Configuration for the Q-value network. 
        - Using MLPModule from torchrl
        - Using ValueOperator from torchrl
        - DiscreteSAC by default look for 'state_value' outkey        
    """
    def __init__(self) -> None:
        self.net_module = MLPModule
        self.net_kwargs = dict(
            in_features = 128,    # use latent representation from world model
            out_features = 5,    # value for actionss given current state
            num_cells = [128, 128],    # number of hidden units in each layer
            activation_class = nn.ReLU,
            layer_class = nn.Linear
        )
        self.in_keys = ['latent']
        self.out_keys = ['state_value']
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
            device="cpu",
        )
        self.buffer_kwargs = dict(
            batch_size=self.batch_size,
            storage=storage            
        )



@typechecked
class AgentConfig(BaseConfig):
    def __init__(
            self,
            actor_config: BaseConfig,            
            qvalue_config: BaseConfig,
            wm_config: BaseConfig,
            replay_buffer_config: BaseConfig,
            value_config: Optional[BaseConfig] = None
        ) -> None:
        self.num_agents = NUM_AGENTS
        self.actor_config = actor_config
        self.lr_actor = 1e-4
        self.value_config = value_config
        self.lr_value = 1e-4
        self.qvalue_config = qvalue_config
        self.lr_qvalue = 1e-4
        self.wm_config = wm_config
        self.lr_wm = 1e-4
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
        qvalue_config = QValueConfig()
        replay_buffer_config = ReplayBufferConfig(self.exp_config)
        self.agent_config = AgentConfig(
            actor_config,
            qvalue_config,
            wm_config,
            replay_buffer_config
        )
        self.env_config = EnvConfig()