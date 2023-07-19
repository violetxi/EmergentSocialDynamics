""" Default configuration for simple_tag_v3 environment for agents using 
social influence as intrinsic reward.
"""
import argparse
import torch.nn as nn
from typeguard import typechecked
from typing import Any, Optional

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
from social_rl.models.wm_nets.social_influence_mlp_dynamics_model import (
    MOAModel,
    SocialInfluenceMLPTensorDictModel
)
from social_rl.environment.petting_zoo_base import PettingZooMPEBase
from social_rl.agents.social_influencce_agent import SocialInfluenceAgent


# @TODO: Use this to set obs_dim, action_dim, latent_dim, num_agents as they are shared
# across multiple compments' kwargs
from dataclasses import dataclass
@dataclass
class KeyParams:
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass



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
    def __init__(
            self, 
            agent_config: BaseConfig,
            args: argparse.Namespace
            ) -> None:
        self.env_name = "mpe"
        self.task_name = "simple_push_v3"
        self.env_class = PettingZooMPEBase        
        self.env_kwargs = dict(
            max_cycles=args.max_episode_len,
            continuous_actions=False
        )



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
        self.out_keys = ["action", "logits"]    # output logits for influence calculation
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
class WmConfig(BaseConfig):
    def __init__(self) -> None:
        self.obs_embedding_kwargs = dict(
            in_features=19,    # observation
            out_features=128,
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            device="cpu",
        )
        self.action_embedding_kwargs = dict(
            in_features=5,
            out_features=32,
            num_cells=[32, 32],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            device="cpu",
        )
        self.latent_head_kwargs = dict(
            in_features=128+32,
            out_features=128,    # latent_dim = 128
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            device="cpu",
        )
        self.moa_head_kwargs = dict(
            in_features=128,
            out_features=5,    # action_dim = 5
            num_cells=[128, 128],
            activation_class=nn.ReLU,
            dropout=0.2,
            layer_class=nn.Linear,
            #device="cpu",
        )
        self.wm_module_cls =  MOAModel
        self.in_keys = [
            "observation", "action", "prev_action", ("next", "observation")]
        self.out_keys = ["latent", "loss_dict", "moa_cf_logits"]
        self.wrapper_class = SocialInfluenceMLPTensorDictModel



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
        self.num_agents = 2
        self.actor_config = actor_config
        self.lr_actor = 1e-4
        self.value_config = value_config
        self.lr_value = 1e-4
        self.qvalue_config = qvalue_config
        self.lr_qvalue = 1e-4
        self.wm_config = wm_config
        self.lr_wm = 1e-4
        self.replay_buffer_config = replay_buffer_config
        self.agent_class = SocialInfluenceAgent
        self.intr_reward_weight = 1.0



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
        self.env_config = EnvConfig(self.agent_config, args)        