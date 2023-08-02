from tensordict import TensorDict
from torch import Tensor
from typeguard import typechecked
from typing import Dict, Optional

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer
# {"gamma": 0.99, "lmbda": 0.95} by default.. reference:
# https://pytorch.org/rl/reference/generated/torchrl.objectives.default_value_kwargs.html#torchrl.objectives.default_value_kwargs
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.modules.models.models import (
    ConvNet, 
    MLP
    )

from social_rl.config.base_config import BaseConfig
from social_rl.agents.base_agent import BaseAgent
from social_rl.models.wm_nets.conv_world_model import WorldModelTensorDictBase


class ConvEncoder(nn.Module):    
    def __init__(self, agent_idx, config):
        super(ConvEncoder, self).__init__()
        self.agent_idx = agent_idx
        self.conv = ConvNet(**config.obs_encoder_kwargs)
        self.fc = MLP(**config.encoder_latent_net_kwargs)

        # self.conv = ConvNet(
        #         in_features=1,    # grayscale, but the actual implemenation should include N stacked frames
        #         num_cells=[32, 32, 32, 32],
        #         kernel_sizes=[5, 3, 3, 3],
        #         strides=1,
        #         activation_class=nn.ReLU,
        #     )
        # self.fc = MLP(
        #     in_features=800,
        #     out_features=128,
        #     num_cells=[128, 128],
        #     activation_class=nn.ReLU
        #     )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.fc(x)
        out_dict = dict(
            latent=x,
            loss_dict={}
        )
        return out_dict


class ConvEncoderTD(WorldModelTensorDictBase):
    def __init__(
            self,
            module,
            in_keys,
            out_keys
    ) -> None:
        super().__init__(module, in_keys, out_keys)

    def forward(
            self, 
            tensordict: TensorDict,
            tensordict_out: TensorDict = None
            ) -> TensorDict:
        if tensordict_out is None:
            tensordict_out = tensordict.clone()

        obs = self.convert_tensordict_to_tensor(
            tensordict.get('observation'), "obs"
            ).to(tensordict.device)
        wm_dict = self.module(obs)

        assert set(list(wm_dict.keys())) == set(self.out_keys), \
            "The output keys of the module must match the out_keys of the TensorDictModel."
        for k in self.out_keys:
            tensordict_out[k] = wm_dict[k]

        return tensordict_out
    
    def loss(self, tensordict: TensorDict) -> TensorDict:
        tensordict_out = self(tensordict)
        return {}, tensordict_out
        


#@typechecked
class VanillaAgentMF(BaseAgent):
    """Vanilla agent: Baseline agent without any intrinsic motivation, only learns 
    to maximize extrinsic reward while learning to predict next state and other agents' actions
    """
    def __init__(
            self,            
            agent_idx: int, 
            agent_id: str, 
            config: BaseConfig,
            actor: TensorDictModule,             
            qvalue: TensorDictModule, 
            world_model: TensorDictModule, 
            replay_buffer: TensorDictReplayBuffer,
            value: Optional[TensorDictModule] = None
        ) -> None:
        super().__init__(
            agent_idx, 
            agent_id, 
            actor,
            qvalue, 
            world_model,    # this will be called world model but just an encoder
            replay_buffer,
            value=value
            )
        self.config = config
        self.prep_optimization()    # initialize loss criterion and optimizer for world model and acotr network


    def prep_optimization(self) -> None:
        """Each agent gets its own loss criterion and optimizer for world model 
        and actor network        
        """        
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config.lr_wm)        
        self.sac_loss = DiscreteSACLoss(
            self.actor, 
            self.qvalue, 
            num_actions=self.actor.spec["action"].space.n
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)        
        self.qvalue_optimizer = torch.optim.Adam(self.qvalue.parameters(), lr=self.config.lr_qvalue)
        if hasattr(self, "value"):
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.config.lr_value)


    def act(self, tensordict: TensorDict) -> Tensor:
        # initial time step in the episode tensordict only has initial observation 
        # agent takes random action
        if "action" not in tensordict.keys():
            num_actions = self.config.actor_config.net_kwargs['out_features'] // 2            
            action = torch.randint(low=1, high=num_actions, size=tensordict.shape)
            return action
        else:
            # after initial time step, agent uses world model to predict next state
            # at this point tensordict has observation, action, next_obs, next_action, prev_action
            tensordict_wm = self.world_model(tensordict)
            tensordict_out = self.actor(tensordict_wm)
            return tensordict_out["action"]
        
    def update_wm_grads(self, tensordict: TensorDict) -> tuple[
        Dict,
        TensorDict]:
        """Update world model
        """
        wm_loss_dict, tensordict = self.world_model.loss(tensordict)
        return wm_loss_dict, tensordict

    def update_actor_grads(self, tensordict: TensorDict) -> Dict[str, float]:
        """Update actor network
        """
        self.wm_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.qvalue_optimizer.zero_grad()

        tensordict = self.sac_loss(tensordict)
        loss_actor = tensordict["loss_actor"]
        loss_qvalue = tensordict["loss_qvalue"]

        loss_actor.backward(retain_graph=True)
        loss_qvalue.backward(retain_graph=True)        

        return {
            "actor_loss": loss_actor.item(),
            "qvalue_loss": loss_qvalue.item(),
        }


    def step_optimizer(self) -> None:
        self.wm_optimizer.step()
        self.actor_optimizer.step()
        self.qvalue_optimizer.step()
        if hasattr(self, "value"):
            self.value_optimizer.step()