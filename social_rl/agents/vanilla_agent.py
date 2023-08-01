from tensordict import TensorDict
from torch import Tensor
from typeguard import typechecked
from typing import Dict, Optional

import torch
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer
# {"gamma": 0.99, "lmbda": 0.95} by default.. reference:
# https://pytorch.org/rl/reference/generated/torchrl.objectives.default_value_kwargs.html#torchrl.objectives.default_value_kwargs
from torchrl.objectives.sac import DiscreteSACLoss

from social_rl.config.base_config import BaseConfig
from social_rl.agents.base_agent import BaseAgent



@typechecked
class VanillaAgent(BaseAgent):
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
            world_model, 
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
        Dict[str, float],
        TensorDict
        ]:
        """Update world model
        """
        self.wm_optimizer.zero_grad()
        loss_dict, tensordict_out = self.world_model.loss(tensordict)        
        loss = loss_dict["loss"]
        loss.backward(retain_graph=True)
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
        return loss_dict, tensordict_out
        

    def update_actor_grads(self, tensordict: TensorDict) -> Dict[str, float]:
        """Update actor network
        """
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