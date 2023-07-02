from tensordict import TensorDict
from torch import Tensor
from typeguard import typechecked
from typing import Dict, Any

import torch
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer
from torchrl.objectives.sac import SACLoss

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
            policy: TensorDictModule, 
            value: TensorDictModule, 
            qvalue: TensorDictModule, 
            world_model: torch.nn.Module, #TensorDictModule, 
            replay_buffer_wm: TensorDictReplayBuffer, 
            replay_buffer_policy: TensorDictReplayBuffer
        ) -> None:
        super().__init__(agent_idx, agent_id, policy, value, qvalue, world_model, replay_buffer_wm, replay_buffer_policy)
        self.config = config
        self.prep_optimization()    # initialize loss criterion and optimizer for world model and policy network


    def prep_optimization(self) -> None:
        """Each agent gets its own loss criterion and optimizer for world model 
        and policy network        
        """        
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config.lr_wm)
        
        self.sac_loss = SACLoss(self.policy, self.qvalue, self.value)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.config.lr_policy)
        self.value_optimuzer = torch.optim.Adam(self.value.parameters(), lr=self.config.lr_value)
        self.qvalue_optimizer = torch.optim.Adam(self.qvalue.parameters(), lr=self.config.lr_qvalue)


    def act(self, tensordict: TensorDict) -> Tensor:
        tensordict_out = self.policy(tensordict)
        return tensordict_out["action"]
    

    def update_wm(self, tensordict: TensorDict) -> Dict[str, float]:
        """Update world model
        """
        self.wm_optimizer.zero_grad()
        tensordict_out = self.world_model(tensordict)
        loss_dict = self.world_model.loss(tensordict_out, tensordict)
        loss = loss_dict["loss"]
        loss.backward()
        self.wm_optimizer.step()
        return loss_dict
    

    def update_policy(self, tensordict: TensorDict) -> Dict[str, Any]:
        """Update policy network
        """
        self.policy_optimizer.zero_grad()
        self.value_optimuzer.zero_grad()
        self.qvalue_optimizer.zero_grad()
        tensordict = self.sac_loss(tensordict)
        loss_actor = tensordict["loss_actor"]
        loss_qvalue = tensordict["loss_qvalue"]
        loss_value = tensordict["loss_value"]
        loss_actor.backward()
        loss_qvalue.backward()
        loss_value.backward()
        self.policy_optimizer.step()        
        self.qvalue_optimizer.step()
        self.value_optimuzer.step()        
        return {
            "loss_policy": loss_actor.item(),
            "loss_qvalue": loss_qvalue.item(),
            "loss_value": loss_value.item()
        }