from tensordict import TensorDict
from torch import Tensor
from typeguard import typechecked
from typing import Dict

import torch
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer
from torchrl.objectives.sac import DiscreteSACLoss    #SACLoss

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
            value: TensorDictModule, 
            qvalue: TensorDictModule, 
            world_model: torch.nn.Module, #TensorDictModule, 
            replay_buffer_wm: TensorDictReplayBuffer, 
            replay_buffer_actor: TensorDictReplayBuffer
        ) -> None:
        super().__init__(agent_idx, agent_id, actor, value, qvalue, world_model, replay_buffer_wm, replay_buffer_actor)
        self.config = config
        self.prep_optimization()    # initialize loss criterion and optimizer for world model and acotr network


    def prep_optimization(self) -> None:
        """Each agent gets its own loss criterion and optimizer for world model 
        and actor network        
        """        
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config.lr_wm)
        
        #self.sac_loss = SACLoss(self.actor, self.qvalue, self.value)
        self.sac_loss = DiscreteSACLoss(
            self.actor, 
            self.qvalue, 
            num_actions=self.actor.spec["action"].space.n
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        #self.value_optimuzer = torch.optim.Adam(self.value.parameters(), lr=self.config.lr_value)
        self.qvalue_optimizer = torch.optim.Adam(self.qvalue.parameters(), lr=self.config.lr_qvalue)


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
    

    def update_wm(self, tensordict: TensorDict) -> tuple[
        Dict[str, torch.Tensor],
        TensorDict
        ]:
        """Update world model
        """
        self.wm_optimizer.zero_grad()
        loss_dict, tensordict_out = self.world_model.loss(tensordict)        
        loss = loss_dict["loss"]
        loss.backward(retain_graph=True)
        #self.wm_optimizer.step()
        return loss_dict, tensordict_out
        

    def update_actor(self, tensordict: TensorDict) -> Dict[str, float]:
        """Update actor network
        """
        self.actor_optimizer.zero_grad()
        self.qvalue_optimizer.zero_grad()

        tensordict = self.sac_loss(tensordict)
        loss_actor = tensordict["loss_actor"]
        loss_qvalue = tensordict["loss_qvalue"]

        loss_actor.backward(retain_graph=True)
        loss_qvalue.backward(retain_graph=True)

        print("Update world model")
        self.wm_optimizer.step()
        print("Update actor network")
        self.actor_optimizer.step()  
        print("Update qvalue network")      
        self.qvalue_optimizer.step()        
        return {
            "loss_actor": loss_actor.item(),
            "loss_qvalue": loss_qvalue.item(),
        }