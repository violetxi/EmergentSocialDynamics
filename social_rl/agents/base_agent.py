"""Abstract base class for all agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from typeguard import typechecked

import torch
from torch import Tensor, nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import TensorDictReplayBuffer



@typechecked
class BaseAgent(ABC):
    def __init__(
            self,
            agent_idx: int,
            agent_id: str,
            actor: TensorDictModule,    # actor network            
            qvalue: TensorDictModule,    # qvalue network
            world_model: nn.Module, #TensorDictModule,   # world model
            replay_buffer: TensorDictReplayBuffer,     # replay buffer
            value: Optional[TensorDictModule] = None     # value network
        ) -> None:
        """Base agent class
        args:
            agent_idx: agent_id in multi-agent setting, interpreted as index for a list of agents
            agent_id: the same as keys used in environment            
            actor: actor network           
            qvalue: qvalue network
            world_model: world model
            replay_buffer: replay buffer for training world model in decentralized fashion            
            value: value network is not needed when agent's action space is discrete
        """
        self.agent_idx = agent_idx
        self.agent_id = agent_id
        self.actor = actor
        self.world_model = world_model        
        self.qvalue = qvalue
        self.replay_buffer = replay_buffer
        if value is not None:
            self.value = value


    def save_model_weights(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'qvalue_state_dict': self.qvalue.state_dict(),
            'world_model_state_dict': self.world_model.state_dict(),
            'value_state_dict': self.value.state_dict() if hasattr(self, "value") else None
        }, path)


    def load_model_weights(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.qvalue.load_state_dict(checkpoint['qvalue_state_dict'])
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        if hasattr(self, "value"):
            self.value.load_state_dict(checkpoint['value_state_dict'])

    
    def set_train(self) -> None:
        self.world_model.train()
        self.actor.train()
        self.qvalue.train()
        if hasattr(self, "value"):
            self.value.train()

    
    def set_eval(self) -> None:
        self.world_model.eval()
        self.actor.eval()
        self.qvalue.eval()
        if hasattr(self, "value"):
            self.value.eval()
        

    @abstractmethod
    def prep_optimization(seslf) -> None:
        """Each agent gets its own loss criterion and optimizer for world model 
        and actor network        
        """
        raise NotImplementedError
    

    @abstractmethod
    def act(self, tensordict: TensorDict) -> Tensor:        
        """Output action given observation based on agent's actor
        args:
            tensordict: a dictionary with observation
        return:
            action: action index
        """
        raise NotImplementedError
    

    @abstractmethod
    def update_wm(self, tensordict: TensorDict) -> Dict[str, Any]:
        """Update world model parameters
        """
        raise NotImplementedError


    @abstractmethod
    def update_actor(self, tensordict: TensorDict) -> Dict[str, Any]:
        """Update actor parameters
        """
        raise NotImplementedError

