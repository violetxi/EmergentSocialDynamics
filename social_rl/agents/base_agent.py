"""Abstract base class for all agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from typeguard import typechecked

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
            value: TensorDictModule,     # value network
            qvalue: TensorDictModule,    # qvalue network
            world_model: nn.Module, #TensorDictModule,   # world model
            replay_buffer_wm: TensorDictReplayBuffer,     # replay buffer
            replay_buffer_actor: TensorDictReplayBuffer,     # replay buffer
        ) -> None:
        """Base agent class
        args:
            agent_idx: agent_id in multi-agent setting, interpreted as index for a list of agents
            agent_id: the same as keys used in environment            
            actor: actor network
            value: value network
            qvalue: qvalue network
            world_model: world model
            replay_buffer_wm: replay buffer for training world model in decentralized fashion
            replay_buffer_actor: replay buffer for training actor in centralized fashion 
            (Different replay buffers are needed due to decentralized training paradigm and different 
            information required to train world model and actor network, respectively)
        """
        self.agent_idx = agent_idx
        self.agent_id = agent_id
        self.actor = actor
        self.world_model = world_model
        self.value = value
        self.qvalue = qvalue
        self.replay_buffer_wm = replay_buffer_wm
        self.replay_buffer_acotr = replay_buffer_actor


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

