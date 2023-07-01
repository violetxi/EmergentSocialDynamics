"""Abstract base class for all agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, Tuple, Union, List
from typeguard import typechecked

from torch import Tensor
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import TensorDictReplayBuffer


@typechecked
class BaseAgent(ABC):
    def __init__(
            self,
            agent_idx: int,
            agent_id: str,
            policy: TensorDictModule,    # policy network
            value: TensorDictModule,     # value network
            qvalue: TensorDictModule,    # qvalue network
            world_model: TensorDictModule,   # world model
            replay_buffer_wm: TensorDictReplayBuffer,     # replay buffer
            replay_buffer_policy: TensorDictReplayBuffer,     # replay buffer
        ) -> None:
        """Base agent class
        args:
            agent_idx: agent_id in multi-agent setting, interpreted as index for a list of agents
            agent_id: the same as keys used in environment            
            policy: policy network
            value: value network
            qvalue: qvalue network
            world_model: world model
            replay_buffer_wm: replay buffer for training world model in decentralized fashion
            replay_buffer_policy: replay buffer for training policy in centralized fashion 
            (Different replay buffers are needed due to decentralized training paradigm and different 
            information required to train world model and policy network, respectively)
        """
        self.agent_idx = agent_idx
        self.agent_id = agent_id
        self.policy = policy
        self.world_model = world_model
        self.value = value
        self.qvalue = qvalue
        self.replay_buffer_wm = replay_buffer_wm
        self.replay_buffer_policy = replay_buffer_policy


    @abstractmethod
    def act(self, tensordict: TensorDict) -> Tensor:        
        """Output action given observation based on agent's policy
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
    def update_policy(self, tensordict: TensorDict) -> Dict[str, Any]:
        """Update policy parameters
        """
        raise NotImplementedError

    