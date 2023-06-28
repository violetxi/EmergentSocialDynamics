from typing import Dict, Any, Callable, Optional, Tuple, Union, List
from torch import nn
from typeguard import typechecked

import torch
from torch import nn
from torchrl.modules.models import MLP
from social_rl.world_model.dynamics_model_base import ForwardDynamicsModelBase



@typechecked
class MLPModule(nn.Module):
    def __init__(self, **kwargs: Dict) -> None:
        """MLP module for world model
        kwargs: should be instantiated in config file
        """
        super().__init__()        
        self.model = MLP(**kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



@typechecked
class MLPDynamicsModel(ForwardDynamicsModelBase):
    """MLP dynamics model for world model
        args: 
            agent_idx: index to help identify which obs and action tensor belong to this agent
            config: should be the callable config class
    """    
    def __init__(self, agent_idx: int, config: Callable) -> None:
        self.agent_idx = agent_idx
        self.config = config
        backbone = MLPModule(**self.config.backbone_kwargs)
        obs_head = MLPModule(**self.config.obs_head_kwargs)
        action_head = MLPModule(**self.config.action_head_kwargs)
        super().__init__(backbone, obs_head, action_head)


    def forward_backbone(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Backbone is shared for obs and action prediction to create a latent representation of 
        agent's observations and actions
        args:
            obs (B, O) or (B, N-1, O): observation tensor self at timestep t
            action (B, A) or (B, N-1, A): action tensor self at timestep t

        return:
            x (B, F) or (B, N-1, F): latent representation of obs and action
        """
        assert len(obs.shape) == len(action.shape), \
            "obs and action must have same number of dimensions"
        
        if len(obs.shape) == 3:
            obs = obs.reshape(obs.size(0), -1)
        return self.backbone(torch.cat([obs, action], dim=-1))


    def forward_obs_head(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Use agent's own observation (t) and action (t) to predict next observation
        args:
            obs (B, O): observation tensor for self at timestep t
            action (B, A): action tensor for self at timestep t

        return: 
            pred_obs(B, O): predicted observation tensor for self at timestep t+1
        """
        x = self.forward_backbone(obs, action)
        return self.obs_head(x)
    

    def forward_action_head(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Use other agents' observation (t) and action (t-1) to predict next action
        args:
            obs (B, N-1, A): observation tensor for other agents at timestep t
            action (B, N-1, A): action tensor for other agents at timestep t-1

        return:
            pred_action (B, N-1, A): predicted action tensor for other agents at timestep t
        """
        x = self.forward_backbone(obs, action)  
        return self.action_head(x)
    
    def compute_obs_loss(self, obs: torch.Tensor, pred_obs: torch.Tensor) -> torch.Tensor:
        """Compute observation loss (MSE)
        args:
            obs (B, O): observation tensor for self at timestep t+1
            pred_obs (B, O): predicted observation tensor for self at timestep t+1

        return:
            loss (1): observation loss
        """
        return torch.mean((obs - pred_obs)**2)
    
    def compute_action_loss(self, action: torch.Tensor, pred_action: torch.Tensor) -> torch.Tensor: 
        """Compute action loss (CE) assuming discrete action space
        args:
            action (B, N-1, A): action tensor for other agents at timestep t
            pred_action (B, N-1, A): predicted action tensor for other agents at timestep t

        return:
            loss (1): action loss
        """
        return torch.nn.functional.cross_entropy(pred_action, action)
    
    def forward(self, obs: torch.Tensor, actions: torch.Tensor, prev_actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
        """Compute loss for both observation and action
        args (all tensors are for all agents):            
            obs (B, N, O): observation at timestep t
            actions (B, N, A): action tensor at timestep t
            prev_actions (B, N, A): action tensor at timestep t-1
            next_obs (B, N, O): observation at timestep t+1

        return:
            loss_dict: dictionary of losses  
        """
        loss_dict = {}
        num_agents = obs.size(1)
        self_obs = obs[:, self.agent_idx, :]
        self_actions = actions[:, self.agent_idx, :]
        self_next_obs = next_obs[:, self.agent_idx, :]
        pred_obs = self.forward_obs_head(self_obs, self_actions)
        obs_loss = self.compute_obs_loss(self_next_obs, pred_obs)
        loss_dict['obs_loss'] = obs_loss

        # create a mask that is True for all agents except the one at agent_index
        mask = torch.ones(num_agents, dtype=bool)
        mask[self.agent_idx] = 0
        other_obs = obs[:, mask, :]
        other_prev_actions = prev_actions[:, mask, :]
        other_actions = actions[:, mask, :]
        pred_action = self.forward_action_head(other_obs, other_prev_actions)
        action_loss = self.compute_action_loss(other_actions, pred_action)
        loss_dict['action_loss'] = action_loss

        loss = obs_loss + action_loss
        loss_dict['loss'] = loss
        return loss_dict


