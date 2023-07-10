from typing import Dict, Any, Optional, Tuple, Union, List
from typeguard import typechecked

import torch
from torch import nn
from tensordict import (
    TensorDict,
    TensorDictBase,
    nn as tdnn
)

from social_rl.models.cores import MLPModule
from social_rl.config.base_config import BaseConfig
from social_rl.utils.utils import convert_tensordict_to_tensor
from social_rl.models.wm_nets.dynamics_model_base import ForwardDynamicsModelBase



@typechecked
class MLPDynamicsModel(ForwardDynamicsModelBase):
    """MLP dynamics model for world model
        args: 
            agent_idx: index to help identify which obs and action tensor belong to this agent
            config: should be the callable config class
    """    
    def __init__(
            self, 
            agent_idx: int, 
            config: BaseConfig            
        ) -> None:
        self.agent_idx = agent_idx
        self.config = config
        backbone = MLPModule(self.config.backbone_kwargs)
        obs_head = MLPModule(self.config.obs_head_kwargs)
        action_head = MLPModule(self.config.action_head_kwargs)        
        action_dim = self.config.action_head_kwargs["out_features"]
        assert action_dim == self.config.backbone_kwargs["in_features"] \
            - self.config.obs_head_kwargs["out_features"], \
            "action_dim must be equal to the difference between backbone in_features and obs_head out_features"
        super().__init__(backbone, obs_head, action_head, action_dim=action_dim)        


    def forward_backbone(
            self, 
            obs: torch.Tensor, 
            action: torch.Tensor
        ) -> torch.Tensor:
        """Backbone is shared for obs and action prediction to create a latent representation of 
        agent's observations and actions
        args:
            obs (B, O) or (N-1, B, O): observation tensor self at timestep t
            action (B, A) or (N-1, B, A): action tensor self at timestep t

        return:
            x (B, F) or (N-1, B, F): latent representation of obs and action
        """
        #print("obs.shape", obs.shape, "action.shape", action.shape)
        assert len(obs.shape) == len(action.shape), \
            "obs and action must have same number of dimensions"
        return self.backbone(torch.cat([obs, action], dim=-1))


    def forward_obs_head(
            self, 
            obs: torch.Tensor, 
            action: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use agent's own observation (t) and action (t) to predict next observation
        args:
            obs (B, O): observation tensor for self at timestep t
            action (B, A): action tensor for self at timestep t

        return: 
            pred_obs(B, O): predicted observation tensor for self at timestep t+1
        """
        x = self.forward_backbone(obs, action)
        return x, self.obs_head(x)
    

    def forward_action_head(
            self, 
            obs: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:
        """Use other agents' observation (t) and action (t-1) to predict next action
        args:
            obs (N-1, B, A): observation tensor for other agents at timestep t
            action (N-1, B, A): action tensor for other agents at timestep t-1

        return:
            pred_action (B, N-1, A): predicted action tensor for other agents at timestep t
        """
        x = self.forward_backbone(obs, action)  
        return self.action_head(x)
    

    def compute_obs_loss(
            self, 
            obs: torch.Tensor, 
            pred_obs: torch.Tensor
        ) -> torch.Tensor:
        """Compute observation loss (MSE)        
        args:
            obs (B, O): observation tensor for self at timestep t+1
            pred_obs (B, O): predicted observation tensor for self at timestep t+1

        return:
            loss (B, 1): observation loss
        """
        return nn.functional.mse_loss(pred_obs, obs, reduction="none")
    

    def compute_action_loss(self, action: torch.Tensor, pred_action: torch.Tensor) -> torch.Tensor: 
        """Compute action loss (CE) assuming discrete action space
        args:
            action (N-1, B, A): action tensor for other agents at timestep t
            pred_action (N-1, B, A): predicted action tensor for other agents at timestep t

        return:
            loss (B, 1): action loss, we don't compute mean loss across batch because of tensor 
            dict elements needs to be of the same size
        """
        # if len(action.shape) == 3:
        #     action = action.reshape(-1, action.size(-1))
        # batch_first
        batch_size = action.size(1)        
        action_labels = torch.argmax(action, dim=-1).reshape(-1)
        pred_actions = pred_action.permute(1, 0, 2).reshape(-1, pred_action.size(-1))
        ce_loss = nn.functional.cross_entropy(pred_actions, action_labels, reduction="none")        
        return ce_loss.reshape(batch_size, -1)
    

    def _permute_and_reshape(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:           
            return x.permute(1, 0, 2)
        return x
            

    def forward(
            self, 
            obs: torch.Tensor, 
            actions: torch.Tensor, 
            prev_actions: torch.Tensor, 
            next_obs: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
        """Compute loss for both observation and action, loss will not be reduced until 
        optimization step due to tensordict elements needs to be of the same batch size
        args (all tensors are for all agents):            
            obs (N, B, O): observation at timestep t
            actions (N, B, A): action tensor at timestep t
            prev_actions (N, B, A): action tensor at timestep t-1
            next_obs (N, B, O): observation at timestep t+1

        return:
            out_dict: dictionary of losses  
        """        
        out_dict = {}
        loss_dict = {}
        # batch_size first
        self_obs = self._permute_and_reshape(obs[self.agent_idx, :, :])    # (B, N, O)
        self_actions = self._permute_and_reshape(actions[self.agent_idx, :, :])
        self_next_obs = self._permute_and_reshape(next_obs[self.agent_idx, :, :])
        latent, pred_obs = self.forward_obs_head(self_obs, self_actions)
        obs_loss = self.compute_obs_loss(self_next_obs, pred_obs)
        
        out_dict['latent'] = latent
        loss_dict['obs_loss'] = obs_loss

        # create a mask that is True for all agents except the one at agent_index
        num_agents = obs.size(0)
        mask = torch.ones(num_agents, dtype=bool)
        mask[self.agent_idx] = 0        
        other_obs = obs[mask, :, :]
        other_prev_actions = prev_actions[mask, :, :]
        other_action = actions[mask, :, :]
        pred_action = self.forward_action_head(other_obs, other_prev_actions)        
        action_loss = self.compute_action_loss(other_action, pred_action)
        loss_dict['action_loss'] = action_loss        
        
        out_dict['loss_dict'] = loss_dict
        return out_dict    



@typechecked
class MLPDynamicsTensorDictModel(tdnn.TensorDictModule):
    """Wrapper for a forward dynamics model to allow it to be compatible with the TensorDictModel API.
    Args:
        module: The forward dynamics model.
        in_keys: The input keys.
        out_keys: The output keys.
    """
    def __init__(
        self,
        module: nn.Module,
        in_keys: List[str],
        out_keys: List[str]
    ) -> None:
        super().__init__(module, in_keys, out_keys)

    
    def forward(
        self,
        tensordict: TensorDict,
        tensordict_out: Optional[TensorDictBase] = None,
        **kwargs,
    ) -> TensorDict:
        """Forward pass for the model.
        Args:
            tensordict: Input tensor dictionary.
            tensordict_out: Output tensor dictionary.
            **kwargs: Additional arguments.
        Returns:
            Output tensor dictionary.
        """        
        if tensordict_out is None:
            tensordict_out = tensordict.clone()

        # obs: torch.Tensor, actions: torch.Tensor, 
        # prev_actions: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:        
        obs = convert_tensordict_to_tensor(
            tensordict.get('observation'), "obs"
            ).to(tensordict.device)
        next_obs = convert_tensordict_to_tensor(
            tensordict.get(('next', 'observation')), "obs"
            ).to(tensordict.device)
        action = convert_tensordict_to_tensor(
            tensordict.get('action'), "action", self.module.action_dim
            ).to(tensordict.device)
        prev_action= convert_tensordict_to_tensor(
            tensordict.get('prev_action'), "action", self.action_dim
            ).to(tensordict.device)
        wm_dict = self.module(obs, action, prev_action, next_obs)

        assert list(wm_dict.keys()) == self.out_keys, \
            "The output keys of the module must match the out_keys of the TensorDictModel."
        
        for k in self.out_keys:
            tensordict_out[k] = wm_dict[k]

        return tensordict_out
    

    def loss(self, tensordict: TensorDict) -> tuple[
            Dict[str, torch.Tensor],
            TensorDict
    ]:
        tensordict_out = self(tensordict)        
        action_loss = tensordict_out['loss_dict']['action_loss'].mean()
        obs_loss = tensordict_out['loss_dict']['obs_loss'].mean()
        loss = action_loss + obs_loss        
        loss_dict = {
            'action_loss': action_loss,
            'obs_loss': obs_loss,
            'loss': loss
        }
        return loss_dict, tensordict_out
        