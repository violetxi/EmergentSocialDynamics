"""World model to predict 
1) self's next state given current state and action,
2) other's action given current state and past action (t-1)
This is used in VanillaAgent, and AdversarialWMIMAgent (vanilla wm)
"""
from typing import Dict, Any, Optional, Tuple, List
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



@typechecked
class MOAModel(nn.Module):
    """MLP MOA model used for SocialInfluenceAgent. It is trained to predict other agents'
    actions given self agent's observation and other agent's actions at previous 
    timestep.
        args: 
            agent_idx: index to help identify which obs and action tensor belong to this agent
            config: should be the callable config class
    """    
    def __init__(
            self, 
            agent_idx: int, 
            config: BaseConfig            
        ) -> None:
        super().__init__()
        self.agent_idx = agent_idx
        self.config = config
        self.obs_embedding = MLPModule(self.config.obs_embedding_kwargs)
        self.action_embedding = MLPModule(self.config.action_embedding_kwargs)
        self.action_dim = self.config.action_embedding_kwargs["in_features"]
        self.latent_head = MLPModule(self.config.latent_head_kwargs)
        self.moa_head = MLPModule(self.config.moa_head_kwargs)        


    def forward_obs_embedding(
            self, 
            obs: torch.Tensor
        ) -> torch.Tensor:
        """Compute embedding of observation
        args:
            obs (B, O) or (N-1, B, O): self observation self at timestep t           

        return:
            x (B, F) or (N-1, B, Eo): embedding of observation
        """
        return self.obs_embedding(obs)


    def forward_action_embedding(
            self,
            action: torch.Tensor
        ) -> torch.Tensor:
        """Compute embedding of action
        args:
            action (N-1, B, A): other agents' actions at timestep t-1
        
        return:
            x (N-1, B, Ea): embedding of action
        """
        action = action.to(self.action_embedding.model[0].weight.dtype)
        return self.action_embedding(action)
   

    def forward_latent_head(
            self,
            obs: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:
        """Compute latent embedding of observation and action, this will be the input 
        to policy and action head
        args:
            obs (B, O): self (learning) or others (SI) observation at timestep t
            action (N-1, B, A): other agent's actions at timestep t-1 (learning) 

        return:
            x (B, F): latent embedding of observation and action
        """
        x_obs = self.forward_obs_embedding(obs) # (B, F)
        x_action = self.forward_action_embedding(action) # (N-1, B, Ea)
        x = torch.cat((x_obs, x_action), dim=-1)
        return self.latent_head(x) # (N-1, B, F)


    def forward_moa_head(
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
        x = self.forward_latent_head(obs, action) # (N-1, B, F)
        return self.moa_head(x) # (N-1, B, A)
    

    def compute_action_loss(self, action: torch.Tensor, pred_action: torch.Tensor) -> torch.Tensor: 
        """Compute action loss (CE) assuming discrete action space
        args:
            action (N-1, B, A): action tensor for other agents at timestep t
            pred_action (N-1, B, A): predicted action tensor for other agents at timestep t

        return:
            loss (B, 1): action loss, we don't compute mean loss across batch because of tensor 
            dict elements needs to be of the same size
        """   
        batch_size = action.size(0)
        action_labels = torch.argmax(action, dim=-1).reshape(-1)
        pred_actions = pred_action.permute(1, 0, 2).reshape(-1, pred_action.size(-1))
        ce_loss = nn.functional.cross_entropy(pred_actions, action_labels, reduction="none")        
        return ce_loss.reshape(batch_size, -1)
    

    def _permute_and_reshape(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:           
            return x.permute(1, 0, 2)
        return x
            

    def compute_counter_factual_logits(
            self, 
            num_actions: int,
            other_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute counter factual action logits given other agents' observation
        args:
            other_obs (N-1, B, O): observation tensor for other agents at timestep t

        return:
            pred_action (B, N-1, A): predicted action tensor for other agents at timestep t
        """
        # write a one liner to create a list of five tensors of shape [B, N-1, num_actions]
        all_action_list = [
            torch.eye(num_actions)[action_idx] 
            for action_idx in range(num_actions)
            ]
        moa_logits = []
        for n in range(num_actions):
            self_action = all_action_list[n].unsqueeze(0).unsqueeze(0)            
            self_action = self_action.repeat(
                other_obs.size(0), other_obs.size(1), 1
                ).to(other_obs.device)
            moa_logits.append(self.forward_moa_head(other_obs, self_action))
        moa_logits = torch.stack(moa_logits) # (num_actions, N-1, B, A)
        counter_factual_logits = torch.mean(moa_logits, dim=0) # (N-1, B, A)
        counter_factual_logits = counter_factual_logits.permute(1, 0, 2) # (B, N-1, A)
        return counter_factual_logits


    def forward(
            self, 
            obs: torch.Tensor, 
            actions: torch.Tensor, 
            prev_actions: torch.Tensor,
            next_obs: Optional[torch.Tensor] = None    # added to be wrapped in MLPDynamicsTensorDictModel
        ) -> Dict[str, torch.Tensor]:
        """Compute loss for and action, loss will not be reduced until optimization step 
        due to tensordict elements needs to be of the same batch size args (all tensors 
        are for all agents):            
            obs (N, B, O): observation at timestep t
            actions (N, B, A): action tensor at timestep t
            prev_actions (N, B, A): action tensor at timestep t-1

        return:
            out_dict: dictionary of losses  
        """        
        out_dict = {}
        loss_dict = {}
        # Compute latent
        self_obs = self._permute_and_reshape(obs[self.agent_idx, :, :])    # (B, N, O)
        self_actions = self._permute_and_reshape(actions[self.agent_idx, :, :]) # (B, N, A)     
        latent = self.forward_latent_head(self_obs, self_actions) # (B, N, F)
        out_dict['latent'] = latent
        # Compute moa loss
        # create a mask that is True for all agents except the one at agent_index    
        num_agents = obs.size(0)
        mask = torch.ones(num_agents, dtype=bool)
        mask[self.agent_idx] = 0
        moa_obs = self_obs.unsqueeze(0).repeat_interleave(num_agents-1, dim=0) # (N-1, B, O)
        moa_obs = self._permute_and_reshape(moa_obs)
        other_prev_actions = self._permute_and_reshape(prev_actions[mask, :, :])
        other_action = self._permute_and_reshape(actions[mask, :, :])
        pred_action = self.forward_moa_head(moa_obs, other_prev_actions)        
        action_loss = self.compute_action_loss(other_action, pred_action)
        loss_dict['loss'] = action_loss        
        # Compute social influence reward
        num_actions = actions.size(-1)
        other_obs = obs[mask, :, :] # (N-1, B, O)
        other_obs = other_obs
        moa_cf_logits = self.compute_counter_factual_logits(num_actions, other_obs)
        out_dict['moa_cf_logits'] = moa_cf_logits
        out_dict['loss_dict'] = loss_dict        
        return out_dict    



@typechecked
class SocialInfluenceMLPTensorDictModel(tdnn.TensorDictModule):
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
        assert set(wm_dict.keys()) == set(self.out_keys), \
            "The output keys of the module must match the out_keys of the TensorDictModel."

        for k in self.out_keys:            
            tensordict_out[k] = wm_dict[k]

        return tensordict_out
    

    def loss(self, tensordict: TensorDict) -> tuple[
            Dict[str, torch.Tensor],
            TensorDict
    ]:
        tensordict_out = self(tensordict)
        loss = tensordict_out['loss_dict']['loss'].mean()
        loss_dict = {
            'loss': loss
            }
        return loss_dict, tensordict_out
        