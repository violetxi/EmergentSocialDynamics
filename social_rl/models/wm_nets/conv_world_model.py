from typing import Dict, Any, Optional, Tuple, Union, List
from typeguard import typechecked

import torch
import torch.nn as nn
import torchrl
import tensordict as td
from tensordict import (
    TensorDict,
    TensorDictBase,
    nn as tdnn
)
from torchrl.modules.models.models import (
    ConvNet, 
    MLP
    )

from social_rl.models.cores import DeconvNet



class ConvWorldModel(nn.Module):
    """
    Convolutional World Model
    attributes:
        config: configuration object
        obs_encoder: encode stacked frames across multiple time steps as h
        encoder_latent: use h to output latent representation z
        decoder_fc: output representation of (z, past actions) to reconstruct the next frame
        obs_decoder: use h to reconstruct the next frame
        action_head: predict other's actions (oa) at current time step conditioned on z, a and others 
            past actions
        reward_head: predict future reward conditioned on z, a and oa to predict future reward
    """
    def __init__(self, agent_idx, config):
        super(ConvWorldModel, self).__init__()
        self.agent_idx = agent_idx                
        self.obs_encoder = ConvNet(**config.obs_encoder_kwargs)
        self.encoder_latent = MLP(**config.encoder_latent_net_kwargs)
        self.decoder_fc = MLP(**config.decoder_fc_kwargs)
        self.obs_decoder = DeconvNet(**config.obs_decoder_kwargs)
        self.action_head = MLP(**config.action_head_kwargs)
        self.reward_head = MLP(**config.reward_head_kwargs)
        self.action_dim = config.action_dim
        self.num_agents = self.action_head.out_features // self.action_dim        
    
    def encode(self, x):
        x = self.obs_encoder(x)
        x = x.view(x.size(0), -1)
        z = self.encoder_latent(x)
        return z
    
    def decode(self, x):
        x = self.decoder_fc(x)
        x = x.view(x.size(0), 32, 5, 5)
        x = self.obs_decoder(x)
        return x
    
    def predict_actions(self, x):
        """
        Predict current actions given past actions and latent
        """
        return self.action_head(x)
    
    def predict_rewards(self, x):
        """
        Predict future reward
        """
        return self.reward_head(x)
    
    def compute_rep_loss(self, pred_obs, next_obs):
        """
        Compute reconstruction loss
        """
        batch_size = pred_obs.shape[0]
        pred_obs = pred_obs.reshape(batch_size, -1)
        next_obs = next_obs.reshape(batch_size, -1)
        return nn.functional.mse_loss(pred_obs, next_obs, reduction='none')
    
    def compute_action_loss(self, pred_actions, actions):
        """
        Compute action loss
        args:
            pred_actions: predicted actions (B, N * A)
            actions: actions (B, N, A)
        """
        batch_size = pred_actions.shape[0]
        action_labels = torch.argmax(actions, dim=-1).reshape(-1)
        pred_actions = pred_actions.reshape(-1, self.action_dim)        
        action_loss = nn.functional.cross_entropy(pred_actions, action_labels, reduction='none')        
        return action_loss.reshape(batch_size, -1)
    
    def compute_reward_loss(self, pred_rewards, reward):
        """
        Compute reward loss
        """
        max_neg_reward = 50 * (self.num_agents - 1)
        pred_rewards_scaled = nn.functional.sigmoid(pred_rewards) \
            * (max_neg_reward + 1) - max_neg_reward
        reward_loss = torch.square(pred_rewards_scaled - reward)
        return reward_loss
    
    def forward(self, obs, prev_actions, actions, next_obs, reward):
        """
        Forward pass
        """
        batch_size = obs.shape[0]        
        latent = self.encode(obs)
        prev_actions = prev_actions.reshape(batch_size, -1)
        latent_prev_actions = torch.cat([latent, prev_actions], dim=-1)        
        pred_obs = self.decode(latent_prev_actions)
        pred_actions = self.predict_actions(latent_prev_actions)        
        pred_reward = self.predict_rewards(latent_prev_actions)
        # compute loss        
        rep_loss = self.compute_rep_loss(pred_obs, next_obs)
        action_loss = self.compute_action_loss(pred_actions, actions)
        reward_loss = self.compute_reward_loss(pred_reward, reward)

        out_dict = dict(
            loss_dict=dict(
                rep_loss=rep_loss,
                action_loss=action_loss,
                reward_loss=reward_loss,                
                ),
            latent=latent
            )
        return out_dict
    
        

class WorldModelTensorDictBase(tdnn.TensorDictModule):
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

    
    def convert_tensordict_to_tensor(
            self,
            tensordict: td.TensorDict, 
            td_type: str,
            action_dim: Optional[int] = None,
            ) -> torch.Tensor:
        """Convert a tensordict to a torch tensor
        Args:
            tensordict (td.TensorDict): a tensordict
            td_type (str): the type of tensordict (obs, act, rew etc)
            action_dim (Optional[int], optional): the action dimension to help create 
            one-hot encoding for actions
        Returns:
            tensor_out (torch.Tensor): a tensor of the input (B, N, D)
        """
        if isinstance(tensordict[list(tensordict.keys())[0]], td.MemmapTensor):
            # if input contains memmap, convert to tensor
            # as_tensor can only be called on tensors on cpu
            tensordict = tensordict.cpu()
            for k, v in tensordict.items():
                tensordict[k] = v.as_tensor()

        if td_type == "obs":            
            agent_id = list(tensordict.keys())[self.agent_idx]
            tensor_out = tensordict[agent_id]['curr_obs']            
            if len(tensor_out.shape) == 3:
                # add batch dimension if input is unbatched
                tensor_out = tensor_out.unsqueeze(0)
        elif td_type == "action":
            # assert action_dim is not None, \
            #     "action_dim must be provided for action conversion"
            tensor_out = torch.stack(list(tensordict.values()))    # (num_agents, )
            tensor_out = torch.nn.functional.one_hot(tensor_out, self.action_dim)    # (num_agents, action_dim)
            if len(tensor_out.shape) == 2:
                # add batch dimension if input is unbatched
                tensor_out = tensor_out.unsqueeze(0)
        elif td_type == "reward":
            agent_id = list(tensordict.keys())[self.agent_idx]            
            tensor_out = tensordict[agent_id]
            if len(tensor_out.shape) == 1:
                # add batch dimension if input is unbatched
                tensor_out = tensor_out.unsqueeze(0)    # (B, num_agents, 1)
        else:
            raise NotImplementedError(f"Conversion for td_type {td_type} not implemented")
            
        return tensor_out

    
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
        obs = self.convert_tensordict_to_tensor(
            tensordict.get('observation'), "obs"
            ).to(tensordict.device)
        next_obs = self.convert_tensordict_to_tensor(
            tensordict.get(('next', 'observation')), "obs"
            ).to(tensordict.device)
        action = self.convert_tensordict_to_tensor(
            tensordict.get('action'), "action", self.action_dim
            ).to(tensordict.device)
        prev_action= self.convert_tensordict_to_tensor(
            tensordict.get('prev_action'), "action", self.action_dim
            ).to(tensordict.device)
        if 'extr_reward' in tensordict.keys():
            reward = self.convert_tensordict_to_tensor(
                tensordict.get('extr_reward'), "reward"
                ).to(tensordict.device)
        else:
            reward = self.convert_tensordict_to_tensor(
                tensordict.get(('next', 'reward')), "reward"
                ).to(tensordict.device)
        
        wm_dict = self.module(obs, prev_action, action, next_obs, reward)        
        assert set(list(wm_dict.keys())) == set(self.out_keys), \
            "The output keys of the module must match the out_keys of the TensorDictModel."

        for k in self.out_keys:
            tensordict_out[k] = wm_dict[k]

        return tensordict_out
    

    def loss(self, tensordict: TensorDict) -> tuple[
            Dict[str, torch.Tensor],
            TensorDict
    ]:         
        tensordict_out = self(tensordict)
        loss_dict =  tensordict_out['loss_dict']
        rep_loss = loss_dict['rep_loss'].mean()
        action_loss = loss_dict['action_loss'].mean()
        reward_loss = loss_dict['reward_loss'].mean()    
        loss = rep_loss + action_loss + reward_loss

        loss_dict = {
            'rep_loss': rep_loss,
            'action_loss': action_loss,
            'reward_loss': reward_loss,
            'loss': loss
        }
        return loss_dict, tensordict_out
        