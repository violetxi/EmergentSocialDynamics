from typing import Any, Dict, Optional, Sequence, Tuple, Union


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch
from tianshou.utils.net.common import MLP


class IMRewardModule(nn.Module):
    """Model-based intrinsic motivation, forward, backward dynamics for self, other
    agents' actions and reward for self conditioned on current statee and other agents
    actions

    :param torch.nn.Module feature_net: a self-defined feature_net which output a
        flattened hidden state.
    :param int feature_dim: input dimension of the feature net.
    :param int action_dim: dimension of the action space.
    :param intn num_other_agents: number of other agents in the env.
    :param hidden_sizes: hidden layer sizes for forward and inverse models.
    :param device: device for the module.
    """

    def __init__(
        self,
        feature_net: nn.Module,
        feature_dim: int,
        action_dim: int,
        num_other_agents: int,
        hidden_sizes: Sequence[int] = (),
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        # predict next state given current state and action
        self.forward_model = MLP(
            feature_dim + action_dim,
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )
        # predict action at current step given current annd future state
        self.inverse_model = MLP(
            feature_dim * 2,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            device=device
        )        
        # predict reward given state and action given current state and 
        # other agents' actions
        self.reward_model = MLP(
            feature_dim + num_other_agents * action_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            device=device
        )
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.device = device

    def forward(
        self, 
        batch: Batch,        
        **kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Mapping: curr_obs, curr_act, next_obs -> mse_loss, pred_act."""
        curr_act = batch.act
        prev_other_act = batch.obs.observation.other_agent_actions
        rew = batch.rew
        # features for current and next obs    
        phi1, phi2 = self.feature_net(batch.obs), self.feature_net(batch.obs_next)
        # representation for state+action
        curr_act = to_torch(curr_act, dtype=torch.long, device=self.device)
        phi2_hat = self.forward_model(
            torch.cat([phi1, F.one_hot(curr_act, num_classes=self.action_dim)], dim=1)
        )
        forward_mse_loss = 0.5 * F.mse_loss(phi2_hat, phi2, reduction="none").sum(1)        
        act_hat = self.inverse_model(torch.cat([phi1, phi2], dim=1))
        # reward prediction        
        prev_other_act = to_torch(prev_other_act, dtype=torch.long, device=self.device)        
        bs, n_agents = prev_other_act.shape                
        prev_other_act_one_hot = F.one_hot(
            prev_other_act, num_classes=self.action_dim).reshape(bs, -1)
        rew = to_torch(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        pred_rew = self.reward_model(torch.cat([phi1, prev_other_act_one_hot], dim=1))
        rew_mse_loss = 0.5 * F.mse_loss(pred_rew, rew, reduction="none").sum(1)
        return forward_mse_loss, rew_mse_loss, act_hat