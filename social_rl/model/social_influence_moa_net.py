from typing import Any, Dict, Optional, Sequence, Tuple, Union


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from tianshou.data import Batch, to_torch, to_numpy
from tianshou.utils.net.common import MLP


class SocialInfluenceMOAModule(nn.Module):
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
        action_dim: int,
        num_other_agents: int,
        flatten_dim: int,
        cnn_hidden_dim: int,
        cnn_output_dim: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        super().__init__()
        self.feature_net = feature_net
        self.fc = MLP(
            flatten_dim,
            output_dim=cnn_output_dim,
            hidden_sizes=[cnn_hidden_dim, cnn_hidden_dim],
            device=device
        )
        self.rnn = nn.GRU(
            input_size=cnn_output_dim + action_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            )
        self.rnn_num_layers = rnn_num_layers
        self.state = None
        self.rnn_fc = nn.Linear(
            rnn_hidden_size, 
            num_other_agents * action_dim
            )
        self.num_other_agents = num_other_agents
        self.action_dim = action_dim
        self.device = device            

    def forward(
        self, 
        curr_obs: torch.Tensor,
        prev_act: Union[torch.Tensor, np.ndarray],
        prev_other_act: Union[torch.Tensor, np.ndarray],
        state: Optional[torch.Tensor] = None,
        **kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Mapping: self obs and prev action -> other agents' actions"""        
        prev_act = to_torch(prev_act, dtype=torch.long, device=self.device)
        reshaped_logits = self.compute_logits(curr_obs, prev_act)
        # other agents' actions as ground truth
        prev_other_act = to_torch(prev_other_act, dtype=torch.long, device=self.device)
        total_loss = 0
        # cross-entropy loss for each agent
        for agent in range(self.num_other_agents):
            agent_logits = reshaped_logits[:, agent, :]
            agent_ground_truth = prev_other_act[:, agent]
            loss = F.cross_entropy(agent_logits, agent_ground_truth)
            total_loss += loss
        average_loss = total_loss / self.num_other_agents                        
        return average_loss
    
    def compute_logits(self, obs, act):
        # features for current          
        bs, ts, c, h, w = obs.shape
        phi1 = self.fc(self.feature_net(obs.reshape(bs * ts, c, h, w)))
        # self action
        act = F.one_hot(act, num_classes=self.action_dim).reshape(bs, -1)
        # (bs, t, feats)
        rnn_input = torch.cat([phi1, act], dim=1).unsqueeze(1)        
        if self.state is None:
            self.state = torch.zeros(self.rnn.num_layers, bs, self.rnn.hidden_size).to(self.device)        
        out, self.state = self.rnn(rnn_input, self.state)
        logits = self.rnn_fc(out[:, -1, :])                
        reshaped_logits = logits.reshape(bs, self.num_other_agents, self.action_dim)
        return reshaped_logits

    def compute_influence_reward(
            self, 
            curr_obs: torch.Tensor,
            prev_act: Union[torch.Tensor, np.ndarray],
            prev_other_act: Union[torch.Tensor, np.ndarray]
            ) -> torch.Tensor:
        with torch.no_grad():
            prev_act = to_torch(prev_act, dtype=torch.long, device=self.device)
            prev_other_act = to_torch(prev_other_act, dtype=torch.long, device=self.device)
            # other agents action logits given self action
            pred_logits = self.compute_logits(curr_obs, prev_act)
            pred_probs = F.softmax(pred_logits, dim=-1)
            # compute marginal probability of other agents' actions
            prev_act_logits = F.one_hot(prev_act, num_classes=self.action_dim)
            prev_act_probs = F.softmax(prev_act_logits.float(), dim=-1)
            marginal_probs = torch.sum(pred_probs * prev_act_probs, dim=-1)
            counter_fact_probs = torch.gather(pred_probs, dim=-1, index=prev_other_act.unsqueeze(-1)).squeeze(-1)
            kl_value = F.kl_div(counter_fact_probs.log(), marginal_probs, reduction='none').sum(dim=-1)            
        return to_numpy(kl_value)
