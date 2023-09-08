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
            input_size=cnn_output_dim + action_dim * (num_other_agents+1),
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            )                
        self.rnn_fc = nn.Linear(
            rnn_hidden_size, 
            num_other_agents * action_dim
            )
        self.rnn_num_layers = rnn_num_layers
        self.num_other_agents = num_other_agents        
        self.action_dim = action_dim
        self.state = None
        self.device = device            

    def forward(
        self, 
        curr_obs: torch.Tensor,
        prev_act: Union[torch.Tensor, np.ndarray],
        prev_other_act: Union[torch.Tensor, np.ndarray],
        current_other_act: Union[torch.Tensor, np.ndarray],
        state: Optional[torch.Tensor] = None,
        **kwargs: Any
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Mapping: self obs and prev action -> other agents' actions"""        
        prev_act = to_torch(prev_act, dtype=torch.long, device=self.device)
        prev_other_act = to_torch(prev_other_act, dtype=torch.long, device=self.device)
        current_other_act = to_torch(current_other_act, dtype=torch.long, device=self.device)
        all_act = torch.cat([prev_act, prev_other_act], dim=-1)
        reshaped_logits = self.compute_logits(curr_obs, all_act)
        # other agents' actions as ground truth        
        total_loss = 0
        # cross-entropy loss for each agent
        for agent in range(self.num_other_agents):
            agent_logits = reshaped_logits[:, agent, :]
            agent_ground_truth = current_other_act[:, agent]
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
            prev_other_act: Union[torch.Tensor, np.ndarray],
            action_logits: torch.Tensor,
            prev_visibility: Optional[Union[torch.Tensor, np.ndarray]] = None,
            ) -> torch.Tensor:
        """Compute influence reward for self conditioned on other agents' actions
        and self action
        :param torch.Tensor curr_obs: current observation.
        :param torch.Tensor prev_act: previous action.
        :param torch.Tensor prev_other_act: previous other agents' actions.
        :param torch.Tensor action_logits: logits of self action.
        """
        with torch.no_grad():
            prev_act = to_torch(prev_act, dtype=torch.long, device=self.device)
            prev_other_act = to_torch(prev_other_act, dtype=torch.long, device=self.device)
            all_act = torch.cat([prev_act, prev_other_act], dim=-1)            
            # compute counter factual probability of other agents' actions conditioned 
            # on all possible self actions
            counter_factual_logits = []
            for self_act in range(self.action_dim):
                self_act = torch.tensor(self_act).repeat(
                    prev_other_act.shape[0]).unsqueeze(1).to(self.device)
                all_act = torch.cat([self_act, prev_other_act], dim=-1)
                counter_factual_logit = self.compute_logits(curr_obs, all_act)
                counter_factual_logits.append(counter_factual_logit)
            # (bs, action_dim, num_other_agents, action_dim)
            counter_factual_logits = torch.stack(counter_factual_logits).permute(1, 0, 2, 3)
            counter_factiual_probs = F.softmax(counter_factual_logits, dim=-1)            
            # use agent's action as indices to select predicted logits for other agents            
            predicted_logits = torch.gather(
                counter_factual_logits, 1, prev_act.unsqueeze(2).unsqueeze(3).expand(
                -1, -1, self.num_other_agents, self.action_dim)
                ).squeeze(1)            
            pred_probs = F.softmax(predicted_logits, dim=-1)
            # compute marginal probability of other agents' actions                        
            act_probs = F.softmax(action_logits, dim=-1).unsqueeze(1).unsqueeze(1)
            marginal_probs = torch.sum(counter_factiual_probs * act_probs, dim=1)
            kl_value = F.kl_div(pred_probs.log(), marginal_probs, reduction='none').sum(dim=-1)
        if prev_visibility is not None:
            kl_value = kl_value * to_torch(prev_visibility, device=self.device)
        influence = kl_value.sum(dim=-1)
        return to_numpy(influence)
