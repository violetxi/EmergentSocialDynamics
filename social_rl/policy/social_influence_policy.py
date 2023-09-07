from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import BasePolicy
from tianshou.utils.net.discrete import IntrinsicCuriosityModule

from social_rl.model.social_influence_moa_net import SocialInfluenceMOAModule


class SocialInfluencePolicy(BasePolicy):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param BasePolicy policy: a base policy to add to (PPO)
    :param float svo: desire SVO for the agent.
    :param float reward_scale: scale for the intrinsic reward.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        reward_scale: float,
        lr_scale: float,
        model_lr: float,
        num_other_agents: int,
        action_dim: int,
        flatten_dim: int,
        cnn_hidden_dim: int,
        cnn_output_dim: int,
        rnn_hidden_size: int,
        rnn_num_layers: int,        
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy       
        self.reward_scale = reward_scale
        self.lr_scale = lr_scale
        self.model_lr = model_lr
        self.model = SocialInfluenceMOAModule(
            feature_net=self.policy.actor.preprocess.conv,    # sharing conv with policy
            action_dim=action_dim,
            num_other_agents=num_other_agents,
            flatten_dim=flatten_dim,
            cnn_hidden_dim=cnn_hidden_dim,
            cnn_output_dim=cnn_output_dim,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.model_lr
        )


    def train(
            self, 
            mode: bool = True
            ) -> "SocialInfluencePolicy":
        """Set the module in training mode."""
        self.policy.train(mode)
        self.training = mode
        self.model.train(mode)
        return self

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data by inner policy.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        return self.policy.forward(batch, state, **kwargs)

    def exploration_noise(
            self, 
            act: Union[np.ndarray, Batch],
            batch: Batch
            ) -> Union[np.ndarray, Batch]:
        return self.policy.exploration_noise(act, batch)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()

    def process_fn(
        self, 
        batch: Batch, 
        buffer: ReplayBuffer, 
        indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """         
        curr_obs = batch.obs.observation.curr_obs.cuda()
        prev_act = batch.obs.observation.self_actions
        other_act = batch.obs.observation.other_agent_actions
        intr_rew = self.model.compute_influence_reward(curr_obs, prev_act, other_act)
        intr_rew = intr_rew * self.reward_scale
        batch.policy = Batch(
            orig_rew=batch.rew,             
            intr_rew=intr_rew,
            )
        batch.rew = batch.rew + intr_rew
        # reset state for the inner policy
        # self.policy.actor.preprocess.state = None
        return self.policy.process_fn(batch, buffer, indices)

    def post_process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> None:
        """Post-process the data from the provided replay buffer.

        Typical usage is to update the sampling weight in prioritized
        experience replay. Used in :meth:`update`.
        """
        self.policy.post_process_fn(batch, buffer, indices)
        batch.rew = batch.policy.orig_rew  # restore original reward

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        res = self.policy.learn(batch, **kwargs)
        # moa forward
        self.model_optimizer.zero_grad()
        curr_obs = batch.obs.observation.curr_obs.cuda()
        prev_act = batch.obs.observation.self_actions
        other_act = batch.obs.observation.other_agent_actions
        # state reset happens for every batch
        moa_loss = self.model(curr_obs, prev_act, other_act) * self.lr_scale
        moa_loss.backward()
        self.model_optimizer.step()
        res.update(
            {"moa_loss": moa_loss.item()}
        )
        return res
