from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import BasePolicy
from tianshou.utils.net.discrete import IntrinsicCuriosityModule


class IMRewardPolicy(BasePolicy):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param BasePolicy policy: a base policy to add ICM to.
    :param IntrinsicCuriosityModule model: the ICM model.
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param float lr_scale: the scaling factor for ICM learning.
    :param float forward_loss_weight: the weight for forward model loss.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        model: IntrinsicCuriosityModule,
        optim: torch.optim.Optimizer,
        lr_scale: float,
        reward_scale: float,
        forward_loss_weight: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        self.model = model
        self.optim = optim
        self.lr_scale = lr_scale
        self.reward_scale = reward_scale
        self.forward_loss_weight = forward_loss_weight

    def train(self, mode: bool = True) -> "IMRewardPolicy":
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

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        return self.policy.exploration_noise(act, batch)

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        if hasattr(self.policy, "set_eps"):
            self.policy.set_eps(eps)  # type: ignore
        else:
            raise NotImplementedError()

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        """Pre-process the data from the provided replay buffer.

        Used in :meth:`update`. Check out :ref:`process_fn` for more information.
        """        
        curr_obs = batch.obs.observation.curr_obs
        curr_act = batch.act
        next_obs = batch.obs_next.observation.curr_obs
        prev_other_act = batch.obs.observation.other_agent_actions
        rew = batch.rew
        forward_mse_loss, rew_mse_loss, act_hat = self.model(
            curr_obs, curr_act, next_obs, prev_other_act, rew
            )                
        batch.policy = Batch(
            orig_rew=batch.rew, 
            act_hat=act_hat, 
            forward_mse_loss=forward_mse_loss,
            rew_mse_loss=rew_mse_loss
            )
        # add forwrad and reward loss to reward
        batch.rew += to_numpy((forward_mse_loss + rew_mse_loss) * self.reward_scale)
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
        self.optim.zero_grad()
        act_hat = batch.policy.act_hat
        act = to_torch(batch.act, dtype=torch.long, device=act_hat.device)
        inverse_loss = F.cross_entropy(act_hat, act).mean()
        forward_loss = batch.policy.forward_mse_loss.mean()
        rew_loss = batch.policy.rew_mse_loss.mean()
        # @TODO: try different weighting for rew loss
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss +
            self.forward_loss_weight * forward_loss + rew_loss
        ) * self.lr_scale
        loss.backward()
        self.optim.step()
        res.update(
            {
                "loss/im_reward": loss.item(),
                "loss/im_reward/forward": forward_loss.item(),
                "loss/im_reward/inverse": inverse_loss.item(),
                "loss/im_reward/rew": rew_loss.item(),
            }
        )        
        return res