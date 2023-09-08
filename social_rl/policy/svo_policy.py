from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch
from tianshou.policy import BasePolicy
from tianshou.utils.net.discrete import IntrinsicCuriosityModule


class SVOPolicy(BasePolicy):
    """Implementation of Intrinsic Curiosity Module. arXiv:1705.05363.

    :param BasePolicy policy: a base policy to add ICM to.
    :param float svo: desire SVO for the agent.
    :param float reward_scale: scale for the intrinsic reward.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        svo: float,
        reward_scale: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.policy = policy
        assert svo != -1, "svo must be set"
        self.svo = svo.pop()
        self.reward_scale = reward_scale

    def train(
            self, 
            mode: bool = True
            ) -> "SVOPolicy":
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
        # compute current SVO
        epsilon = 1e-10
        mean_other_rew = batch.obs.observation.other_rews.mean(axis=1)
        svo_hat = np.arctan2(mean_other_rew, batch.rew + epsilon)
        svo_hat = np.clip(svo_hat, 0, np.pi/2)
        self.mean_svo_hat = svo_hat.mean()
        intr_rew = np.abs(svo_hat - self.svo) * self.reward_scale        
        batch.policy = Batch(
            orig_rew=batch.rew,             
            intr_rew=intr_rew,
            )
        # add intr_rew and reward loss to reward
        batch.rew -= intr_rew
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
        res.update(
            {"mean_svo_hat": self.mean_svo_hat}
        )
        return res
