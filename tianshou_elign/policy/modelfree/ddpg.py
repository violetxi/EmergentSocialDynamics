import torch
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from typing import Dict, Tuple, Union, Optional

from tianshou.policy import BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch_as


class DDPGPolicy(BasePolicy):
    """Implementation of Deep Deterministic Policy Gradient. arXiv:1509.02971
    :param torch.nn.Module actor: the actor network following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer actor_optim: the optimizer for actor network.
    :param torch.nn.Module critic: the critic network. (s, a -> Q(s, a))
    :param torch.optim.Optimizer critic_optim: the optimizer for critic
        network.
    :param float tau: param for soft update of the target network, defaults to
        0.005.
    :param float gamma: discount factor, in [0, 1], defaults to 0.99.
    :param float exploration_noise: the noise intensity, add to the action,
        defaults to 0.1.
    :param action_range: the action range (minimum, maximum).
    :type action_range: (float, float)
    :param bool reward_normalization: normalize the reward to Normal(0, 1),
        defaults to ``False``.
    :param bool ignore_done: ignore the done flag while training the policy,
        defaults to ``False``.
    :param int estimation_step: greater than 1, the number of steps to look
        ahead.
    .. seealso::
        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(self,
                 actor: torch.nn.Module,
                 actor_optim: torch.optim.Optimizer,
                 critic: torch.nn.Module,
                 critic_optim: torch.optim.Optimizer,
                 tau: float = 0.005,
                 gamma: float = 0.99,
                 exploration_noise: float = 0.1,
                 action_range: Optional[Tuple[float, float]] = None,
                 reward_normalization: bool = False,
                 ignore_done: bool = False,
                 estimation_step: int = 1,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if actor is not None:
            self.actor, self.actor_old = actor, deepcopy(actor)
            self.actor_old.eval()
            self.actor_optim = actor_optim
        if critic is not None:
            self.critic, self.critic_old = critic, deepcopy(critic)
            self.critic_old.eval()
            self.critic_optim = critic_optim
        assert 0 <= tau <= 1, 'tau should in [0, 1]'
        self._tau = tau
        assert 0 <= gamma <= 1, 'gamma should in [0, 1]'
        self._gamma = gamma
        assert 0 <= exploration_noise, 'noise should not be negative'
        self._eps = exploration_noise
        assert action_range is not None
        self._range = action_range
        self._action_bias = (action_range[0] + action_range[1]) / 2
        self._action_scale = (action_range[1] - action_range[0]) / 2
        # it is only a little difference to use rand_normal
        # self.noise = OUNoise()
        self._rm_done = ignore_done
        self._rew_norm = reward_normalization
        assert estimation_step > 0, 'estimation_step should greater than 0'
        self._n_step = estimation_step

    def set_eps(self, eps: float) -> None:
        """Set the eps for exploration."""
        self._eps = eps

    def train(self) -> None:
        """Set the module in training mode, except for the target network."""
        self.training = True
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        """Set the module in evaluation mode, except for the target network."""
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def sync_weight(self) -> None:
        """Soft-update the weight for the target network."""
        for o, n in zip(self.actor_old.parameters(), self.actor.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)
        for o, n in zip(
                self.critic_old.parameters(), self.critic.parameters()):
            o.data.copy_(o.data * (1 - self._tau) + n.data * self._tau)

    def _target_q(self, buffer: ReplayBuffer,
                  indice: np.ndarray) -> torch.Tensor:
        batch = buffer[indice]  # batch.obs_next: s_{t+n}
        with torch.no_grad():
            target_q = self.critic_old(batch.obs_next, self(
                batch, model='actor_old', input='obs_next', eps=0).act)
        return target_q

    def process_fn(self, batch: Batch, buffer: ReplayBuffer,
                   indice: np.ndarray) -> Batch:
        if self._rm_done:
            batch.done = batch.done * 0.
        batch = self.compute_nstep_return(
            batch, buffer, indice, self._target_q,
            self._gamma, self._n_step, self._rew_norm)
        return batch

    def forward(self, batch: Batch,
                state: Optional[Union[dict, Batch, np.ndarray]] = None,
                model: str = 'actor',
                input: str = 'obs',
                eps: Optional[float] = None,
                **kwargs) -> Batch:
        """Compute action over the given batch data.
        :param float eps: in [0, 1], for exploration use.
        :return: A :class:`~tianshou.data.Batch` which has 2 keys:
            * ``act`` the action.
            * ``state`` the hidden state.
        .. seealso::
            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        model = getattr(self, model)
        obs = getattr(batch, input)
        logits, h = model(obs, state=state, info=batch.info)
        logits += self._action_bias
        if eps is None:
            eps = self._eps
        if eps > 0:
            # noise = np.random.normal(0, eps, size=logits.shape)
            # logits += to_torch(noise, device=logits.device)
            # noise = self.noise(logits.shape, eps)
            logits += torch.randn(
                size=logits.shape, device=logits.device) * eps
        logits = logits.clamp(self._range[0], self._range[1])
        return Batch(act=logits, state=h)

    def learn(self, batch: Batch, **kwargs) -> Dict[str, float]:
        current_q = self.critic(batch.obs, batch.act)
        target_q = to_torch_as(batch.returns, current_q)
        target_q = target_q[:, None]
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        actor_loss = -self.critic(batch.obs, self(batch, eps=0).act).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.sync_weight()
        return {
            'loss/actor': actor_loss.item(),
            'loss/critic': critic_loss.item(),
        }