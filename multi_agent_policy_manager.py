from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import BasePolicy

try:
    from tianshou.env.pettingzoo_env import PettingZooEnv
except ImportError:
    PettingZooEnv = None  # type: ignore


class MultiAgentPolicyManager(BasePolicy):
    """Multi-agent policy manager for MARL.

    This multi-agent policy manager accepts a list of
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to each
    of these policies when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to each policy. A figure in
    :ref:`marl_example` can help you better understand this procedure.
    """

    def __init__(
        self, policies: List[BasePolicy], 
        env_agents: List[str],
        action_space: gym.spaces.Space,         
        **kwargs: Any
    ) -> None:
        super().__init__(action_space=action_space, **kwargs)
        assert (
             len(policies) == len(env_agents)
        ), "One policy must be assigned for each agent."

        self.env_agents = env_agents        
        self.agent_idx = {agent: i for i, agent in enumerate(self.env_agents)}
        for i, policy in enumerate(policies):
            # agent_id 0 is reserved for the environment proxy
            # (this MultiAgentPolicyManager)
            policy.set_agent_id(self.env_agents)

        self.policies = dict(zip(self.env_agents, policies))

    def replace_policy(self, policy: BasePolicy, agent_id: int) -> None:
        """Replace the "agent_id"th policy in this manager."""
        policy.set_agent_id(agent_id)
        self.policies[agent_id] = policy

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)        
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        
        for agent, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)

    def exploration_noise(self, act: Union[np.ndarray, Batch],
                          batch: Batch) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""
        for agent_id, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent_id)[0]
            if len(agent_index) == 0:
                continue
            act[agent_index] = policy.exploration_noise(
                act[agent_index], batch[agent_index]
            )
        return act

    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's forward.

        :param batch: a Batch of data with the following keys:
            'obs' - , 
            'obs_next', 
            'rew', 
            'done', 
            'act', 
            'policy', 
            'info',
        :param state: if None, it means all agents have no state. If not
            None, it should contain keys of "agent_1", "agent_2", ...

        :return: a Batch with the following contents:

        ::

            {
                "act": (batch_size, )
                }
                "state": {
                    "agent_1": output state of agent_1's policy for the state
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
                "out": {
                    "agent_1": output of agent_1's policy for the input
                    "agent_2": xxx
                    ...
                    "agent_n": xxx}
            }
        """
        # results: List[Tuple[bool, np.ndarray, Batch, Union[np.ndarray, Batch],
        #                     Batch]] = []        
        results = {}
        agent_ids = self.env_agents
        for agent_id, policy in self.policies.items():
            tmp_batch_dict = {}
            agent_idx = agent_ids.index(agent_id)            
            for k, v in batch.items():
                if v is None:
                    tmp_batch_dict[k] = None
                else:
                    if k in ['obs', 'obs_next']:
                        tmp_batch_dict[k] = v.get(agent_id)
                    else:                                        
                        tmp_batch_dict[k] = v[:, agent_idx]

            tmp_batch = Batch(tmp_batch_dict)                        
            out = policy(
                batch=tmp_batch,
                state=None if state is None else state[agent_id],
                **kwargs
            )            
            act = out.act
            each_state = out.state \
                if (hasattr(out, "state") and out.state is not None) \
                else Batch()
            results[agent_id] = {
                'out': out,  # shape (B, ...) or (N_env, ...)
                'act': act,  # shape (B, ) or (N_env, )
                'state': each_state    # shape (B, ...) or (N_env, ...)
            }
        holder = Batch(results) 
        breakpoint()  
        return holder

    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
        """Dispatch the data to all policies for learning.

        :return: a dict with the following contents:

        ::

            {
                "agent_1/item1": item 1 of agent_1's policy.learn output
                "agent_1/item2": item 2 of agent_1's policy.learn output
                "agent_2/xxx": xxx
                ...
                "agent_n/xxx": xxx
            }
        """
        results = {}
        for agent_id, policy in self.policies.items():
            breakpoint()
            data = batch[agent_id]
            if not data.is_empty():
                out = policy.learn(batch=data, **kwargs)
                for k, v in out.items():
                    results[agent_id + "/" + k] = v
        return results
