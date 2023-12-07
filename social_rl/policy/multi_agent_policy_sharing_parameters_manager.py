from typing import Any, Dict, List, Optional, Union
from copy import deepcopy

import gym
import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import BasePolicy


class MultiAgentPolicySharingParametersManager(BasePolicy):
    """Multi-agent policy manager for **decentralized** MARL.
    *** Adapted from https://github.com/thu-ml/tianshou/blob/master/tianshou/policy/multiagent/mapolicy.py ***

    This multi-agent policy manager accepts a list of
    :class:`~tianshou.policy.BasePolicy`. It dispatches the batch data to each
    of these policies when the "forward" is called. The same as "process_fn"
    and "learn": it splits the data and feeds them to each policy. A figure in
    :ref:`marl_example` can help you better understand this procedure.
    """

    def __init__(
        self, 
        policies: List[BasePolicy], 
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


    def reset_hidden_state(self) -> None:
        """Reset the hidden state of each policy."""
        policy = self.policies[self.env_agents[0]]
        policy.critic.preprocess.state = None
        policy.actor.preprocess.state = None


    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        """Dispatch batch data from obs.agent_id to every policy's process_fn.
        Ensure every thing starts with (bs, n_agents, ...) shape.
        """
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)        
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        
        tmp_batch_dict = {}
        results = {}        
        for k, v in batch.items():
            if isinstance(v, Batch) and v.is_empty():
                # for keys with no data populated, we use empty Batch()
                tmp_batch_dict[k] = v
            else:
                if isinstance(v, Batch):
                    # stack all the observations
                    v_list = [v.get(agent_id) for agent_id in self.env_agents]                    
                    tmp_batch_dict[k] = Batch.stack(v_list, axis=1)
                else:
                    tmp_batch_dict[k] = v
        
        tmp_batch = Batch(tmp_batch_dict)
        # all indicies should be for one agennt (bs)
        tmp_indice = np.arange(tmp_batch.obs.shape[0])
        results = self.policies[self.env_agents[0]].process_fn(tmp_batch, buffer, tmp_indice)        
        
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        # return Batch(results)
        return results


    def exploration_noise(
            self, 
            act: Union[np.ndarray, Batch],
            batch: Batch
            ) -> Union[np.ndarray, Batch]:
        """Add exploration noise from sub-policy onto act."""        
        agent_ids = self.env_agents        
        for agent_id, policy in self.policies.items():
            agent_idx = agent_ids.index(agent_id)
            act[:, agent_idx] = policy.exploration_noise(
                act[:, agent_idx], batch
            )
        return act


    def forward(  # type: ignore
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Organize batch data for decentralized execution using a shared policy.
        """      
        results = {}
        tmp_batch_dict = {}
        agent_ids = self.env_agents        
        for k, v in batch.items():
            if isinstance(v, Batch) and v.is_empty():
                # for keys with no data populated, we use empty Batch()
                tmp_batch_dict[k] = v
            else:
                if isinstance(v, Batch):                    
                    v_list = [v.get(agent_id) for agent_id in agent_ids]
                    #tmp_batch_dict[k] = Batch.stack(v_list)               
                    tmp_batch_dict[k] = Batch.stack(v_list, axis=1)
        tmp_batch = Batch(tmp_batch_dict)
        policy = self.policies[agent_ids[0]]
        out = policy(
            batch=tmp_batch,
            state=None, # we keep track of state in actor network
            **kwargs
        )
        
        n_agents, _, bs, _ = out.state.shape        
        out.act = out.act.reshape(n_agents, bs)        
        for agent_id in agent_ids:
            agent_idx = agent_ids.index(agent_id)
            results[agent_id] = {
                'out': out.logits[agent_idx, :],
                'act': out.act[agent_idx, :],
                'state': out.state[agent_idx, :]
            }        
        holder = Batch(results)
        return holder


    def learn(self, batch: Batch,
              **kwargs: Any) -> Dict[str, Union[float, List[float]]]:
        """Aggregate all agents trajectories and learn from them..
        """
        results = {}
        # reset hidden states for each state
        self.reset_hidden_state()
        # create a batch with all agents' trajectories to train a single policy
        policy = self.policies[self.env_agents[0]]        
        
        if not batch.is_empty():
            results = policy.learn(batch, **kwargs)                    
        return results