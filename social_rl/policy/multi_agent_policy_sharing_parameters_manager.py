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
        :param batch: a Batch of data with the following keys, keys without data
            will has an empty Batch() as value:
            - 'obs': {agent_id: nested_obs_dict},
            - 'obs_next': {agent_id: nested_obs_dict},
            - 'rew': (batch_size, num_agents),
            - 'act': (batch_size, num_agents),
            - 'done': (batch_size, 1), one for each episode (true when all agents are done)
            - 'terminated': (batch_size, 1), one for each episode (true when any agent is done)
            - 'truncated': (batch_size, 1), one for each episode (true when any agent is done)            
            - 'policy': @TODO this is not used yet
            - 'info': {agent_id: {}}, @TODO if info is not empty, it should be a nested dict 
        :param buffer: a ReplayBuffer to store data.
        :param indice: the indices of samples which are selected from buffer. **Note: because our 
            buffer is a multi-agent buffer which stores item in the same index as nested data structure 
            like np.ndarray or dictionary, which are used agent_id or agent_idx to identify. TianShou's 
            indice is expecting global index in the Batch object, therefore in our case they just use all 
            the indices in a batch.**
        """        
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)        
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        
        output = Batch()
        results = {}
        agent_ids = self.env_agents
        for agent_id, policy in self.policies.items():
            output[agent_id] = Batch()         
            tmp_batch_dict = {}
            agent_idx = agent_ids.index(agent_id)        
            for k, v in batch.items():
                if v is None:
                    tmp_batch_dict[k] = Batch()                    
                else:
                    if k in ['obs', 'obs_next']:
                        tmp_batch_dict[k] = v.get(agent_id)
                    else:          
                        if k in ['done', 'terminated', 'truncated', 'info', 'policy']:
                            tmp_batch_dict[k] = v
                        else:                 
                            tmp_batch_dict[k] = v[:, agent_idx]

            # @TODO currently we don't support hidden state, will update later
            #state = None
            tmp_batch = Batch(tmp_batch_dict)
            # all indicies should be for current agennt
            tmp_indice = np.arange(len(tmp_batch.obs))            
            results[agent_id] = policy.process_fn(tmp_batch, buffer, tmp_indice)
        
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)


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
                    # stack all the observations
                    v_list = [v.get(agent_id) for agent_id in agent_ids]
                    tmp_batch_dict[k] = Batch.stack(v_list)                
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
        shared_batch_list = [agent_batch for agent_id, agent_batch in batch.items()]        
        shared_batch = Batch.cat(shared_batch_list)
        if not shared_batch.is_empty():
            results = policy.learn(shared_batch, **kwargs)                    
        return results