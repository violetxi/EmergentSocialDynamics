import functools
from typing import Any, Dict, List, Tuple, Union

import numpy as np

import gym
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector, wrappers

from envs.cleanup import CleanupEnv
from envs.harvest import HarvestEnv
from envs.env_creator import get_env_creator



class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "social_dilemmas"}

    def __init__(
            self, 
            env_kwargs: Dict[str, Any],
            render_mode: str =None,            
            ) -> None:
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """        
        self._env = get_env_creator(**env_kwargs)
        self.possible_agents = list(self._env.agents.keys())
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        obs_space = self._env.observation_space
        observation_space = {}
        observation_space['observation'] = obs_space
        observation_space["action_mask"] = gym.spaces.Box(
            low=0, high=1, shape=(9,), dtype=np.int8)   
        return observation_space     
        
    
     # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._env.action_space

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        return self._env.render(mode=self.render_mode)    

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]        
        obs = self._env.reset()
        observations = {}
        for agent_id, ob in obs.items():
            observations[agent_id] = {
                'observation': ob, 
                'action_mask': np.ones(self._env.action_space.n, "int8")
                }
    
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - info
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        obs, rewards, terminations, infos  = self._env.step(actions)          
        observations = {}
        for agent_id, ob in obs.items():
            observations[agent_id] = {
                'observation': ob, 
                'action_mask': np.ones(self._env.action_space.n, "int8")
                }
        truncations = {agent: False for agent in self.agents}
        return observations, rewards, terminations, truncations, infos



if __name__ == '__main__':
    env_kwargs = dict(
        env="harvest",
        num_agents=2,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
    )
    raw_env = parallel_env(env_kwargs, render_mode="rgb_array")





# from functools import lru_cache

# from gym.utils import EzPickle
# from pettingzoo.utils import wrappers
# from pettingzoo.utils.conversions import parallel_to_aec
# from pettingzoo.utils.env import ParallelEnv

# from envs.env_creator import get_env_creator



# MAX_CYCLES = 1000

# def parallel_env(max_cycles=MAX_CYCLES, **ssd_args):
#     return _parallel_env(max_cycles, **ssd_args)


# def raw_env(max_cycles=MAX_CYCLES, **ssd_args):
#     return parallel_to_aec(parallel_env(max_cycles, **ssd_args))


# def env(max_cycles=MAX_CYCLES, **ssd_args):
#     aec_env = raw_env(max_cycles, **ssd_args)
#     aec_env = wrappers.AssertOutOfBoundsWrapper(aec_env)
#     aec_env = wrappers.OrderEnforcingWrapper(aec_env)
#     return aec_env


# class ssd_parallel_env(ParallelEnv):
#     def __init__(self, env, max_cycles):
#         self.ssd_env = env
#         self.max_cycles = max_cycles
#         self.possible_agents = list(self.ssd_env.agents.keys())
#         self.ssd_env.reset()
#         self.observation_space = lru_cache(maxsize=None)(lambda agent_id: env.observation_space)
#         self.observation_spaces = {agent: env.observation_space for agent in self.possible_agents}
#         self.action_space = lru_cache(maxsize=None)(lambda agent_id: env.action_space)
#         self.action_spaces = {agent: env.action_space for agent in self.possible_agents}

#     def reset(self):
#         self.agents = self.possible_agents[:]
#         self.num_cycles = 0
#         self.dones = {agent: False for agent in self.agents}
#         return self.ssd_env.reset()

#     def seed(self, seed=None):
#         return self.ssd_env.seed(seed)

#     def render(self, mode="human"):
#         return self.ssd_env.render(mode=mode)

#     def close(self):
#         self.ssd_env.close()

#     def step(self, actions):
#         obss, rews, self.dones, infos = self.ssd_env.step(actions)
#         del self.dones["__all__"]
#         self.num_cycles += 1
#         if self.num_cycles >= self.max_cycles:
#             self.dones = {agent: True for agent in self.agents}
#         self.agents = [agent for agent in self.agents if not self.dones[agent]]
#         return obss, rews, self.dones, infos


# class _parallel_env(ssd_parallel_env, EzPickle):
#     metadata = {"render.modes": ["human", "rgb_array"]}

#     def __init__(self, max_cycles, **ssd_args):
#         EzPickle.__init__(self, max_cycles, **ssd_args)
#         env = get_env_creator(**ssd_args)(ssd_args["num_agents"])
#         super().__init__(env, max_cycles)


