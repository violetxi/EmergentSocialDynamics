import functools
from typing import Any, Dict

import numpy as np

#import gym
import gymnasium as gym
from pettingzoo import ParallelEnv

from envs.env_creator import get_env_creator



class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "social_dilemmas"}

    def __init__(
            self, 
            env_kwargs: Dict[str, Any],
            max_cycles: int =5000,            
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
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.steps = 0

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
        # @TODO: gymnaisum is assumed for PPO actor class in TianShou, so we need to 
        # change action_space to gymnasium's action space only for this purpose
        # maybe consider redoing the environment in gymnasium if it's worthwile
        # but currently this hack works...
        n_actions = self._env.action_space.n
        return gym.spaces.Discrete(n_actions)        

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
        - terminations: SSD doesn't terminate, so terminate when max_cycles is reached
        - truncations
        - info
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        self.steps += 1
        actions
        obs, rewards, terminations, infos  = self._env.step(actions)
        if self.steps > self.max_cycles:
            terminations = {agent: True for agent in self.agents}
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