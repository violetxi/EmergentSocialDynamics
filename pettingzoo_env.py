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
            base_env_kwargs: Dict[str, Any],
            max_cycles: int =5000,            
            render_mode: str =None,
            collect_frames: bool =False,
            ) -> None:
        """
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - render_mode

        These attributes should not be changed after initialization.
        """        
        self._env = get_env_creator(**base_env_kwargs)
        self.possible_agents = list(self._env.agents.keys())
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.max_cycles = max_cycles  
        self.render_mode = render_mode
        self.collect_frames = collect_frames
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
    
    def collect_frame(self, infos):
        # this is done because PettingZoo ACEnv expects non-empty infos 
        # for each agent, so we add the frames to the last agent's info
        for i, agent in enumerate(self.agents):
            frame = self.render()
            if i == len(self.agents) - 1:
                infos[agent]['frame'] = frame
            else:
                infos[agent]['frame'] = []
        return infos

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played. It also resets the `steps` variable which counts
        the number of steps in the current episode.
        Returns the observations for each agent
        """
        self.steps = 0        
        self.agents = self.possible_agents[:]        
        obs = self._env.reset()
        observations = {}
        for agent_id, ob in obs.items():
            observations[agent_id] = {
                'observation': ob, 
                'action_mask': np.ones(self._env.action_space.n, "int8")
                }
        # infos = {agent: {} for agent in self.agents}
        # if self.collect_frames:
        #     self.collect_frame(infos)

        return observations

    def step(self, actions):
        #@TODO: it's better to return things as np.array and use index to keep track of 
        # agents
        """        
        :param actions: a dict of actions for each agent

        :return: observations, rewards, dones, infos        
        - observations
        - rewards
        - terminations: SSD doesn't terminate, so terminate when max_cycles is reached
        - info
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        self.steps += 1
        actions = dict(zip(self.possible_agents, actions))
        obs, rewards, terminations, infos = self._env.step(actions)                
        # observations        
        observations = {}
        for agent_id, ob in obs.items():
            observations[agent_id] = {
                'observation': ob, 
                #'action_mask': np.ones(self._env.action_space.n, "int8")
                }
        breakpoint()
        # reward
        #rewards = np.array(list(rewards.values()))        
        # done
        if self.steps == self.max_cycles:
            terminations = {agent: True for agent in self.agents}
        # __all__ is a special key for PettingZoo envs that indicates all agents are done        
        if '__all__' in terminations:
            del terminations['__all__']
        #terminations = np.array(list(terminations.values()))
        
        if self.collect_frames:
            if self.collect_frames:
                self.collect_frame(infos)
        
        return observations, rewards, terminations, infos
    


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