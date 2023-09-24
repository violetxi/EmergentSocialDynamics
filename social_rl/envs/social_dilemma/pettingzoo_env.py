import time
import functools
from typing import Any, Dict
from collections import deque
from copy import deepcopy

import numpy as np
import gymnasium as gym
from pettingzoo import ParallelEnv

from social_rl.envs.social_dilemma.env_creator import get_env_creator


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["rgb_array"], "name": "social_dilemmas"}
    """PettingZoo ParallelEnv Wrapper for social dilemma environments
    :param base_env_kwargs: keyword arguments to pass to the base ssd env
    :param max_cycles: maximum number of steps per episode
    :param render_mode: render mode for the environment
    :param stack_num: number of frames to stack for observation and action
    """
    def __init__(
            self, 
            base_env_kwargs: Dict[str, Any],
            max_cycles: int =5000,            
            render_mode: str =None,
            stack_num: int = 1,
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
        self.stack_num = stack_num
        if self.stack_num > 1:            
            self.observation_history = deque(maxlen=self.stack_num - 1)
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
        if seed is not None:
            self.seed = seed
        self._env.seed(self.seed)
        obs = self._env.reset()
        obs_out = {}
        info = {}
        for agent_id, ob in obs.items():
            if self.stack_num > 1:
                obs_out[agent_id] = {
                    'observation': {},
                    'action_mask': np.ones(self._env.action_space.n, "int8")
                }                
            else:
                obs_out[agent_id] = {
                    'observation': ob, 
                    'action_mask': np.ones(self._env.action_space.n, "int8")
                    }
            info[agent_id] = {}
        if self.stack_num > 1:
            self.stack_obs(obs, obs_out)        
        return obs_out, info

    def stack_obs(self, obs, obs_out):
        # if not enough history to stack, repeat "wait" action until 
        # we have enough history
        if self.steps < self.stack_num:
            self.observation_history.extend([
                obs for _ in range(self.stack_num - 1)]
                )
            self.steps = self.stack_num
        # stack current observation with past history
        obs_history_list = list(self.observation_history)
        obs_history_list.append(obs)
        agent_obs_dict = {
            agent_id : {
                'curr_obs': [],
                'other_agent_actions': [],
                'self_actions': [],
                'visible_agents': [],
                'prev_visible_agents': []
                } for agent_id in self.agents
            }
        # reformat history into dict of lists
        for obs_hist in obs_history_list:
            for agent_id in self.agents:
                obs_hist_agent = obs_hist[agent_id]
                for k, v in obs_hist_agent.items():
                    agent_obs_dict[agent_id][k].append(v)
        # add current observation to history
        for agent_id, obs_agent in obs.items():
            for k, _ in obs_agent.items():
                obs_out[agent_id]['observation'][k] = np.stack(
                    np.stack(agent_obs_dict[agent_id][k])
                    )
        # add current observation to history
        self.observation_history.append(obs)

    def step(self, actions):
        """        
        :param actions: a dict of actions for each agent

        :return: observations, rewards, dones, infos in np.ndarray formation and agent are 
        index in the order of self.possible_agents        
            - observations: list of observations in dict format for each agent
            - rewards
            - terminations: SSD doesn't terminate, so terminate when max_cycles is reached
            - info
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """                        
        self.steps += 1
        actions = dict(zip(self.possible_agents, actions))
        obs, rewards, terminations, info = self._env.step(actions)                        
        obs_out = {}
        rewards_out = []
        terminations_out = []
        truncations_out = []        
        # done
        if self.steps == self.max_cycles:
            terminations = {agent: True for agent in self.agents}
        # __all__ is a special key for PettingZoo envs that indicates all agents are done        
        if '__all__' in terminations:
            del terminations['__all__']
        # stack observations and actions if asked for
        for agent_id in self.possible_agents:
            if self.stack_num > 1:
                obs_out[agent_id] = {
                    'observation': {},
                    'action_mask': np.ones(self._env.action_space.n, "int8")
                }                
            else:
                obs_out[agent_id] = {
                    'observation': obs[agent_id],
                    'action_mask': np.ones(self._env.action_space.n, "int8")
                }            
            rewards_out.append(rewards[agent_id])
            # terminations and truncations are the same for all agents, keep one per
            # env to be compatible with TianShou Buffer processes
            terminations_out.append(terminations[agent_id]) 
            truncations_out.append(False) # no truncation in SSD
            info[agent_id] = {}                
        # stack observations and actions if asked for                   
        if self.stack_num > 1:
            self.stack_obs(obs, obs_out)            

        return obs_out, rewards_out, np.all(terminations_out), \
            np.all(truncations_out), info

    def get_agent_colors(self) -> Dict[str, np.ndarray]:
        # get agent colors from the environment, used for plotting
        agent_colors = {
        agent_id: self._env.color_map[str(
            int(agent_id.split('_')[1]) + 1
            ).encode()] / 255.0
        for agent_id in self.possible_agents
        }
        return agent_colors


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