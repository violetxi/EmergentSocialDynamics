import numpy as np
from typing import Dict, Optional
from types import ModuleType
from typeguard import typechecked

import gym
from tensordict.tensordict import TensorDict
from torchrl.envs.common import _EnvWrapper
from torchrl.data.tensor_specs import CompositeSpec

from social_rl.config.base_config import BaseConfig
from social_rl.utils.utils import gym_to_torchrl_spec_transform
from social_rl.environment.social_dilemma.map_env import MapEnv
from social_rl.environment.social_dilemma.env_creator import get_env_creator


@typechecked
class SocialDilemmaEnv(_EnvWrapper):
    """Social Dilemma environment wrapper including CleanUp and Harvest
    args:
        seed: random seed (controlling randomness in the environment, cleanup: pollution and apple respawning, harvest: apple respawning)
        config: config object (e.g. class Config)
        kwargs: kwargs for config object (e.g. device)
    """
    def __init__(
            self,
            seed: int,
            config: BaseConfig,
            base_kwargs: Dict
    ) -> None:
        self.seed = seed
        self.config = config
        super().__init__(**base_kwargs)


    def _init_env(self) -> None:
        """Initialize the environment: only needs to change the seed for environment
        """
        np.random.seed(self.seed)
        

    def _build_env(self) -> MapEnv:
        """Build the environment from config
        """
        env_creator = get_env_creator(
            self.config.task_name, 
            **self.config.env_kwargs
            )
        env = env_creator(None)
        return env
    

    def _check_kwargs(self, kwargs: Dict) -> None:
        pass


    def _set_seed(self, seed: int) -> None:
        """Set the seed for the environment
        """
        np.random.seed(seed)


    def _reset(
            self, 
            tensordict: TensorDict, 
            **kwargs
            ) -> TensorDict:
        obs = self._env.reset()
        obs = {"observation": obs}
        tensordict_out = TensorDict(
            source=obs,
            batch_size=self.batch_size,
            device=self.device
            )
        tensordict_out.setdefault(
            "done",
            self.done_spec.zero(),
        )
        return tensordict_out
    

    def _step(
            self,
            tensordict: TensorDict
    ) -> TensorDict:
        td_actions = tensordict.get("action").to_dict()
        actions = {k: v.numpy() for k, v in td_actions.items()}

        for _ in range(self.wrapper_frame_skip):
            observations, rewards, dones, info = self._env.step(actions)
        
        # we don't stop or truncate the episode until all agents are done or reach max episode length
        done = all(dones.values())
        # set done to True if either done or truncated        
        obs_dict = {
            "observation": observations,
            "reward": rewards,
            "done": done,
        }

        obs_dict = {("next", key): val for key, val in obs_dict.items()}
        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )

        return tensordict_out
    

    def _make_specs(self, env: MapEnv) -> None:
        self.categorical_action_encoding = not env.aec_env.env.env.continuous_actions     
        action_spaces = {agent : env.action_space(agent) for agent in env.possible_agents}    
        action_spec = gym_to_torchrl_spec_transform(
            action_spaces,
            device=self.device,
            categorical_action_encoding=self.categorical_action_encoding,
        )
        self.action_spec = action_spec

        observation_spaces = {
            'observation': {
                agent : env.observation_space(agent) for agent in env.possible_agents
                }
        }        
        observation_spec = gym_to_torchrl_spec_transform(
            observation_spaces,
            device=self.device,
            categorical_action_encoding=self.categorical_action_encoding,
        )
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(
                observation=observation_spec)
        self.observation_spec = observation_spec        
        reward_spaces = {
            agent: 
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32) 
            for agent in env.possible_agents}
        reward_spec = gym_to_torchrl_spec_transform(
            reward_spaces,
            device=self.device,
            categorical_action_encoding=self.categorical_action_encoding,
        )
        # this is important because it would not pass the check in the base class here
        # https://github.com/pytorch/rl/blob/e21e4cf9c5e557957b5a14cf611a81f15e12dc2c/torchrl/envs/common.py#L614
        reward_spec = reward_spec.unsqueeze(0) # add batch dim
        self.reward_spec = reward_spec 


    def render(self):
        """Return rendered rgb_array (c, w, h)"""
        return self._env.render(mode='rgb_array')
