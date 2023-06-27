import numpy as np
from typing import Dict, Optional, Callable, Any
from types import ModuleType
from typeguard import typechecked

import gymnasium as gym
import pettingzoo.mpe as mpe
from pettingzoo.utils.env import ParallelEnv
from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from torchrl.envs.common import _EnvWrapper
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec

from social_rl.utils.utils import gym_to_torchrl_spec_transform


@typechecked
class PettingZooMPEBase(_EnvWrapper):
    def __init__(self,seed: int, config: Callable, **kwargs) -> None:
        """
        Args:
            seed: random seed
            config: config object (need to be a callable, e.g. class Config)
        """        
        self.seed = seed
        self.config = config().env_config
        self.kwargs = kwargs
        super().__init__(**kwargs)


    def _get_mpe_env_cls(self, env_name: str) -> ModuleType:
        """Get the MPE env class from pettingzoo.mpe
        Currently only support non-communicative environments
        """
        if env_name == 'simple_v3':
            cls = mpe.simple_v3                
        elif env_name == 'simple_tag_v3':
            cls = mpe.simple_tag_v3
        elif env_name == 'simple_spread_v3':
            cls = mpe.simple_spread_v3
        elif env_name == 'simple_adversary_v3':
            cls = mpe.simple_adversary_v3
        elif env_name == 'simple_push_v3':
            cls = mpe.simple_push_v3
        else:
            raise NotImplementedError(f"Env {env_name} not implemented. Currently only support non-communicative environment")
        return cls


    def _init_env(self) -> None:
        # initialize env so it's ready to run
        self._env.reset(self.seed)


    def _build_env(self) -> ParallelEnv:
        env_cls = self._get_mpe_env_cls(self.config.task_name)
        env = env_cls.parallel_env(**self.config.env_kwargs)
        return env


    def _check_kwargs(self, kwargs: Dict) -> None:
        pass


    def _set_seed(self, seed: int) -> None:
        # super().set_seed() set torch.manual_seed() already
        # PettingZoo doesn't not allow access to internal _seed() method in base class
        # therefore seeding is done in reset() method for higher level env APIs
        self._env.reset(seed)    


    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        obs, infos = self._env.reset(**kwargs)        
        obs = {"observation": obs}
        tensordict_out = TensorDict(
            source=obs,    # {agent: obs, ...}
            batch_size=self.batch_size,
            device=self.device,
        )
        tensordict_out.setdefault(
            "done",
            self.done_spec.zero(),
        )
        return tensordict_out
    

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # for MARL env, tensordict['action'] is a dict of {agent: action, ...}
        # PettingZoo parallel_env then takes on step() with with all agents' actions at once
        td_actions = tensordict.get("action").to_dict()
        actions = {k: v.numpy() for k, v in td_actions.items()}

        for _ in range(self.wrapper_frame_skip):            
            observations, rewards, dones, truncations, infos = self._env.step(actions)

        # we don't stop or truncate the episode until all agents are done or reach max episode length
        done = all(dones.values())
        truncated = all(truncations.values())
        # set done to True if either done or truncated
        done = done | truncated
        obs_dict = {
            "observation": observations,
            "reward": rewards,
            "done": done,
            #"truncated": truncated
        }
        obs_dict = {("next", key): val for key, val in obs_dict.items()}
        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )

        return tensordict_out
    

    # @TODO: implement in MPEEnv class
    def _make_specs(self, env: ParallelEnv) -> None:
        self.categorical_action_encoding = not env.aec_env.env.env.continuous_actions     
        action_spaces = {agent : env.action_space(agent) for agent in env.possible_agents}
        self.action_spec = gym_to_torchrl_spec_transform(
            action_spaces,
            device=self.device,
            categorical_action_encoding=self.categorical_action_encoding,
        )
        observation_spaces = {
            'observation': {agent : env.observation_space(agent) for agent in env.possible_agents}
        }        
        observation_spec = gym_to_torchrl_spec_transform(
            observation_spaces,
            device=self.device,
            categorical_action_encoding=self.categorical_action_encoding,
        )        
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(observation=observation_spec)
        self.observation_spec = observation_spec
        
        # MA envs reward spaces will be a dict of {agent: reward_space, ...}
        # reward_spaces = {agent: UnboundedContinuousTensorSpec(
        #     shape=[1],
        #     device=self.device,
        #     ) for agent in env.possible_agents}
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

