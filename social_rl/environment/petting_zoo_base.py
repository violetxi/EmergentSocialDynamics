import numpy as np
from typing import Dict, Optional, Any, Callable
from typeguard import typechecked

from tensordict import TensorDict
from tensordict.tensordict import TensorDictBase
from pettingzoo.mpe import simple_tag_v3
from torchrl.envs.common import _EnvWrapper
from torchrl.data.tensor_specs import (
    CompositeSpec,
    UnboundedContinuousTensorSpec,
)
from utils import gym_to_torchrl_spec_transform


@typechecked
class PettingZooMPEBase(_EnvWrapper):
    def __init__(self, seed: int, env: Callable, **kwargs) -> None:
        self.seed = seed
        self._env = env
        self.kwargs = kwargs


    def _check_kwargs(self, kwargs: Dict):
        pass


    def _set_seed(self, seed: int) -> None:
        # super().set_seed() set torch.manual_seed() already
        # PettingZoo doesn't not allow access to internal _seed() method in base class
        # therefore seeding is done in reset() method for higher level env APIs
        self.env.reset(seed)    


    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDictBase:
        obs, infos = self.env.reset(**kwargs)
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
        actions = tensordict.get("action").to_dict()        

        for _ in range(self.wrapper_frame_skip):
            observations, rewards, dones, truncations, infos = env.step(actions)
        obs_dict = {
            "observation": observations,
            "reward": rewards,
            "done": dones,
            "truncation": truncations,
            "info": infos
        }        
        tensordict_out = TensorDict(
            obs_dict, batch_size=tensordict.batch_size, device=self.device
        )
        return tensordict_out
    

    # @TODO: implement in MPEEnv class
    def _init_env(self):
        self._env.reset(self.seed)

    
    # @TODO: implement in MPEEnv class
    def _build_env(self, **kwargs) -> Any:
        pass    
    

    # @TODO: implement in MPEEnv class
    def _make_specs(self, env: Callable) -> None:
        self.action_spec = gym_to_torchrl_spec_transform(
            env.action_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        observation_spec = gym_to_torchrl_spec_transform(
            env.observation_space,
            device=self.device,
            categorical_action_encoding=self._categorical_action_encoding,
        )
        if not isinstance(observation_spec, CompositeSpec):
            observation_spec = CompositeSpec(observation=observation_spec)
        self.observation_spec = observation_spec
        if hasattr(env, "reward_space") and env.reward_space is not None:
            self.reward_spec = gym_to_torchrl_spec_transform(
                env.reward_space,
                device=self.device,
                categorical_action_encoding=self._categorical_action_encoding,
            )
        else:
            self.reward_spec = UnboundedContinuousTensorSpec(
                shape=[1],
                device=self.device,
            )
        



if __name__ == '__main__':
    kwargs = dict(
        num_good=2, 
        num_adversaries=2, 
        num_obstacles=2)
    env = simple_tag_v3.parallel_env(**kwargs)
    obs, infos = env.reset()
    step = 20
    for s in range(step):        
        actions = {}
        for agent in env.agents:
            actions[agent] = env.action_spaces[agent].sample()
        breakpoint()      
        observations, rewards, dones, truncations, infos = env.step(actions)
