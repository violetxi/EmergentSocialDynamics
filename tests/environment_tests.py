"""Torch RL default check_env_specs makes a few assumption about the reward spec that
is not always true for MARL scenario. Eg, assuming reward is a non-composite spec, whereas
in MARL, reward is a composite spec of {agent: reward_spec, ...}.
"""
from typeguard import typechecked
from types import ModuleType
from typing import Any, Dict, Optional, Callable

from torchrl.envs.utils import check_env_specs, step_mdp
from torchrl.envs.common import _EnvWrapper

import social_rl.environment.petting_zoo_base as petting_zoo_base
from social_rl.config.social_dilemma_configs.cleanup_config import Config


@typechecked
def test_env_rollout(env: _EnvWrapper):
    """Test rollout in MPE env"""
    rollout_length = 10
    print(f">>> Testing rollout in {env.config.env_name} env for {rollout_length} steps")
    rollouts = env.rollout(
        rollout_length,
        auto_reset=True)
    for i, rollout in enumerate(rollouts):
        print(f"Testing rollout {i}")
        if i < rollout_length - 1:
            obs_next = rollout['next']['observation']
            obs_plus_one = rollouts[i+1]['observation']
            for k, v in obs_next.items():
                obs_check_passed = (obs_next == obs_plus_one).all()
                assert obs_check_passed, \
                    f"Observation {k} at rollout {i} and {i+1} are not the same"
    print("Rollout passed!")
        

if __name__ == '__main__':
    seed = 0
    env = petting_zoo_base.PettingZooMPEBase(seed, Config)
    test_env_rollout(env)