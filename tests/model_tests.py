"""Torch RL default check_env_specs makes a few assumption about the reward spec that
is not always true for MARL scenario. Eg, assuming reward is a non-composite spec, whereas
in MARL, reward is a composite spec of {agent: reward_spec, ...}.
"""
from typeguard import typechecked
from types import ModuleType
from typing import Any, Dict, Optional, Callable

from torch import nn
from torchrl.modules.tensordict_module.common import is_tensordict_compatible

from social_rl.config.sim_tag_default import Config
from social_rl.models.world_models.mlp_dynamics_model import MLPDynamicsModel



@typechecked
def test_tensordict_compatible(model: nn.Module) -> None:
    print(f"MLPDynamicsModel is tensordict compatible: {is_tensordict_compatible(model)}")



if __name__ == '__main__':
    agent_idx = 0    
    model = MLPDynamicsModel(agent_idx, Config)
    test_tensordict_compatible(model)