from typeguard import typechecked
from typing import List

import torch.nn as nn
from tensordict.nn import (
    TensorDictModule,
    TensorDictSequential
)

from social_rl.config.base_config import BaseConfig



@typechecked
class TensorDictPolicyNet(TensorDictModule):
    """TensorDictModule wrapper for policy network. This assume there is a 
    single nn.Module for the network
    args:
        config: config object coming from config.agent_config.policy_config        
    """
    def __init__(
            self,
            config: BaseConfig
        ) -> None:
        assert config.policy.policy_cls_type == "dict", \
            "policy_cls_type must be dict for TensorDictPolicy"
        assert isinstance(config.policy.net_module, nn.Module), \
            "net_module must be a nn.Module for TensorDictPolicy"
        net_module = config.policy.net_module(**config.policy.net_kwargs)
        super().__init__(
            module=net_module,
            in_keys=config.policy.in_keys,
            out_keys=config.policy.out_keys
        )


    def forward(self, x: dict) -> dict:
        tensordict_out = super().forward(x)
        return tensordict_out


  
@typechecked
class TensorDictSequentialPolicyNet(TensorDictSequential):
    """TensorDictSequential wrapper for policy network. This assume there is a 
    list of nn.Module modules for the network
    """
    def __init__(
            self,
            config: BaseConfig
        ) -> None:
        assert config.policy.policy_cls_type == "sequential", \
            "policy_cls_type must be sequential for TensorDictSequentialPolicy"
        assert isinstance(config.policy.net_modules_kwargs, list), \
            "net_modules_kwargs must be a list for nn.Module for TensorDictSequentialPolicy"
        net_modules = []
        for net_module_kwargs in config.policy.net_modules_kwargs:
            net_module = config.policy.net_module(**net_module_kwargs)
            net_modules.append(net_module)
        super().__init__(
            modules=net_modules,
            in_keys=config.policy.in_keys,
            out_keys=config.policy.out_keys
        )


    def forward(self, x: dict) -> dict:
        tensordict_out = super().forward(x)
        return tensordict_out