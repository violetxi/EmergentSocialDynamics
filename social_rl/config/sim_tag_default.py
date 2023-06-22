""" Default configuration for simple_tag_v3 environment.
"""
from typeguard import typechecked
from social_rl.env.PettingZoo import PettingZooBase

@typechecked
class EnvConfig:
    def __init__(self) -> None:
        self.env_name = "mpe"
        self.task_name = "simple_tag_v3"
        self.env_class = PettingZooBase
        self.env_kwargs = dict(
            num_good=4, 
            num_adversaries=4,
            num_obstacles=3, 
            max_cycles=25, 
            continuous_actions=False,
            #render_mode='rgb_array'
        )

class WmConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class PolicyConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class ActorConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class ExpConfig:
    def __init__(self) -> None:
        pass  # fill in with your parameters


class Config:
    def __init__(self) -> None:
        self.env_config = EnvConfig()
        self.wm_config = WmConfig()
        self.policy_config = PolicyConfig()
        self.actor_config = ActorConfig()
        self.exp_config = ExpConfig()