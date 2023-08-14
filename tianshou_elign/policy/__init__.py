from tianshou_elign.policy.base import BasePolicy
from tianshou_elign.policy.modelfree.ddpg import DDPGPolicy
from tianshou_elign.policy.modelfree.sac import SACPolicy

from tianshou_elign.policy.modelfree.sacd_multi import SACDMultiPolicy
from tianshou_elign.policy.modelfree.sacd_multi_cc import SACDMultiCCPolicy
from tianshou_elign.policy.modelfree.sacd_multi_wm import SACDMultiWMPolicy
from tianshou_elign.policy.modelfree.sacd_multi_cc_wm import SACDMultiCCWMPolicy

# different IM
from tianshou_elign.policy.modelfree.sacd_multi_wm_new_im import SACDMultiWMPolicyNewIM


__all__ = [
    'BasePolicy',
    'DDPGPolicy',
    'SACPolicy',
    'SACDMultiPolicy',
    'SACDMultiCCPolicy',
    'SACDMultiWMPolicy',
    'SACDMultiCCWMPolicy',
    'SACDMultiWMPolicyNewIM'
]
