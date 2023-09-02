import gym
import time
import torch
import warnings
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Union, Optional, Callable

from tianshou.data import (
    Batch,
    CachedReplayBuffer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReplayBufferManager,
    VectorReplayBuffer,
    to_numpy,
)
from tianshou.data.batch import _alloc_by_keys_diff
from tianshou.policy import BasePolicy
# customized tianshou
from social_rl.tianshou_elign.utils import MovAvg, VecMovAvg, VecTotal
from social_rl.tianshou_elign.env import BaseVectorEnv
from social_rl.tianshou_elign.env import BaseRewardLogger


import psutil
class Collector(object):
    """Collector enables the policy to interact with different types of envs with \
    exact number of steps or episodes.

    :param policy: an instance of the :class:`~tianshou.policy.BasePolicy` class.
    :param env: a ``gym.Env`` environment or an instance of the
        :class:`~tianshou.env.BaseVectorEnv` class.
    :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer` class.
        If set to None, it will not store the data. Default to None.
    :param function preprocess_fn: a function called before the data has been added to
        the buffer, see issue #42 and :ref:`preprocess_fn`. Default to None.
    :param bool exploration_noise: determine whether the action needs to be modified
        with corresponding policy's exploration noise. If so, "policy.
        exploration_noise(act, batch)" will be called automatically to add the
        exploration noise into action. Default to False.
    :param function initialize_state_fn: a function called before the data
        has been added to the buffer, see issue #42, defaults to ``None``.
    :param int stat_size: for the moving average of recording speed, defaults
        to 100.
    :param reward_aggregator: support either VecTotal or VecMovAvg. 
    :param reward_logger: supports custom reward logging.

    The "preprocess_fn" is a function called before the data has been added to the
    buffer with batch format. It will receive only "obs" and "env_id" when the
    collector resets the environment, and will receive the keys "obs_next", "rew",
    "terminated", "truncated, "info", "policy" and "env_id" in a normal env step.
    Alternatively, it may also accept the keys "obs_next", "rew", "done", "info",
    "policy" and "env_id".
    It returns either a dict or a :class:`~tianshou.data.Batch` with the modified
    keys and values. Examples are in "test/base/test_collector.py".

    .. note::

        Please make sure the given environment has a time limitation if using n_episode
        collect option.

    .. note::

        In past versions of Tianshou, the replay buffer that was passed to `__init__`
        was automatically reset. This is not done in the current implementation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: Union[gym.Env, BaseVectorEnv],
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
        # customized tianshou parameters
        initialize_state_fn: Callable[[Any], Union[dict, Batch]] = None,
        stat_size: Optional[int] = 100,
        reward_aggregator: object = VecTotal,
        reward_logger: object = BaseRewardLogger,
        benchmark_logger: object = None,
    ) -> None:
        super().__init__()
        if isinstance(env, gym.Env) and not hasattr(env, "__len__"):
            warnings.warn("Single environment detected, wrap to DummyVectorEnv.")
            self.env = DummyVectorEnv([lambda: env])  # type: ignore
        else:
            self.env = env  # type: ignore
        self.env_num = len(self.env)
        self.exploration_noise = exploration_noise
        self._assign_buffer(buffer)
        self.policy = policy
        self.preprocess_fn = preprocess_fn
        self._action_space = self.env.action_space
        """ customized tianshou attributes """
        self.initialize_state_fn = initialize_state_fn
        self.stat_size = stat_size
        self.reward_aggregator = reward_aggregator
        self.reward_logger = reward_logger
        self.benchmark_logger = benchmark_logger        
        # avoid creating attribute outside __init__        
        self.reset(False)

    def _assign_buffer(self, buffer: Optional[ReplayBuffer]) -> None:
        """Check if the buffer matches the constraint."""
        if buffer is None:
            buffer = VectorReplayBuffer(self.env_num, self.env_num)
        elif isinstance(buffer, ReplayBufferManager):
            assert buffer.buffer_num >= self.env_num
            if isinstance(buffer, CachedReplayBuffer):
                assert buffer.cached_buffer_num >= self.env_num
        else:  # ReplayBuffer or PrioritizedReplayBuffer
            assert buffer.maxsize > 0
            if self.env_num > 1:
                if isinstance(buffer, ReplayBuffer):
                    buffer_type = "ReplayBuffer"
                    vector_type = "VectorReplayBuffer"
                if isinstance(buffer, PrioritizedReplayBuffer):
                    buffer_type = "PrioritizedReplayBuffer"
                    vector_type = "PrioritizedVectorReplayBuffer"
                raise TypeError(
                    f"Cannot use {buffer_type}(size={buffer.maxsize}, ...) to collect "
                    f"{self.env_num} envs,\n\tplease use {vector_type}(total_size="
                    f"{buffer.maxsize}, buffer_num={self.env_num}, ...) instead."
                )
        self.buffer = buffer

    def reset(
        self,
        reset_buffer: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reset the environment, statistics, current data and possibly replay memory.

        :param bool reset_buffer: if true, reset the replay buffer that is attached
            to the collector.
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)
        """
        # use empty Batch for "state" so that self.data supports slicing
        # convert empty Batch to None when passing data to policy
        self.data = Batch(
            obs={},
            act={},
            rew={},
            terminated={},
            truncated={},
            done={},
            obs_next={},
            info={},
            policy={}
        )
        self.reset_env(gym_reset_kwargs)
        if reset_buffer:
            self.reset_buffer()
        self.reset_stat()
        """ Customized tianshou reset """


    def reset_stat(self) -> None:
        """Reset the statistic variables."""
        self.collect_step, self.collect_episode, self.collect_time = 0, 0, 0.0
        """ Customized tianshou reset_stat"""
        self.step_speed = MovAvg(self.stat_size)
        self.episode_speed = MovAvg(self.stat_size)

    def reset_buffer(self, keep_statistics: bool = False) -> None:
        """Reset the data buffer."""
        self.buffer.reset(keep_statistics=keep_statistics)

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:        
        """Reset all of the environments."""
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs, info = self.env.reset(**gym_reset_kwargs)                
        if self.preprocess_fn:
            obs = self.preprocess_fn(obs=obs)

        self.data.info = info
        self.data.obs = obs     # (env_num,)        

    def _reset_state(self, id: Union[int, List[int]]) -> None:
        """Reset the hidden state: self.data.state[id]."""
        if hasattr(self.data.policy, "hidden_state"):
            state = self.data.policy.hidden_state  # it is a reference
            if isinstance(state, torch.Tensor):
                state[id].zero_()
            elif isinstance(state, np.ndarray):
                state[id] = None if state.dtype == object else 0
            elif isinstance(state, Batch):
                state.empty_(id)

    def _reset_env_with_ids(
        self,
        local_ids: Union[List[int], np.ndarray],
        global_ids: Union[List[int], np.ndarray],
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        gym_reset_kwargs = gym_reset_kwargs if gym_reset_kwargs else {}
        obs_reset, info = self.env.reset(global_ids, **gym_reset_kwargs)
        if self.preprocess_fn:
            obs_reset = self.preprocess_fn(obs=obs_reset)
        self.data.info[local_ids] = info
        self.data.obs_next[local_ids] = obs_reset

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        render_mode: Optional[str] =None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode.

        To ensure unbiased sampling result with n_episode option, this function will
        first collect ``n_episode - env_num`` episodes, then for the last ``env_num``
        episodes, they will be collected evenly from each env.

        :param int n_step: how many steps you want to collect.
        :param int n_episode: how many episodes you want to collect.
        :param bool random: whether to use random policy for collecting data. Default
            to False.
        :param float render: the sleep time between rendering consecutive frames.
            Default to None (no rendering).
        :param bool no_grad: whether to retain gradient in policy.forward(). Default to
            True (no gradient retaining).
        :param gym_reset_kwargs: extra keyword arguments to pass into the environment's
            reset function. Defaults to None (extra keyword arguments)

        .. note::

            One and only one collection number specification is permitted, either
            ``n_step`` or ``n_episode``.

        :return: A dict including the following keys

            * ``n/ep`` collected number of episodes.
            * ``n/st`` collected number of steps.
            * ``rews`` array of episode reward over collected episodes.
            * ``lens`` array of episode length over collected episodes.
            * ``idxs`` array of episode start index in buffer over collected episodes.
            * ``rew`` mean of episodic rewards.
            * ``len`` mean of episodic lengths.
            * ``rew_std`` standard error of episodic rewards.
            * ``len_std`` standard error of episodic lengths.
            * ``frames`` step-wise frames over collected episodes.
            * ``agent_step_rewards`` step-wise individual agent's reward over 
                    collected episodes.
        """        
        if n_step is not None:
            assert n_episode is None, (
                f"Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
            if not n_step % self.env_num == 0:
                warnings.warn(
                    f"n_step={n_step} is not a multiple of #env ({self.env_num}), "
                    "which may cause extra transitions collected into the buffer."
                )
            ready_env_ids = np.arange(self.env_num)
        elif n_episode is not None:
            assert n_episode > 0, \
                f"n_episode={n_episode} must be positive."
            # assert this to make tracking step-wise agent rewards and frames easier
            assert n_episode % self.env_num == 0 or self.env_num % n_episode == 0,\
                f"n_episode={n_episode} must be a multiple of #env ({self.env_num})."
            ready_env_ids = np.arange(min(self.env_num, n_episode))            
            # at initial reset, only self.data.obs has data, therefore we need to slice this 
            # to ensure there's no dimension mismatch error
            self.data.obs = self.data.obs[:min(self.env_num, n_episode)]            
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []
        # track extra information for analysis
        if n_episode is not None:
            frames = [[] for _ in range(n_episode)]
            step_agent_rews = [[] for _ in range(n_episode)]
            step_agent_intr_rews = [[] for _ in range(n_episode)]
        else:
            frames = [[] for _ in range(self.env_num)]
            step_agent_rews = [[] for _ in range(self.env_num)]
            step_agent_intr_rews = [[] for _ in range(self.env_num)]

        while True:
            assert len(self.data) == len(ready_env_ids)
            # restore the state: if the last state is None, it won't store
            last_state = self.data.policy.pop("hidden_state", None)
            # get the next action
            if random:
                try:
                    act_sample = [
                        self._action_space[i].sample() for i in ready_env_ids
                    ]
                except TypeError:  # envpool's action space is not for per-env
                    act_sample = [self._action_space.sample() for _ in ready_env_ids]
                act_sample = self.policy.map_action_inverse(act_sample)  # type: ignore
                self.data.update(act=act_sample)
            else:
                if no_grad:
                    with torch.no_grad():  # faster than retain_grad version
                        # self.data.obs will be used by agent to get result                        
                        cpu_percent_start = psutil.cpu_percent(interval=None)
                        result = self.policy(self.data, last_state)
                        cpu_percent_end = psutil.cpu_percent(interval=None)
                        avg_cpu = (cpu_percent_start + cpu_percent_end) / 2
                        #print(f"Average CPU usage during self.policy: {avg_cpu}")
                else:
                    result = self.policy(self.data, last_state)
                # update state / policy into self.data
                policy = result.get("policy", Batch())                
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                # update action for multi-agent into self.data 
                
                act = []
                for k, v in result.items():
                    act.append(v.act)
                act = np.stack(to_numpy(act), axis=0)    # (n_agent, env_eps)
                act = np.swapaxes(act, 1, 0)    # (n_env, env_eps)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)                
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            # only processed if self.action_scaling it True
            action_remap = self.policy.map_action(self.data.act)
            # step in env, output shape: (num_ep, num_agents)            
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,  # type: ignore
                ready_env_ids
            )
            # organize step_agent_rews as (num_episode, n_ep_steps, rew_shape)                       
            for i, r in enumerate(rew):
                if n_episode is not None:
                    idx = i + episode_count
                else:
                    idx = i
                step_agent_rews[idx].append(deepcopy(r))
            
            done = np.logical_or(terminated, truncated)

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                done=done,
                info=info
            )            
            if self.preprocess_fn:                
                obs_next = self.preprocess_fn(obs=obs_next)
                self.data.obs_next = obs_next         
                

            if render_mode == "rgb_array":
                # organize frames as (num_episode, n_ep_steps, frames_shape)
                frame = self.env.render()               
                for i, f in enumerate(frame):
                    if n_episode is not None:
                        idx = i + episode_count
                    else:
                        idx = i
                    frames[idx].append(f)

            if render:                                
                if render > 0 and not np.isclose(render, 0):
                    time.sleep(render)
                        
            # add data into the buffer
            ptr, ep_rew, ep_len, ep_idx = self.buffer.add(
                self.data, buffer_ids=ready_env_ids
            )
            # collect statistics
            step_count += len(ready_env_ids)             
            if np.any(done):                
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]                             
                episode_count += len(env_ind_local)
                episode_lens.append(ep_len[env_ind_local])
                episode_rews.append(ep_rew[env_ind_local])
                episode_start_indices.append(ep_idx[env_ind_local])
                # now we copy obs_next to obs, but since there might be
                # finished episodes, we have to reset finished envs first.
                self._reset_env_with_ids(
                    env_ind_local, env_ind_global, gym_reset_kwargs
                )
                for i in env_ind_local:
                    self._reset_state(i)

                # remove surplus env id from ready_env_ids
                # to avoid bias in selecting environments
                if n_episode:
                    surplus_env_num = len(ready_env_ids) - (n_episode - episode_count)                    
                    if surplus_env_num > 0:
                        mask = np.ones_like(ready_env_ids, dtype=bool)
                        #mask[env_ind_local[:surplus_env_num]] = False
                        # surplus starts at the end of the list because index for action 
                        # is the same as the env_id
                        mask[env_ind_local[-surplus_env_num:]] = False
                        ready_env_ids = ready_env_ids[mask]
                        self.data = self.data[mask]

            self.data.obs = self.data.obs_next

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

        if n_episode:
            self.data = Batch(
                obs={},
                act={},
                rew={},
                terminated={},
                truncated={},
                done={},
                obs_next={},
                info={},
                policy={}
            )
            self.reset_env()

        if episode_count > 0:
            rews, lens, idxs = list(
                map(
                    np.concatenate,
                    [episode_rews, episode_lens, episode_start_indices]
                )
            )
            rew_mean, rew_std = rews.mean(), rews.std()
            len_mean, len_std = lens.mean(), lens.std()
        else:
            rews, lens, idxs = np.array([]), np.array([], int), np.array([], int)
            rew_mean = rew_std = len_mean = len_std = 0

        output = {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
            "step_agent_rews": step_agent_rews
        }
        # store rendered frames
        if render_mode == "rgb_array":
            output["frames"] = frames
        return output
