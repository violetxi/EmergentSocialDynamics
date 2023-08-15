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
from tianshou_elign.utils import MovAvg, VecMovAvg, VecTotal
from tianshou_elign.env import BaseVectorEnv
from tianshou_elign.env import BaseRewardLogger
#from tianshou_elign.data import ListReplayBuffer, to_numpy


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
            processed_data = self.preprocess_fn(
                obs=obs, info=info, env_id=np.arange(self.env_num)
            )
            obs = processed_data.get("obs", obs)
            info = processed_data.get("info", info)
        self.data.info = info
        self.data.obs = obs

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
            processed_data = self.preprocess_fn(
                obs=obs_reset, info=info, env_id=global_ids
            )
            obs_reset = processed_data.get("obs", obs_reset)
            info = processed_data.get("info", info)
        self.data.info[local_ids] = info

        self.data.obs_next[local_ids] = obs_reset

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
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
            assert n_episode > 0
            ready_env_ids = np.arange(min(self.env_num, n_episode))
            self.data = self.data[:min(self.env_num, n_episode)]
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
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / policy into self.data
                policy = result.get("policy", Batch())                
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                # update action for multi-agent into self.data                
                act = [[]for _ in range(self.env_num)]
                for i, (k, v) in enumerate(result.items()):
                    for j, action in enumerate(to_numpy(v.act)):
                        act[j].append(action) 
                act = np.stack(act, axis=0)                                
                #act = to_numpy(result.act)        
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)                
                self.data.update(policy=policy, act=act)

            # get bounded and remapped actions first (not saved into buffer)
            # only processed if self.action_scaling it True
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,  # type: ignore
                ready_env_ids
            )            
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
                self.data.update(
                    self.preprocess_fn(
                        obs_next=self.data.obs_next,
                        rew=self.data.rew,
                        done=self.data.done,
                        info=self.data.info,
                        policy=self.data.policy,
                        env_id=ready_env_ids,
                        act=self.data.act,
                    )
                )

            if render:
                self.env.render()
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
                        mask[env_ind_local[:surplus_env_num]] = False
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

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }


class AsyncCollector(Collector):
    """Async Collector handles async vector environment.

    The arguments are exactly the same as :class:`~tianshou.data.Collector`, please
    refer to :class:`~tianshou.data.Collector` for more detailed explanation.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseVectorEnv,
        buffer: Optional[ReplayBuffer] = None,
        preprocess_fn: Optional[Callable[..., Batch]] = None,
        exploration_noise: bool = False,
    ) -> None:
        # assert env.is_async
        warnings.warn("Using async setting may collect extra transitions into buffer.")
        super().__init__(
            policy,
            env,
            buffer,
            preprocess_fn,
            exploration_noise,
        )

    def reset_env(self, gym_reset_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().reset_env(gym_reset_kwargs)
        self._ready_env_ids = np.arange(self.env_num)

    def collect(
        self,
        n_step: Optional[int] = None,
        n_episode: Optional[int] = None,
        random: bool = False,
        render: Optional[float] = None,
        no_grad: bool = True,
        gym_reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect a specified number of step or episode with async env setting.

        This function doesn't collect exactly n_step or n_episode number of
        transitions. Instead, in order to support async setting, it may collect more
        than given n_step or n_episode transitions and save into buffer.

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
        """
        # collect at least n_step or n_episode
        if n_step is not None:
            assert n_episode is None, (
                "Only one of n_step or n_episode is allowed in Collector."
                f"collect, got n_step={n_step}, n_episode={n_episode}."
            )
            assert n_step > 0
        elif n_episode is not None:
            assert n_episode > 0
        else:
            raise TypeError(
                "Please specify at least one (either n_step or n_episode) "
                "in AsyncCollector.collect()."
            )

        ready_env_ids = self._ready_env_ids

        start_time = time.time()

        step_count = 0
        episode_count = 0
        episode_rews = []
        episode_lens = []
        episode_start_indices = []

        while True:
            whole_data = self.data
            self.data = self.data[ready_env_ids]
            assert len(whole_data) == self.env_num  # major difference
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
                        result = self.policy(self.data, last_state)
                else:
                    result = self.policy(self.data, last_state)
                # update state / act / policy into self.data
                policy = result.get("policy", Batch())
                assert isinstance(policy, Batch)
                state = result.get("state", None)
                if state is not None:
                    policy.hidden_state = state  # save state into buffer
                act = to_numpy(result.act)
                if self.exploration_noise:
                    act = self.policy.exploration_noise(act, self.data)
                self.data.update(policy=policy, act=act)

            # save act/policy before env.step
            try:
                whole_data.act[ready_env_ids] = self.data.act
                whole_data.policy[ready_env_ids] = self.data.policy
            except ValueError:
                _alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                whole_data[ready_env_ids] = self.data  # lots of overhead

            # get bounded and remapped actions first (not saved into buffer)
            action_remap = self.policy.map_action(self.data.act)
            # step in env
            obs_next, rew, terminated, truncated, info = self.env.step(
                action_remap,  # type: ignore
                ready_env_ids
            )
            done = np.logical_or(terminated, truncated)

            # change self.data here because ready_env_ids has changed
            try:
                ready_env_ids = info["env_id"]
            except Exception:
                ready_env_ids = np.array([i["env_id"] for i in info])
            self.data = whole_data[ready_env_ids]

            self.data.update(
                obs_next=obs_next,
                rew=rew,
                terminated=terminated,
                truncated=truncated,
                info=info
            )
            if self.preprocess_fn:
                try:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            terminated=self.data.terminated,
                            truncated=self.data.truncated,
                            info=self.data.info,
                            env_id=ready_env_ids,
                            act=self.data.act,
                        )
                    )
                except TypeError:
                    self.data.update(
                        self.preprocess_fn(
                            obs_next=self.data.obs_next,
                            rew=self.data.rew,
                            done=self.data.done,
                            info=self.data.info,
                            env_id=ready_env_ids,
                            act=self.data.act,
                        )
                    )

            if render:
                self.env.render()
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

            try:
                whole_data.obs[ready_env_ids] = self.data.obs_next
                whole_data.rew[ready_env_ids] = self.data.rew
                whole_data.done[ready_env_ids] = self.data.done
                whole_data.info[ready_env_ids] = self.data.info
            except ValueError:
                _alloc_by_keys_diff(whole_data, self.data, self.env_num, False)
                self.data.obs = self.data.obs_next
                whole_data[ready_env_ids] = self.data  # lots of overhead
            self.data = whole_data

            if (n_step and step_count >= n_step) or \
                    (n_episode and episode_count >= n_episode):
                break

        self._ready_env_ids = ready_env_ids

        # generate statistics
        self.collect_step += step_count
        self.collect_episode += episode_count
        self.collect_time += max(time.time() - start_time, 1e-9)

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

        return {
            "n/ep": episode_count,
            "n/st": step_count,
            "rews": rews,
            "lens": lens,
            "idxs": idxs,
            "rew": rew_mean,
            "len": len_mean,
            "rew_std": rew_std,
            "len_std": len_std,
        }





# class Collector(object):
#     """The :class:`~tianshou.data.Collector` enables the policy to interact
#     with different types of environments conveniently.

#     :param policy: an instance of the :class:`~tianshou.policy.BasePolicy`
#         class.
#     :param env: a ``gym.Env`` environment or an instance of the
#         :class:`~tianshou.env.BaseVectorEnv` class.
#     :param buffer: an instance of the :class:`~tianshou.data.ReplayBuffer`
#         class, or a list of :class:`~tianshou.data.ReplayBuffer`. If set to
#         ``None``, it will automatically assign a small-size
#         :class:`~tianshou.data.ReplayBuffer`.
#     :param function preprocess_fn: a function called before the data has been
#         added to the buffer, see issue #42, defaults to ``None``.
#     :param function initialize_state_fn: a function called before the data
#         has been added to the buffer, see issue #42, defaults to ``None``.
#     :param int stat_size: for the moving average of recording speed, defaults
#         to 100.
#     :param reward_aggregator: support either VecTotal or VecMovAvg. 
#     :param reward_logger: supports custom reward logging.

#     The ``preprocess_fn`` is a function called before the data has been added
#     to the buffer with batch format, which receives up to 7 keys as listed in
#     :class:`~tianshou.data.Batch`. It will receive with only ``obs`` when the
#     collector resets the environment. It returns either a dict or a
#     :class:`~tianshou.data.Batch` with the modified keys and values. Examples
#     are in "test/base/test_collector.py".

#     Example:
#     ::

#         policy = PGPolicy(...)  # or other policies if you wish
#         env = gym.make('CartPole-v0')
#         replay_buffer = ReplayBuffer(size=10000)
#         # here we set up a collector with a single environment
#         collector = Collector(policy, env, buffer=replay_buffer)

#         # the collector supports vectorized environments as well
#         envs = VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(3)])
#         buffers = [ReplayBuffer(size=5000) for _ in range(3)]
#         # you can also pass a list of replay buffer to collector, for multi-env
#         # collector = Collector(policy, envs, buffer=buffers)
#         collector = Collector(policy, envs, buffer=replay_buffer)

#         # collect at least 3 episodes
#         collector.collect(n_episode=3)
#         # collect 1 episode for the first env, 3 for the third env
#         collector.collect(n_episode=[1, 0, 3])
#         # collect at least 2 steps
#         collector.collect(n_step=2)
#         # collect episodes with visual rendering (the render argument is the
#         #   sleep time between rendering consecutive frames)
#         collector.collect(n_episode=1, render=0.03)

#         # sample data with a given number of batch-size:
#         batch_data = collector.sample(batch_size=64)
#         # policy.learn(batch_data)  # btw, vanilla policy gradient only
#         #   supports on-policy training, so here we pick all data in the buffer
#         batch_data = collector.sample(batch_size=0)
#         policy.learn(batch_data)
#         # on-policy algorithms use the collected data only once, so here we
#         #   clear the buffer
#         collector.reset_buffer()

#     For the scenario of collecting data from multiple environments to a single
#     buffer, the cache buffers will turn on automatically. It may return the
#     data more than the given limitation.

#     .. note::

#         Please make sure the given environment has a time limitation.
#     """

#     def __init__(self,
#                  policy: BasePolicy,
#                  env: Union[gym.Env, BaseVectorEnv],
#                  buffer: Optional[Union[ReplayBuffer, List[ReplayBuffer]]] = None,
#                  preprocess_fn: Callable[[Any], Union[dict, Batch]] = None,
#                  initialize_state_fn: Callable[[Any], Union[dict, Batch]] = None,
#                  stat_size: Optional[int] = 100,
#                  reward_aggregator: object = VecTotal,
#                  reward_logger: object = BaseRewardLogger,
#                  benchmark_logger: object = None,
#                  **kwargs) -> None:
#         super().__init__()
#         self.env = env
#         self.env_num = len(env)        
#         self.num_agents = len(env.possible_agents[0])
#         self.collect_time = 0
#         self.collect_step = 0
#         self.collect_episode = 0
#         self.buffer = buffer
#         self.policy = policy
#         self.preprocess_fn = preprocess_fn
#         self.initialize_state_fn = initialize_state_fn      
#         self.process_fn = policy.process_fn
#         self._multi_env = isinstance(env, BaseVectorEnv)
#         self._multi_buf = False  # True if buf is a list
#         # need multiple cache buffers only if storing in one buffer
#         self._cached_buf = []
#         if self._multi_env:
#             if isinstance(self.buffer, list):
#                 assert len(self.buffer) == self.env_num, \
#                     'The number of data buffer does not match the number of ' \
#                     'input env.'
#                 self._multi_buf = True
#             elif isinstance(self.buffer, ReplayBuffer) or self.buffer is None:
#                 self._cached_buf = [
#                     ListReplayBuffer() for _ in range(self.env_num)]                
#             else:
#                 raise TypeError('The buffer in data collector is invalid!')
#         self.stat_size = stat_size
#         self.reward_aggregator = reward_aggregator
#         self.reward_logger = reward_logger
#         self.benchmark_logger = benchmark_logger
#         self.reset_stat()

#     def reset_stat(self) -> None:
#         """Reset all related variables in the collector."""
#         # state over batch is either a list, an np.ndarray, or a torch.Tensor
#         self.reset_env()
#         self.reset_buffer()
#         self.step_speed = MovAvg(self.stat_size)
#         self.episode_speed = MovAvg(self.stat_size)
#         self.collect_step = 0
#         self.collect_episode = 0
#         self.collect_time = 0

#     def reset_buffer(self) -> None:
#         """Reset the main data buffer."""
#         if self._multi_buf:
#             for b in self.buffer:
#                 b.reset()
#         else:
#             if self.buffer is not None:
#                 self.buffer.reset()        

#     def get_env_num(self) -> int:
#         """Return the number of environments the collector have."""
#         return self.env_num

#     def reset_env(self) -> None:
#         """Reset all of the environment(s)' states and reset all of the cache
#         buffers (if need).
#         """
#         self.state = None
#         self._reset_state(list(range(self.get_env_num())))        
#         self._obs = self.env.reset()
#         if not self._multi_env:
#             self._obs = self._make_batch(self._obs)
#         if self.preprocess_fn:
#             result = self.preprocess_fn(Batch(
#                 obs=self._obs, policy=None))
#             self._obs = result.get('obs', self._obs)
#         self._act = self._rew = self._done = self._info = None
#         if self._multi_env:
#             self.reward = np.zeros((self.env_num, self.num_agents))
#             self.length = np.zeros(self.env_num)
#         else:
#             self.reward = np.zeros((1, self.num_agents))
#             self.length = 0
#         self.reward_agg = self.reward_aggregator(
#                 ndim=self.env_num * self.num_agents)
#         if self.benchmark_logger:
#             self.benchmark_logger.reset()
#         for b in self._cached_buf:
#             b.reset()

#     def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
#         """Reset all the seed(s) of the given environment(s)."""
#         if hasattr(self.env, 'seed'):
#             return self.env.seed(seed)

#     def render(self, **kwargs) -> None:
#         """Render all the environment(s)."""
#         if hasattr(self.env, 'render'):
#             return self.env.render(**kwargs)

#     def close(self) -> None:
#         """Close the environment(s)."""
#         if hasattr(self.env, 'close'):
#             self.env.close()

#     def _make_batch(self, data: Any) -> Union[Any, np.ndarray]:
#         """Return [data]."""
#         if isinstance(data, np.ndarray):
#             return data[None]
#         else:
#             return np.array([data])

#     def _reset_state(self, id: Union[int, List[int]]) -> None:
#         """Reset self.state[id]."""
#         if self.initialize_state_fn is not None:
#             self.state = self.initialize_state_fn(self.state, id)
#         elif self.state is None:
#             return
#         elif isinstance(self.state, list):
#             self.state[id] = None
#         elif isinstance(self.state, (dict, Batch)):
#             for k in self.state.keys():
#                 if isinstance(self.state[k], list):
#                     self.state[k][id] = None
#                 elif isinstance(self.state[k], (torch.Tensor, np.ndarray)):
#                     self.state[k][id] = 0
#         elif isinstance(self.state, (torch.Tensor, np.ndarray)):
#             self.state[id] = 0

#     def collect(self,
#                 n_step: int = 0,
#                 n_episode: Union[int, List[int]] = 0,
#                 random: bool = False,
#                 render: Optional[float] = None,
#                 render_mode: Optional[str] = None,
#                 log_fn: Optional[Callable[[dict], None]] = None
#                 ) -> Dict[str, float]:
#         """Collect a specified number of step or episode.

#         :param int n_step: how many steps you want to collect.
#         :param n_episode: how many episodes you want to collect (in each
#             environment).
#         :param bool random: whether to use random policy for collecting data,
#             defaults to ``False``.
#         :type n_episode: int or list
#         :param float render: the sleep time between rendering consecutive
#             frames, defaults to ``None`` (no rendering).
#         :param function log_fn: a function which receives env info, typically
#             for tensorboard logging.

#         .. note::

#             One and only one collection number specification is permitted,
#             either ``n_step`` or ``n_episode``.

#         :return: A dict including the following keys

#             * ``n/ep`` the collected number of episodes.
#             * ``n/st`` the collected number of steps.
#             * ``v/st`` the speed of steps per second.
#             * ``v/ep`` the speed of episode per second.
#             * ``rew`` the mean reward over collected episodes.
#             * ``rews`` (n_ep, n_agents) the total reward for each agent 
#                         over collected episodes.
#             * ``len`` the mean length over collected episodes.
#         """
#         warning_count = 0
#         if not self._multi_env:
#             n_episode = np.sum(n_episode)
#         start_time = time.time()
#         if n_step is None:
#             n_step = 0
#         if n_episode is None:
#             n_episode = 0            
#         assert sum([(n_step != 0), (n_episode != 0)]) == 1, \
#             "One and only one collection number specification is permitted!"
#         cur_step = 0
#         cur_episode = np.zeros(self.env_num) if self._multi_env else 0
#         reward_logger = self.reward_logger()
#         length_sum = 0
#         frames = []
#         # keep tracks of total reward for each agent over collected episodes
#         ep_total_reward = []
#         while True:
#             if warning_count >= 100000:
#                 warnings.warn(
#                     'There are already many steps in an episode. '
#                     'You should add a time limitation to your environment!',
#                     Warning)
            
#             # Initialize batch.                        
#             batch = Batch(
#                 obs=self._obs, act=self._act, rew=self._rew,
#                 done=self._done, obs_next=None, info=self._info)
#             if self.preprocess_fn:
#                 batch = self.preprocess_fn(batch)                

#             # Choose actions.
#             if random:
#                 action_space = self.env.action_space
#                 if isinstance(action_space, list):
#                     result = Batch(act=[a.sample() for a in action_space])
#                 else:
#                     result = Batch(act=self._make_batch(action_space.sample()))
#             else:
#                 with torch.no_grad():
#                     result = self.policy(batch)

#             # @TODO: get hidden state implement this when model needs it
#             self.state = result.get('state', None)
#             # Take environment step.
#             # need to convert action from {agent_id: action (env_num,)} 
#             # to np.array (env_num, agent_num)
#             agent_ids = self.env.possible_agents[0]
#             self._act = np.array([[0 for _ in agent_ids] for _ in range(self.env_num)])            
#             for agent_id, v in result.items():                
#                 for i in range(self.env_num):
#                     agent_idx = agent_ids.index(agent_id)
#                     self._act[i][agent_idx] = to_numpy(v.act[i])
            
#             obs_next, self._rew, self._done, self._info = self.env.step(
#                 self._act if self._multi_env else self._act[0])            

#             # Update batch elements.
#             if not self._multi_env:
#                 obs_next = self._make_batch(obs_next)
#                 self._rew = self._make_batch(self._rew)
#                 self._done = self._make_batch(self._done)
#                 self._info = self._make_batch(self._info)

#             # Logging and rendering.
#             if log_fn:
#                 log_fn(self._info if self._multi_env else self._info[0])
#             if self.benchmark_logger:
#                 self.benchmark_logger.add(self._info)
#             if render:
#                 if render_mode:
#                     frame = self.env.render(mode=render_mode)[0]
#                     frames.append(frame)
#                 else:
#                     self.env.render()
#                 if render > 0:
#                     time.sleep(render)

#             # Preprocess new batch of data with step-wise information
#             if self.preprocess_fn:
#                 result = self.preprocess_fn(Batch(
#                     obs=self._obs, act=self._act, rew=self._rew,
#                     done=self._done, obs_next=obs_next, info=self._info,
#                     ))                    
#                 self._obs = result.get('obs', self._obs)
#                 self._act = result.get('act', self._act)
#                 self._rew = result.get('rew', self._rew)
#                 self._done = result.get('done', self._done)
#                 obs_next = result.get('obs_next', obs_next)
#                 self._info = result.get('info', self._info)

#             # Update cummulative rewards. 
#             # self._rew: (num_env, num_agents) is step reward 
#             # self.reward: (num_env, num_agents) is cummulative rewards
#             self.length += 1            
#             if self._rew.ndim == 1:
#                 self.reward = self.reward_agg.add(self._rew).reshape(
#                         self.env_num, 1)
#             else:
#                 self.reward = self.reward_agg.add(self._rew.reshape(
#                         self.env_num * self.num_agents)).reshape(
#                                 self.env_num, self.num_agents)            

#             # Add data to buffer and update episode information with logging for multi-agent envs
#             if self._multi_env:           
#                 for i in range(self.env_num):
#                     data = {
#                         'obs': self._obs[i], 'act': self._act[i],
#                         'rew': self._rew[i], 'done': self._done[i],
#                         'obs_next': obs_next[i], 'info': self._info[i],
#                     }
#                     breakpoint()
#                     if self._cached_buf:
#                         warning_count += 1
#                         self._cached_buf[i].add(**data)
#                     elif self._multi_buf:
#                         warning_count += 1
#                         self.buffer[i].add(**data)
#                         cur_step += 1
#                     else:
#                         warning_count += 1
#                         if self.buffer is not None:
#                             self.buffer.add(**data)
#                         cur_step += 1

#                     # when all environment terminates, update reward_logger, self.reward                                    
#                     if np.all(self._done[i]):                        
#                         if n_step != 0 or np.isscalar(n_episode) or \
#                                 cur_episode[i] < n_episode[i]:
#                             cur_episode[i] += 1
#                             reward_logger.add(self.reward[i])
#                             length_sum += self.length[i]
#                             if self._cached_buf:
#                                 cur_step += len(self._cached_buf[i])
#                                 if self.buffer is not None:
#                                     self.buffer.update(self._cached_buf[i])
#                         ep_total_reward.append(deepcopy(self.reward[i]))
#                         self.reward = self.reward_agg.reset(
#                                 i*self.num_agents, (i+1)*self.num_agents).reshape(
#                             self.env_num, self.num_agents)
#                         if self.benchmark_logger:
#                             self.benchmark_logger.episode_end(i)
#                         self.length[i] = 0
#                         if self._cached_buf:
#                             self._cached_buf[i].reset()
#                 # if any environment terminates, reset its state and environment
#                 if np.any(self._done):
#                     done_envs = np.where(self._done[:, 0])[0]
#                     self._reset_state(done_envs)                    
#                     obs_next = self.env.reset(done_envs)
#                     if self.preprocess_fn:
#                          resnext = self.preprocess_fn(                            
#                             Batch(obs=obs_next),
#                             )
#                          obs_next = resnext.get('obs', obs_next)
                         
#                 if n_episode != 0:
#                     if isinstance(n_episode, list) and \
#                             (cur_episode >= np.array(n_episode)).all() or \
#                             np.isscalar(n_episode) and \
#                             cur_episode.sum() >= n_episode:
#                         break
#             else:
#                 if self.buffer is not None:
#                     self.buffer.add(
#                         self._obs[0], self._act[0], self._rew[0],
#                         self._done[0], obs_next[0], self._info[0],
#                     )                      
#                 cur_step += 1
#                 if np.all(self._done):
#                     cur_episode += 1
#                     reward_logger.add(self.reward[0])
#                     length_sum += self.length                    
#                     ep_total_reward.append(self.reward[i])
#                     self.reward = self.reward_agg.reset(0, self.num_agents)
#                     if self.benchmark_logger:
#                         self.benchmark_logger.episode_end(0)
#                     self.length = 0
#                     self._reset_state([0])
#                     obs_next = self._make_batch(self.env.reset())
#                     if self.preprocess_fn:
#                         resnext = self.preprocess_fn(                            
#                             Batch(obs=obs_next), id=0)
#                         obs_next = resnext.get('obs', obs_next)                        
#                 if n_episode != 0 and cur_episode >= n_episode:
#                     break
#             # 
#             if n_step != 0 and cur_step >= n_step:
#                 break
#             self._obs = obs_next
#         self._obs = obs_next

#         # Collect values for next iterations
#         if self._multi_env:
#             cur_episode = sum(cur_episode)
#         duration = max(time.time() - start_time, 1e-9)
#         self.step_speed.add(cur_step / duration)
#         self.episode_speed.add(cur_episode / duration)
#         self.collect_step += cur_step
#         self.collect_episode += cur_episode
#         self.collect_time += duration
#         n_episode = max(cur_episode, 1)
#         output = {
#             'n/ep': cur_episode,
#             'n/st': cur_step,
#             'v/st': self.step_speed.get(),
#             'v/ep': self.episode_speed.get(),
#             'len': length_sum / n_episode,
#             # @TODO: add flexible std computation, currently all episode 
#             # length will be the same
#             'len_std': 0.0,
#         }        
#         # Gather rewards to log.
#         for k, v in reward_logger.log().items():
#             output[k] = v                
#         output['rews'] = np.array(ep_total_reward)
#         # Gather benchmark data to log.
#         if self.benchmark_logger:
#             for k, v in self.benchmark_logger.log().items():
#                 output[k] = v

#         # Store rendered frames.
#         if render and render_mode == 'rgb_array':
#             output['frames'] = frames

#         return output

#     def sample(self, batch_size: int, global_step: int = None) -> Batch:
#         """Sample a data batch from the internal replay buffer. It will call
#         :meth:`~tianshou.policy.BasePolicy.process_fn` before returning
#         the final batch data.

#         :param int batch_size: ``0`` means it will extract all the data from
#             the buffer, otherwise it will extract the data with the given
#             batch_size.
#         :param global_step: what step of training are we on.
#         """
#         if self._multi_buf:
#             if batch_size > 0:
#                 lens = [len(b) for b in self.buffer]
#                 total = sum(lens)
#                 batch_index = np.random.choice(
#                     len(self.buffer), batch_size, p=np.array(lens) / total)
#             else:
#                 batch_index = np.array([])
#             batch_data = Batch()
#             for i, b in enumerate(self.buffer):
#                 cur_batch = (batch_index == i).sum()
#                 if batch_size and cur_batch or batch_size <= 0:
#                     batch, indice = b.sample(cur_batch)
#                     batch = self.process_fn(batch, b, indice,
#                                             global_step=global_step)
#                     batch_data.append(batch)
#         else:
#             batch_data, indice = self.buffer.sample(batch_size)
#             # batch_data = self.process_fn(batch_data, self.buffer, indice,
#             #                              global_step=global_step)            
#             batch_data, intr_rews = self.process_fn(batch_data, self.buffer, indice)
#         return batch_data, intr_rews, indice
