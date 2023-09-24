import gym
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe
from typing import List, Tuple, Union, Optional, Callable


from tianshou.env.utils import CloudpickleWrapper


class BaseVectorEnv(ABC, gym.Env):
    """Base class for vectorized environments wrapper. Usage:
    ::

        env_num = 8
        envs = VectorEnv([lambda: gym.make(task) for _ in range(env_num)])
        assert len(envs) == env_num

    It accepts a list of environment generators. In other words, an environment
    generator ``efn`` of a specific task means that ``efn()`` returns the
    environment of the given task, for example, ``gym.make(task)``.

    All of the VectorEnv must inherit :class:`~tianshou.env.BaseVectorEnv`.
    Here are some other usages:
    ::

        envs.seed(2)  # which is equal to the next line
        envs.seed([2, 3, 4, 5, 6, 7, 8, 9])  # set specific seed for each env
        obs = envs.reset()  # reset all environments
        obs = envs.reset([0, 5, 7])  # reset 3 specific environments
        obs, rew, done, info = envs.step([1] * 8)  # step synchronously
        envs.render()  # render all environments
        envs.close()  # close all environments
    """

    def __init__(
            self, 
            env_fns: List[Callable[[], gym.Env]]
            ) -> None:
        self._env_fns = env_fns
        self.env_num = len(env_fns)
        self._obs = None
        self._rew = None
        self._terminnated = None
        self._truncated = None
        self._info = None

    def __len__(self) -> int:
        """Return len(self), which is the number of environments."""
        return self.env_num

    def __getattribute__(self, key):
        """Switch between the default attribute getter or one
           looking at wrapped environment level depending on the key."""
        if key not in ('observation_space', 'action_space'):
            return super().__getattribute__(key)
        else:
            return self.__getattr__(key)

    @abstractmethod
    def __getattr__(
        self, 
        key: str
        ) -> None:
        """Try to retrieve an attribute from each individual wrapped
           environment, if it does not belong to the wrapping vector
           environment class."""
        pass

    @abstractmethod
    def reset(
        self, 
        id: Optional[Union[int, List[int]]] = None
        ) -> None:
        """Reset the state of all the environments and return initial
        observations if id is ``None``, otherwise reset the specific
        environments with given id, either an int or a list.
        """
        pass

    @abstractmethod
    def step(
        self, 
        action: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run one timestep of all the environments’ dynamics. When the end of
        episode is reached, you are responsible for calling reset(id) to reset
        this environment’s state.

        Accept a batch of action and return a tuple (obs, rew, done, info).

        :param numpy.ndarray action: a batch of action provided by the agent.

        :return: A tuple including four items:

            * ``obs`` a numpy.ndarray, the agent's observation of current \
                environments
            * ``rew`` a numpy.ndarray, the amount of rewards returned after \
                previous actions
            * ``done`` a numpy.ndarray, whether these episodes have ended, in \
                which case further step() calls will return undefined results
            * ``info`` a numpy.ndarray, contains auxiliary diagnostic \
                information (helpful for debugging, and sometimes learning)
        """
        pass

    @abstractmethod
    def seed(
        self, 
        seed: Optional[Union[int, List[int]]] = None
        ) -> None:
        """Set the seed for all environments.

        Accept ``None``, an int (which will extend ``i`` to
        ``[i, i + 1, i + 2, ...]``) or a list.

        :return: The list of seeds used in this env's random number generators.
        The first value in the list should be the "main" seed, or the value
        which a reproducer should pass to 'seed'.
        """
        pass

    @abstractmethod
    def render(
        self, 
        **kwargs
        ) -> None:
        """Render all of the environments."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close all of the environments.

        Environments will automatically close() themselves when garbage
        collected or when the program exits.
        """
        pass


class VectorEnv(BaseVectorEnv):
    """Dummy vectorized environment wrapper, implemented in for-loop.

    .. seealso::
    ********


        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(
            self, 
            env_fns: List[Callable[[], gym.Env]]
            ) -> None:
        super().__init__(env_fns)
        self.envs = [_() for _ in env_fns]

    def __getattr__(
            self, 
            key: str
            ):
        return [getattr(env, key) if hasattr(env, key) else None
                for env in self.envs]    

    def reset(
            self, 
            id: Optional[Union[int, List[int]]] = None
            ) -> None:
        obs_list, info_list = [], []
        if id is None:
            # get output for all envs if no id is specified
            for e in self.envs:
                obs, info = e.reset()                
                obs_list.append(obs)
                info_list.append(info)
            self._obs = np.stack(obs_list)
            self._info = np.stack(info_list)
            return self._obs, self._info
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                obs, info = self.envs[i].reset()
                self._obs[i] = obs
                self._info[i] = info
            return self._obs[id], self._info[id]

    def step(
            self, 
            action: np.ndarray,
            id: Optional[Union[int, List[int], np.ndarray]] = None,
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if id is None:
            result = [e.step(a) for e, a in zip(self.envs, action)]
            self._obs, self._rew, self._terminated, self._truncated, self._info \
                = zip(*result)
            self._obs = np.stack(self._obs)
            self._rew = np.stack(self._rew)
            self._terminated = np.stack(self._terminated)
            self._truncated = np.stack(self._truncated)
            self._info = np.stack(self._info)
            return self._obs, self._rew, self._terminated, self._truncated, self._info
        else:
            reward_list, terminated_list, truncated_list = [], [], []
            if np.isscalar(id):
                id = [id]
            for i in id:
                obs, rew, terminated, truncated, info \
                    = self.envs[i].step(action[i])
                self._obs[i] = obs
                if self._rew is not None:
                    # when one of them is not None, the others should not be
                    self._rew[i] = rew
                    self._terminated[i] = terminated                
                    self._truncated[i] = truncated
                else:
                    reward_list.append(rew)
                    terminated_list.append(terminated)
                    truncated_list.append(truncated)                    
                self._info[i] = info
            if self._rew is None:
                self._rew = np.stack(reward_list)
                self._terminated = np.stack(terminated_list)
                self._truncated = np.stack(truncated_list)
            # only return stepped envs
            return self._obs[id], self._rew[id], self._terminated[id], \
                self._truncated[id], self._info[id]

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:        
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        result = []
        for e, s in zip(self.envs, seed):
            e.reset(seed=s)            
        return result

    def render(self, **kwargs) -> None:
        result = []
        for e in self.envs:
            if hasattr(e, 'render'):                
                result.append(e.render(**kwargs))
        return result

    def close(self) -> None:
        return [e.close() for e in self.envs]


def worker(parent, p, env_fn_wrapper):
    parent.close()
    env = env_fn_wrapper.data()
    try:
        while True:
            cmd, data = p.recv()
            if cmd == 'step':
                p.send(env.step(data))
            elif cmd == 'reset':
                p.send(env.reset())
            elif cmd == 'close':
                p.send(env.close())
                p.close()
                break
            elif cmd == 'render':
                p.send(env.render(**data) if hasattr(env, 'render') else None)
            elif cmd == 'seed':
                p.send(env.seed(data) if hasattr(env, 'seed') else None)
            elif cmd == 'getattr':
                p.send(getattr(env, data) if hasattr(env, data) else None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class SubprocVectorEnv(BaseVectorEnv):
    """Vectorized environment wrapper based on subprocess.

    .. seealso::

        Please refer to :class:`~tianshou.env.BaseVectorEnv` for more detailed
        explanation.
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]) -> None:
        super().__init__(env_fns)
        self.closed = False
        self.parent_remote, self.child_remote = \
            zip(*[Pipe() for _ in range(self.env_num)])
        self.processes = [
            Process(target=worker, args=(
                parent, child, CloudpickleWrapper(env_fn)), daemon=True)
            for (parent, child, env_fn) in zip(
                self.parent_remote, self.child_remote, env_fns)
        ]
        for p in self.processes:
            p.start()
        for c in self.child_remote:
            c.close()

    def __getattr__(self, key):
        for p in self.parent_remote:
            p.send(['getattr', key])
        return [p.recv() for p in self.parent_remote]

    def step(
            self, 
            action: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert len(action) == self.env_num
        for p, a in zip(self.parent_remote, action):
            p.send(['step', a])
        result = [p.recv() for p in self.parent_remote]
        self._obs, self._rew, self._done, self._info = zip(*result)
        self._obs = np.stack(self._obs)
        self._rew = np.stack(self._rew)
        self._done = np.stack(self._done)
        self._info = np.stack(self._info)
        return self._obs, self._rew, self._done, self._info

    def reset(self, id: Optional[Union[int, List[int]]] = None) -> None:
        if id is None:
            for p in self.parent_remote:
                p.send(['reset', None])
            self._obs = np.stack([p.recv() for p in self.parent_remote])
            return self._obs
        else:
            if np.isscalar(id):
                id = [id]
            for i in id:
                self.parent_remote[i].send(['reset', None])
            for i in id:
                self._obs[i] = self.parent_remote[i].recv()
            return self._obs[id]

    def seed(self, seed: Optional[Union[int, List[int]]] = None) -> None:
        if np.isscalar(seed):
            seed = [seed + _ for _ in range(self.env_num)]
        elif seed is None:
            seed = [seed] * self.env_num
        for p, s in zip(self.parent_remote, seed):
            p.send(['seed', s])
        return [p.recv() for p in self.parent_remote]

    def render(self, **kwargs) -> None:
        for p in self.parent_remote:
            p.send(['render', kwargs])
        return [p.recv() for p in self.parent_remote]

    def close(self) -> None:
        if self.closed:
            return
        for p in self.parent_remote:
            p.send(['close', None])
        result = [p.recv() for p in self.parent_remote]
        self.closed = True
        for p in self.processes:
            p.join()
        return result

