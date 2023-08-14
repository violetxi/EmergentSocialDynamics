import torch
import copy
import pprint
import warnings
import numpy as np
from functools import reduce
from numbers import Number
from typing import Any, List, Tuple, Union, Iterator, Optional

# Disable pickle warning related to torch, since it has been removed
# on torch master branch. See Pull Request #39003 for details:
# https://github.com/pytorch/pytorch/pull/39003
warnings.filterwarnings(
    "ignore", message="pickle support for Storage will be removed in 1.5.")


def _is_batch_set(data: Any) -> bool:
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (dict, Batch)):
            return True
    elif isinstance(data, np.ndarray):
        if isinstance(data.item(0), (dict, Batch)):
            return True
    return False


def _valid_bounds(length: int, index: Union[
        slice, int, np.integer, np.ndarray, List[int]]) -> bool:
    if isinstance(index, (int, np.integer)):
        return -length <= index and index < length
    elif isinstance(index, (list, np.ndarray)):
        return _valid_bounds(length, np.min(index)) and \
            _valid_bounds(length, np.max(index))
    elif isinstance(index, slice):
        if index.start is not None:
            start_valid = _valid_bounds(length, index.start)
        else:
            start_valid = True
        if index.stop is not None:
            stop_valid = _valid_bounds(length, index.stop - 1)
        else:
            stop_valid = True
        return start_valid and stop_valid


def _create_value(inst: Any, size: int) -> Union['Batch', np.ndarray]:
    if isinstance(inst, np.ndarray):
        return np.full((size, *inst.shape),
                       fill_value=None if inst.dtype == np.object else 0,
                       dtype=inst.dtype)
    elif isinstance(inst, torch.Tensor):
        return torch.full((size, *inst.shape),
                          fill_value=None if inst.dtype == np.object else 0,
                          device=inst.device,
                          dtype=inst.dtype)
    elif isinstance(inst, (dict, Batch)):
        zero_batch = Batch()
        for key, val in inst.items():
            zero_batch.__dict__[key] = _create_value(val, size)
        return zero_batch
    elif isinstance(inst, (np.generic, Number)):
        return _create_value(np.asarray(inst), size)
    else:  # fall back to np.object
        return np.array([None for _ in range(size)])


class Batch:
    """Tianshou provides :class:`~tianshou.data.Batch` as the internal data
    structure to pass any kind of data to other methods, for example, a
    collector gives a :class:`~tianshou.data.Batch` to policy for learning.
    Here is the usage:
    ::

        >>> import numpy as np
        >>> from tianshou.data import Batch
        >>> data = Batch(a=4, b=[5, 5], c='2312312')
        >>> # the list will automatically be converted to numpy array
        >>> data.b
        array([5, 5])
        >>> data.b = np.array([3, 4, 5])
        >>> print(data)
        Batch(
            a: 4,
            b: array([3, 4, 5]),
            c: '2312312',
        )

    In short, you can define a :class:`Batch` with any key-value pair. The
    current implementation of Tianshou typically use 7 reserved keys in
    :class:`~tianshou.data.Batch`:

    * ``obs`` the observation of step :math:`t` ;
    * ``act`` the action of step :math:`t` ;
    * ``rew`` the reward of step :math:`t` ;
    * ``done`` the done flag of step :math:`t` ;
    * ``obs_next`` the observation of step :math:`t+1` ;
    * ``info`` the info of step :math:`t` (in ``gym.Env``, the ``env.step()``\
        function return 4 arguments, and the last one is ``info``);
    * ``policy`` the data computed by policy in step :math:`t`;

    :class:`~tianshou.data.Batch` object can be initialized using wide variety
    of arguments, starting with the key/value pairs or dictionary, but also
    list and Numpy arrays of :class:`dict` or Batch instances. In which case,
    each element is considered as an individual sample and get stacked
    together:
    ::

        >>> data = Batch([{'a': {'b': [0.0, "info"]}}])
        >>> print(data[0])
        Batch(
            a: Batch(
                b: array(['0.0', 'info'], dtype='<U32'),
            ),
        )

    :class:`~tianshou.data.Batch` has the same API as a native Python
    :class:`dict`. In this regard, one can access to stored data using string
    key, or iterate over stored data:
    ::

        >>> data = Batch(a=4, b=[5, 5])
        >>> print(data["a"])
        4
        >>> for key, value in data.items():
        >>>     print(f"{key}: {value}")
        a: 4
        b: [5, 5]


    :class:`~tianshou.data.Batch` is also reproduce partially the Numpy API for
    arrays. It also supports the advanced slicing method, such as batch[:, i],
    if the index is valid. You can access or iterate over the individual
    samples, if any:
    ::

        >>> data = Batch(a=np.array([[0.0, 2.0], [1.0, 3.0]]), b=[[5, -5]])
        >>> print(data[0])
        Batch(
            a: array([0., 2.])
            b: array([ 5, -5]),
        )
        >>> for sample in data:
        >>>     print(sample.a)
        [0., 2.]
        [1., 3.]

        >>> print(data.shape)
        [1, 2]
        >>> data[:, 1] += 1
        >>> print(data)
        Batch(
            a: array([[0., 3.],
                      [1., 4.]]),
            b: array([[ 5, -4]]),
        )

    Similarly, one can also perform simple algebra on it, and stack, split or
    concatenate multiple instances:
    ::

        >>> data_1 = Batch(a=np.array([0.0, 2.0]), b=5)
        >>> data_2 = Batch(a=np.array([1.0, 3.0]), b=-5)
        >>> data = Batch.stack((data_1, data_2))
        >>> print(data)
        Batch(
            b: array([ 5, -5]),
            a: array([[0., 2.],
                      [1., 3.]]),
        )
        >>> print(np.mean(data))
        Batch(
            b: 0.0,
            a: array([0.5, 2.5]),
        )
        >>> data_split = list(data.split(1, False))
        >>> print(list(data.split(1, False)))
        [Batch(
            b: array([5]),
            a: array([[0., 2.]]),
        ), Batch(
            b: array([-5]),
            a: array([[1., 3.]]),
        )]
        >>> data_cat = Batch.cat(data_split)
        >>> print(data_cat)
        Batch(
            b: array([ 5, -5]),
            a: array([[0., 2.],
                      [1., 3.]]),
        )

    Note that stacking of inconsistent data is also supported. In which case,
    None is added in list or :class:`np.ndarray` of objects, 0 otherwise.
    ::

        >>> data_1 = Batch(a=np.array([0.0, 2.0]))
        >>> data_2 = Batch(a=np.array([1.0, 3.0]), b='done')
        >>> data = Batch.stack((data_1, data_2))
        >>> print(data)
        Batch(
            a: array([[0., 2.],
                      [1., 3.]]),
            b: array([None, 'done'], dtype=object),
        )

    Also with method empty (which will set to 0 or ``None`` (with np.object))
    ::

        >>> data.empty_()
        >>> print(data)
        Batch(
            a: array([[0., 0.],
                      [0., 0.]]),
            b: array([None, None], dtype=object),
        )
        >>> data = Batch(a=[False,  True], b={'c': [2., 'st'], 'd': [1., 0.]})
        >>> data[0] = Batch.empty(data[1])
        >>> data
        Batch(
            a: array([False,  True]),
            b: Batch(
                   c: array([0., 3.]),
                   d: array([0., 0.]),
               ),
        )

    :meth:`~tianshou.data.Batch.shape` and :meth:`~tianshou.data.Batch.__len__`
    methods are also provided to respectively get the shape and the length of
    a :class:`Batch` instance. It mimics the Numpy API for Numpy arrays, which
    means that getting the length of a scalar Batch raises an exception.
    ::

        >>> data = Batch(a=[5., 4.], b=np.zeros((2, 3, 4)))
        >>> data.shape
        [2]
        >>> len(data)
        2
        >>> data[0].shape
        []
        >>> len(data[0])
        TypeError: Object of type 'Batch' has no len()

    Convenience helpers are available to convert in-place the stored data into
    Numpy arrays or Torch tensors.

    Finally, note that :class:`~tianshou.data.Batch` instance are serializable
    and therefore Pickle compatible. This is especially important for
    distributed sampling.
    """

    def __init__(self,
                 batch_dict: Optional[Union[
                     dict, 'Batch', Tuple[Union[dict, 'Batch']],
                     List[Union[dict, 'Batch']], np.ndarray]] = None,
                 **kwargs) -> None:        
        if _is_batch_set(batch_dict):
            self.stack_(batch_dict)
        elif isinstance(batch_dict, (dict, Batch)):
            for k, v in batch_dict.items():                
                if isinstance(v, dict) or _is_batch_set(v):
                    self.__dict__[k] = Batch(v)
                else:
                    if isinstance(v, list):
                        v = np.array(v)
                    self.__dict__[k] = v
        if len(kwargs) > 0:
            self.__init__(kwargs)

    def __setattr__(self, key: str, value: Any):
        """self[key] = value"""
        if isinstance(value, list):
            if _is_batch_set(value):
                value = Batch(value)
            else:
                value = np.array(value)
        elif isinstance(value, dict):
            value = Batch(value)
        self.__dict__[key] = value

    def __getstate__(self):
        """Pickling interface. Only the actual data are serialized for both
        efficiency and simplicity.
        """
        state = {}
        for k, v in self.items():
            if isinstance(v, Batch):
                v = v.__getstate__()
            state[k] = v
        return state

    def __setstate__(self, state):
        """Unpickling interface. At this point, self is an empty Batch instance
        that has not been initialized, so it can safely be initialized by the
        pickle state.
        """
        self.__init__(**state)

    def __getitem__(self, index: Union[
            str, slice, int, np.integer, np.ndarray, List[int]]) -> 'Batch':
        """Return self[index]."""
        if isinstance(index, str):
            return self.__dict__[index]
        b = Batch()
        for k, v in self.items():
            if isinstance(v, Batch) and len(v.__dict__) == 0:
                b.__dict__[k] = Batch()
            else:
                b.__dict__[k] = v[index]
        return b

    def __setitem__(
            self,
            index: Union[str, slice, int, np.integer, np.ndarray, List[int]],
            value: Any) -> None:
        """Assign value to self[index]."""
        if isinstance(index, str):
            self.__dict__[index] = value
            return
        if not isinstance(value, (dict, Batch)):
            raise TypeError("Batch does not supported value type "
                            f"{type(value)} for item assignment.")
        if not set(value.keys()).issubset(self.__dict__.keys()):
            raise KeyError(
                "Creating keys is not supported by item assignment.")
        for key, val in self.items():
            try:
                self.__dict__[key][index] = value[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][index] = Batch()
                elif isinstance(val, np.ndarray) and \
                        val.dtype == np.integer:
                    # Fallback for np.array of integer,
                    # since neither None or nan is supported.
                    self.__dict__[key][index] = 0
                else:
                    self.__dict__[key][index] = None

    def __iadd__(self, other: Union['Batch', Number]):
        """Algebraic addition with another :class:`~tianshou.data.Batch`
        instance in-place."""
        if isinstance(other, Batch):
            for (k, r), v in zip(self.__dict__.items(),
                                 other.__dict__.values()):
                if r is None:
                    continue
                else:
                    self.__dict__[k] += v
            return self
        elif isinstance(other, Number):
            for k, r in self.items():
                if r is None:
                    continue
                else:
                    self.__dict__[k] += other
            return self
        else:
            raise TypeError("Only addition of Batch or number is supported.")

    def __add__(self, other: Union['Batch', Number]):
        """Algebraic addition with another :class:`~tianshou.data.Batch`
        instance out-of-place."""
        return copy.deepcopy(self).__iadd__(other)

    def __imul__(self, val: Number):
        """Algebraic multiplication with a scalar value in-place."""
        assert isinstance(val, Number), \
            "Only multiplication by a number is supported."
        for k in self.__dict__.keys():
            self.__dict__[k] *= val
        return self

    def __mul__(self, val: Number):
        """Algebraic multiplication with a scalar value out-of-place."""
        return copy.deepcopy(self).__imul__(val)

    def __itruediv__(self, val: Number):
        """Algebraic division wibyth a scalar value in-place."""
        assert isinstance(val, Number), \
            "Only division by a number is supported."
        for k in self.__dict__.keys():
            self.__dict__[k] /= val
        return self

    def __truediv__(self, val: Number):
        """Algebraic division wibyth a scalar value out-of-place."""
        return copy.deepcopy(self).__itruediv__(val)

    def __repr__(self) -> str:
        """Return str(self)."""
        s = self.__class__.__name__ + '(\n'
        flag = False
        for k, v in self.items():
            rpl = '\n' + ' ' * (6 + len(k))
            obj = pprint.pformat(v).replace('\n', rpl)
            s += f'    {k}: {obj},\n'
            flag = True
        if flag:
            s += ')'
        else:
            s = self.__class__.__name__ + '()'
        return s

    def keys(self) -> List[str]:
        """Return self.keys()."""
        return self.__dict__.keys()

    def values(self) -> List[Any]:
        """Return self.values()."""
        return self.__dict__.values()

    def items(self) -> List[Tuple[str, Any]]:
        """Return self.items()."""
        return self.__dict__.items()

    def get(self, k: str, d: Optional[Any] = None) -> Union['Batch', Any]:
        """Return self[k] if k in self else d. d defaults to None."""
        return self.__dict__.get(k, d)

    def to_numpy(self) -> None:
        """Change all torch.Tensor to numpy.ndarray. This is an in-place
        operation.
        """
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                self.__dict__[k] = v.detach().cpu().numpy()
            elif isinstance(v, Batch):
                v.to_numpy()

    def to_torch(self,
                 dtype: Optional[torch.dtype] = None,
                 device: Union[str, int, torch.device] = 'cpu'
                 ) -> None:
        """Change all numpy.ndarray to torch.Tensor. This is an in-place
        operation.
        """
        if not isinstance(device, torch.device):
            device = torch.device(device)

        for k, v in self.items():
            if isinstance(v, (np.generic, np.ndarray)):
                v = torch.from_numpy(v).to(device)
                if dtype is not None:
                    v = v.type(dtype)
                self.__dict__[k] = v
            if isinstance(v, torch.Tensor):
                if dtype is not None and v.dtype != dtype:
                    must_update_tensor = True
                elif v.device.type != device.type:
                    must_update_tensor = True
                elif device.index is not None and \
                        device.index != v.device.index:
                    must_update_tensor = True
                else:
                    must_update_tensor = False
                if must_update_tensor:
                    if dtype is not None:
                        v = v.type(dtype)
                    self.__dict__[k] = v.to(device)
            elif isinstance(v, Batch):
                v.to_torch(dtype, device)

    def append(self, batch: 'Batch') -> None:
        warnings.warn('Method :meth:`~tianshou.data.Batch.append` will be '
                      'removed soon, please use '
                      ':meth:`~tianshou.data.Batch.cat`')
        return self.cat_(batch)

    def cat_(self, batch: 'Batch') -> None:
        """Concatenate a :class:`~tianshou.data.Batch` object into current
        batch.
        """
        assert isinstance(batch, Batch), \
            'Only Batch is allowed to be concatenated in-place!'
        for k, v in batch.items():
            if v is None:
                continue
            if not hasattr(self, k) or self.__dict__[k] is None:
                self.__dict__[k] = copy.deepcopy(v)
            elif isinstance(v, np.ndarray) and v.ndim > 0:
                self.__dict__[k] = np.concatenate([self.__dict__[k], v])
            elif isinstance(v, torch.Tensor):
                self.__dict__[k] = torch.cat([self.__dict__[k], v])
            elif isinstance(v, Batch):
                self.__dict__[k].cat_(v)
            else:
                s = 'No support for method "cat" with type '\
                    f'{type(v)} in class Batch.'
                raise TypeError(s)

    @staticmethod
    def cat(batches: List['Batch']) -> 'Batch':
        """Concatenate a :class:`~tianshou.data.Batch` object into a single
        new batch.
        """
        batch = Batch()
        for batch_ in batches:
            batch.cat_(batch_)
        return batch

    def stack_(self,
               batches: List[Union[dict, 'Batch']],
               axis: int = 0) -> None:
        """Stack a :class:`~tianshou.data.Batch` object i into current batch.
        """
        if len(self.__dict__) > 0:
            batches = [self] + list(batches)
        keys_map = list(map(lambda e: set(e.keys()), batches))
        keys_shared = set.intersection(*keys_map)
        values_shared = [
            [e[k] for e in batches] for k in keys_shared]
        for k, v in zip(keys_shared, values_shared):
            if isinstance(v[0], (dict, Batch)):
                self.__dict__[k] = Batch.stack(v, axis)
            elif isinstance(v[0], torch.Tensor):
                self.__dict__[k] = torch.stack(v, axis)
            else:
                self.__dict__[k] = np.stack(v, axis)
        keys_partial = reduce(set.symmetric_difference, keys_map)
        for k in keys_partial:
            for i, e in enumerate(batches):
                val = e.get(k, None)
                if val is not None:
                    try:
                        self.__dict__[k][i] = val
                    except KeyError:
                        self.__dict__[k] = \
                            _create_value(val, len(batches))
                        self.__dict__[k][i] = val

    @staticmethod
    def stack(batches: List['Batch'], axis: int = 0) -> 'Batch':
        """Stack a :class:`~tianshou.data.Batch` object into a single new
        batch.
        """
        batch = Batch()
        batch.stack_(batches, axis)
        return batch

    def empty_(self) -> 'Batch':
        """Return an empty a :class:`~tianshou.data.Batch` object with 0 or
        ``None`` filled.
        """
        for k, v in self.items():
            if v is None:
                continue
            if isinstance(v, Batch):
                self.__dict__[k].empty_()
            elif isinstance(v, np.ndarray) and v.dtype == np.object:
                self.__dict__[k].fill(None)
            elif isinstance(v, torch.Tensor):  # cannot apply fill_ directly
                self.__dict__[k] = torch.zeros_like(self.__dict__[k])
            else:  # np
                self.__dict__[k] *= 0
                if hasattr(v, 'dtype') and v.dtype.kind in 'fc':
                    self.__dict__[k] = np.nan_to_num(self.__dict__[k])
        return self

    @staticmethod
    def empty(batch: 'Batch') -> 'Batch':
        """Return an empty :class:`~tianshou.data.Batch` object with 0 or
        ``None`` filled, the shape is the same as the given
        :class:`~tianshou.data.Batch`.
        """
        batch = Batch(**batch)
        batch.empty_()
        return batch

    def __len__(self) -> int:
        """Return len(self)."""
        r = []
        for v in self.__dict__.values():
            if isinstance(v, Batch) and len(v.__dict__) == 0:
                continue
            elif hasattr(v, '__len__') and (not isinstance(
                    v, (np.ndarray, torch.Tensor)) or v.ndim > 0):
                r.append(len(v))
            else:
                raise TypeError("Object of type 'Batch' has no len()")
        if len(r) == 0:
            raise TypeError("Object of type 'Batch' has no len()")
        return min(r)

    @property
    def shape(self) -> List[int]:
        """Return self.shape."""
        if len(self.__dict__.keys()) == 0:
            return []
        else:
            data_shape = []
            for v in self.__dict__.values():
                try:
                    data_shape.append(v.shape)
                except AttributeError:
                    raise TypeError("No support for 'shape' method with "
                                    f"type {type(v)} in class Batch.")
            return list(map(min, zip(*data_shape))) if len(data_shape) > 1 \
                else data_shape[0]

    def split(self, size: Optional[int] = None,
              shuffle: bool = True) -> Iterator['Batch']:
        """Split whole data into multiple small batch.

        :param int size: if it is ``None``, it does not split the data batch;
            otherwise it will divide the data batch with the given size.
            Default to ``None``.
        :param bool shuffle: randomly shuffle the entire data batch if it is
            ``True``, otherwise remain in the same. Default to ``True``.
        """
        length = len(self)
        if size is None:
            size = length
        if shuffle:
            indices = np.random.permutation(length)
        else:
            indices = np.arange(length)
        for idx in np.arange(0, length, size):
            yield self[indices[idx:(idx + size)]]
