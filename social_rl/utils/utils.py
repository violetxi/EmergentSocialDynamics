import os
import json
import argparse
import numpy as np
from typeguard import typechecked
from typing import Any, Dict, Optional

import torch
import tensordict as td
from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiDiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.utils import numpy_to_torch_dtype_dict

from social_rl.config.base_config import BaseConfig
from social_rl.agents.base_agent import BaseAgent


""" 
Utility functions for the CleanUP and Harvest environments
adapted from: https://github.com/eugenevinitsky/sequential_social_dilemma_games/blob/bd1176bb3e7f39e947f4809cf1bbcf6a7df4f2a6/utility_funcs.py#L60
"""
@typechecked
def return_view(
    grid: np.ndarray, 
    pos: np.ndarray, 
    row_size: int, 
    col_size: int
    ) -> np.ndarray:
    """Given a map grid, position and view window, returns correct map part

    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros

    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: np.ndarray
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension

    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge, top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size : x + col_size + 1, y - row_size : y + row_size + 1]
    return view


@typechecked
# addapted from: https://github.com/eugenevinitsky/sequential_social_dilemma_games/blob/bd1176bb3e7f39e947f4809cf1bbcf6a7df4f2a6/utility_funcs.py#L93C1-L111C6
def pad_if_needed(
    left_edge: int,
    right_edge: int,
    top_edge: int,
    bot_edge: int,
    matrix: np.ndarray
    ) -> tuple[np.ndarray, int, int]:
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return (
        pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0),
        left_pad,
        top_pad,
    )


@typechecked
def pad_matrix(
    left_pad: int, 
    right_pad: int, 
    top_pad: int,    
    bot_pad: int,
    matrix: np.ndarray,
    const_val: Optional[int] =1
    ) -> np.ndarray:
    pad_mat = np.pad(
        matrix,
        ((left_pad, right_pad), (top_pad, bot_pad)),
        "constant",
        constant_values=(const_val, const_val),
    )
    return pad_mat


@typechecked
def load_config_from_path(path: str, args: argparse.Namespace) -> BaseConfig:
    """
    Load config from path, assuming config is a python file with class Config,  
    and all parameters are defined as class attributes
    """
    base_lib_path = 'social_rl.config.'
    config_name = os.path.basename(path).split('.')[0]
    config_path = base_lib_path + config_name
    config = __import__(config_path, fromlist=['*'])
    return config.Config(args)
    
    

@typechecked
def ensure_dir(directory: str) -> None:
    if not os.path.exists(directory):
        print(f"Creating directory {directory}")
        os.makedirs(directory)



@typechecked
def gym_to_torchrl_spec_transform(
    spec: Any,
    dtype: Optional[torch.dtype] = None, 
    device: torch.device = "cpu", 
    categorical_action_encoding: bool = False,
    which_gym: str = 'gymnasium',
) -> TensorSpec:
    """Maps the gym specs to the TorchRL specs.
    Adapted from 
    https://github.com/pytorch/rl/blob/771ef814f98b30dbe0e1b7acb2625a0bf16a1e08/torchrl/envs/libs/gym.py#L181

    By convention, 'state' keys of Dict specs will be renamed "observation" to match the
    default TorchRL keys.

    """
    if which_gym == 'gymnasium':
        import gymnasium as gym
    else:
        import gym
    
    if isinstance(spec, gym.spaces.tuple.Tuple):
        raise NotImplementedError("gym.spaces.tuple.Tuple mapping not yet implemented")
    if isinstance(spec, gym.spaces.discrete.Discrete):
        action_space_cls = (
            DiscreteTensorSpec
            if categorical_action_encoding
            else OneHotDiscreteTensorSpec
        )
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )        
        return action_space_cls(
            spec.n, device=device, dtype=dtype
        )
    elif isinstance(spec, gym.spaces.multi_binary.MultiBinary):
        return BinaryDiscreteTensorSpec(
            spec.n, device=device, dtype=numpy_to_torch_dtype_dict[spec.dtype]
        )
    elif isinstance(spec, gym.spaces.multi_discrete.MultiDiscrete):
        dtype = (
            numpy_to_torch_dtype_dict[spec.dtype]
            if categorical_action_encoding
            else torch.long
        )
        return (
            MultiDiscreteTensorSpec(
            spec.nvec, device=device, dtype=dtype
        )
            if categorical_action_encoding
            else MultiOneHotDiscreteTensorSpec(
            spec.nvec, device=device, dtype=dtype)
        )
    elif isinstance(spec, gym.spaces.Box):        
        shape = spec.shape
        if not len(shape):
            shape = torch.Size([1])
        if dtype is None:
            dtype = numpy_to_torch_dtype_dict[spec.dtype]
        low = torch.tensor(spec.low, device=device, dtype=dtype)
        high = torch.tensor(spec.high, device=device, dtype=dtype)
        is_unbounded = low.isinf().all() and high.isinf().all()        
        return (
            UnboundedContinuousTensorSpec(shape, device=device, dtype=dtype)
            if is_unbounded
            else BoundedTensorSpec(
                low,
                high,
                shape,
                dtype=dtype,
                device=device,
            )
        )
    elif isinstance(spec, (Dict,)):
        spec_out = {}
        for k in spec.keys():
            key = k
            if k == "state" and "observation" not in spec.keys():
                # we rename "state" in "observation" as "observation" is the conventional name
                # for single observation in torchrl.
                # naming it 'state' will result in envs that have a different name for the state vector
                # when queried with and without pixels
                key = "observation"
            spec_out[key] = gym_to_torchrl_spec_transform(
                spec[k],
                device=device,
                categorical_action_encoding=categorical_action_encoding,
            )
        return CompositeSpec(**spec_out)
    elif isinstance(spec, gym.spaces.dict.Dict):
        return gym_to_torchrl_spec_transform(
            spec.spaces,
            device=device,
            categorical_action_encoding=categorical_action_encoding,
        )
    else:        
        raise NotImplementedError(
            f"spec of type {type(spec).__name__} is currently unaccounted for"
        )
    

@typechecked
def save_args(args: argparse.Namespace, file_path: str):
    """Save args to a json file for reproducibility
    """
    with open(file_path, 'w') as f:
        json.dump(vars(args), f)


@typechecked
def load_args(file_path: str) -> argparse.Namespace:
    with open(file_path, 'r') as f:
        args = json.load(f)
    return argparse.Namespace(**args)


@typechecked
def convert_tensordict_to_tensor(
    tensordict: td.TensorDict, 
    td_type: str,
    action_dim: Optional[int] = None,
    ) -> torch.Tensor:
    """Convert a tensordict to a torch tensor
    Args:
        tensordict (td.TensorDict): a tensordict
        td_type (str): the type of tensordict (obs, act, rew etc)
        action_dim (Optional[int], optional): the action dimension to help create 
        one-hot encoding for actions
    Returns:
        tensor_out (torch.Tensor): a tensor of the input (N, B, D)
    """        
    if isinstance(tensordict[list(tensordict.keys())[0]], td.MemmapTensor):
        # if input contains memmap, convert to tensor
        # as_tensor can only be called on tensors on cpu
        tensordict = tensordict.cpu()
        for k, v in tensordict.items():
            tensordict[k] = v.as_tensor()

    if td_type == "obs":
        # if input is unbatched, add batch dimension        
        max_size = max(tensor.shape[-1] for tensor in tensordict.values())               
        tensor_out = torch.stack(
            [torch.nn.functional.pad(t, (0, max_size - t.shape[-1])) 
             if t.shape[-1] < max_size else t for t in tensordict.values()]
        )
        if len( tensor_out.shape) == 2:
            # add batch dimension if input is unbatched
            tensor_out = tensor_out.unsqueeze(1)
    elif td_type == "action":
        # assert action_dim is not None, \
        #     "action_dim must be provided for action conversion"
        tensor_out = torch.stack(list(tensordict.values()))    # (num_agents, )
        tensor_out = torch.nn.functional.one_hot(tensor_out, action_dim)    # (num_agents, action_dim)
        if len(tensor_out.shape) == 2:
            # add batch dimension if input is unbatched
            tensor_out = tensor_out.unsqueeze(1)
    elif td_type == "reward":
        reward_list = list(tensordict['next']['reward'].values())    # (num_agents, 1)
        if len(tensor_out.shape) == 2:
            # add batch dimension if input is unbatched
            tensor_out = torch.stack(reward_list).unsqueeze(1)    # (1, num_agents, 1)
    else:
        raise NotImplementedError(f"Conversion for td_type {td_type} not implemented")
        
    return tensor_out


