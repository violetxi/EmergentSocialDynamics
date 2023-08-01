"""Core modules for all networks with different architectures
"""
from typeguard import typechecked
from typing import Dict, Optional, Type, Union, Sequence


import torch
from torch import nn
from torchrl.modules.models import MLP
from torchrl.data.utils import DEVICE_TYPING


@typechecked
class MLPModule(nn.Module):
    def __init__(self, kwargs: Dict) -> None:
        """MLP module for world model
        kwargs: should be instantiated in config file
        """
        super().__init__()        
        self.model = MLP(**kwargs)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@typechecked
class DeconvNet(nn.Module):
    """
    Deconvolutional network
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            num_cells: Sequence,
            depth: Optional[int] = None,
            kernel_sizes: Union[Sequence[Union[int, Sequence[int]]], int] = 3,
            strides: Union[Sequence, int] = 1,
            paddings: Union[Sequence, int] = 0,
            activation_class: Type[nn.Module] = nn.ReLU,            
            norm_kwargs: Optional[dict] = None,
            bias_last_layer: bool = True,
            ) -> None:
        super(DeconvNet, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_cells = num_cells
        if depth is None:
            self.depth = len(num_cells)
        else:
            self.depth = depth        
        assert self.depth == len(num_cells), \
            "depth and num_cells should have the same length"
        
        self.kernel_sizes = self._get_net_args(kernel_sizes)
        self.strides = self._get_net_args(strides)
        self.paddings = self._get_net_args(paddings)
        self.activation_class = activation_class
        self.norm_kwargs = norm_kwargs      
        self.bias_last_layer = bias_last_layer
        layers = self._build_layers()        
        self.model = nn.Sequential(*layers)


    def _get_net_args(self, arg):
        if isinstance(arg, int):
            return [arg] * self.depth
        elif isinstance(arg, Sequence): 
            assert len(arg) == self.depth, \
                "arg should be an integer or a sequence with the same length as depth"
            return arg

    
    def _build_layers(self):
        layers = []
        for i in range(self.depth):
            if i == 0:
                in_channels = self.in_features
            else:
                in_channels = self.num_cells[i-1]
            out_channels = self.num_cells[i]
            kernel_size = self.kernel_sizes[i]
            stride = self.strides[i]
            padding = self.paddings[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False
                )
            )
            if self.norm_kwargs is not None:    
                layers.append(
                    nn.BatchNorm2d(
                        num_features=out_channels,
                        **self.norm_kwargs
                    )
                )
            layers.append(
                self.activation_class()
            )
        if self.bias_last_layer:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_cells[-1],
                    out_channels=self.out_features,
                    kernel_size=self.kernel_sizes[-1],
                    stride=self.strides[-1],
                    padding=self.paddings[-1],
                    bias=True
                )
            )
        else:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_cells[-1],
                    out_channels=self.in_features,
                    kernel_size=self.kernel_sizes[-1],
                    stride=self.strides[-1],
                    padding=self.paddings[-1],
                    bias=False
                )
            )
        layers.append(nn.Sigmoid())
        return layers
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
        