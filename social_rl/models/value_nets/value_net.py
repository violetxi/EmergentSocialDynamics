from typeguard import typechecked
from typing import Dict, Any, Callable, Optional, Tuple, Union, List

from social_rl.models.cores import MLPModule



@typechecked
class ValueNet(nn.Module)