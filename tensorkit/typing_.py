from enum import Enum
from typing import *

import numpy as np

__all__ = [
    # enums
    'WeightNormMode', 'PaddingMode', 'ActNormScaleType',

    # tensor types
    'TensorOrData',

    # layer argument types
    'LayerFactory', 'LayerOrLayerFactory',
    'NormalizerFactory', 'NormalizerOrNormalizerFactory',
    'WeightNormArgType', 'TensorInitArgType', 'DataInitArgType', 'PaddingArgType',
]


# enums
class WeightNormMode(str, Enum):
    NONE = 'none'
    """No weight norm."""

    FULL = 'full'
    """Full weight norm, i.e., using `g * v / norm(v)` as the new weight."""

    NO_SCALE = 'no_scale'
    """Weight norm without scale, i.e., using `v / norm(v)`"""


class PaddingMode(str, Enum):
    NONE = 'none'
    """No padding, i.e., `left_padding = right_padding = 0`."""

    HALF = 'half'
    """
    Padding size is half the size of the kernel at each side, i.e.,
    `left_padding = right_padding = kernel_size // 2`.

    The kernel size for each spatial dimension must be odd, in order to
    use this mode.  Also, when `stride` == 1, this mode will cause the
    output shape to be the same as the input shape. 
    """

    FULL = 'full'
    """Padding size is `kernel_size - 1` at each side."""

    DEFAULT = NONE
    """The default padding mode is "none"."""


class ActNormScaleType(str, Enum):
    """Scale type of :class:`tk.layers.ActNorm` and :class:`tk.flows.ActNorm`."""

    EXP = 'exp'
    LINEAR = 'linear'


# tensor types
TensorOrData = Union['Tensor', 'StochasticTensor', np.ndarray, float, int]
"""Types that can be casted into `T.Tensor` via `T.as_tensor`."""


# layer types
LayerFactory = Union[Type['Module'], Callable[[], 'Module']]
LayerOrLayerFactory = Union['Module', LayerFactory]
NormalizerFactory = Union[Type['Module'], Callable[[int], 'Module']]
NormalizerOrNormalizerFactory = Union['Module', NormalizerFactory]
WeightNormArgType = Union[bool, str, WeightNormMode]
TensorInitArgType = Union[int, float, np.ndarray, Callable[..., None]]
DataInitArgType = Union[
    Type['DataDependentInitializer'], 'DataDependentInitializer',
    Callable[..., 'DataDependentInitializer'],
]
PaddingArgType = Union[int, Sequence[int], str, PaddingMode]


# import these types for type annotation
from .backend.core import Module
from .backend.init import DataDependentInitializer
from .stochastic import StochasticTensor
