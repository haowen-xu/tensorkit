from typing import *

import numpy as np


__all__ = ['TensorOrData']

TensorOrData = Union['Tensor', 'StochasticTensor', np.ndarray, float, int]
"""Types that can be casted into `T.Tensor` via `T.as_tensor`."""

from .tensor.core import Tensor
from .stochastic import StochasticTensor
