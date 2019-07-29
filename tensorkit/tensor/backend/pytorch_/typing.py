from typing import *

import numpy as np
import torch

__all__ = [
    'Tensor', 'Variable', 'DType', 'Shape',
    'TensorLike', 'DTypeLike', 'ShapeLike', 'AxisOrAxes', 'AxisOrAxes'
]

# true types
Tensor = torch.Tensor
Variable = torch.Tensor
DType = torch.dtype
Shape = torch.Size


# type annotations
TensorLike = Union[Tensor, Variable, 'TensorWrapper', np.ndarray]
DTypeLike = Union[str, np.dtype, DType, Type[int], Type[float]]
ShapeLike = Sequence[int]
AxisOrAxes = Union[int, Tuple[int, ...], List[int]]
