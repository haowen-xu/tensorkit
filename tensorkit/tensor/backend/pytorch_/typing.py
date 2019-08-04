from typing import *

import numpy as np
import torch

__all__ = [
    'Tensor', 'Variable', 'DType', 'Shape',
    'Number', 'TensorLike', 'DTypeLike', 'ShapeLike',
    'AxisOrAxes', 'AxisOrAxes'
]

# true types
Tensor = torch.Tensor
Variable = torch.Tensor
DType = torch.dtype
Shape = torch.Size


# type annotations
Number = Union[int, float]
TensorLike = Union[Tensor, Variable, 'TensorWrapper', np.ndarray, Number]
DTypeLike = Union[str, np.dtype, DType, Type[int], Type[float]]
ShapeLike = Sequence[int]
AxisOrAxes = Union[int, Tuple[int, ...], List[int]]
