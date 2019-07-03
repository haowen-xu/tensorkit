from typing import *

from . import backend as B
from ..utils import validate_int_tuple_arg
from .utils import default_impl

__all__ = ['log_mean_exp', 'add_n']


@default_impl
def log_mean_exp(x: B.TensorLike,
                 axis: Optional[B.AxisOrAxes] = None,
                 keepdims: bool = False) -> B.Tensor:
    axis = validate_int_tuple_arg('axis', axis, nullable=True)
    x = B.as_tensor(x)
    x_max_keepdims = B.max(x, axis=axis, keepdims=True)
    if not keepdims:
        x_max = B.squeeze(x_max_keepdims, axis=axis)
    else:
        x_max = x_max_keepdims
    mean_exp = B.mean(
        B.exp(x - x_max_keepdims), axis=axis, keepdims=keepdims)
    return x_max + B.log(mean_exp)


@default_impl
def add_n(tensors: Iterable[B.TensorLike]) -> B.Tensor:
    tensors = tuple(map(B.as_tensor, tensors))
    if not tensors:
        raise ValueError('`tensors` must not be emtpy.')
    ret = tensors[0]
    for t in tensors[1:]:
        ret += t
    return ret
