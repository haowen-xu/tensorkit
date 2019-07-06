from typing import *

from .. import tensor as T
from .utils import _require_multi_samples

__all__ = [
    'sgvb_estimator', 'iwae_estimator',
]


@T.jit_function
def _sgvb_estimator(values: T.Tensor,
                    axis: Optional[T.AxisOrAxes],
                    keepdims: bool,
                    neg_grad: bool):
    estimator = values
    if axis is not None:
        estimator = T.reduce_mean(estimator, axis=axis, keepdims=keepdims)
    if neg_grad:
        estimator = -estimator
    return estimator


def sgvb_estimator(values: T.TensorLike,
                   axis: Optional[T.AxisOrAxes] = None,
                   keepdims: bool = False,
                   neg_grad: bool = False) -> T.Tensor:
    return _sgvb_estimator(T.as_tensor(values), axis=axis,
                           keepdims=keepdims, neg_grad=neg_grad)


@T.jit_function
def _iwae_estimator(log_values: T.TensorLike,
                   axis: T.AxisOrAxes,
                   keepdims: bool,
                   neg_grad: bool):
    estimator = T.log_mean_exp(log_values, axis=axis, keepdims=keepdims)
    if neg_grad:
        estimator = -estimator
    return estimator


def iwae_estimator(log_values: T.TensorLike,
                   axis: T.AxisOrAxes,
                   keepdims: bool = False,
                   neg_grad: bool = False) -> T.Tensor:
    _require_multi_samples('iwae estimator', axis)
    return _iwae_estimator(T.as_tensor(log_values), axis=axis,
                           keepdims=keepdims, neg_grad=neg_grad)
