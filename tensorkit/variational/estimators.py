from typing import *

from .. import tensor as T
from .utils import _require_multi_samples

__all__ = [
    'sgvb_estimator', 'iwae_estimator',
]


def sgvb_estimator(values: T.TensorLike,
                   axis: Optional[T.AxisOrAxes] = None,
                   keepdims: bool = False,
                   neg_grad: bool = False) -> T.Tensor:
    values = T.as_tensor(values)
    estimator = values
    if axis is not None:
        estimator = T.mean(estimator, axis=axis, keepdims=keepdims)
    if neg_grad:
        estimator = -estimator
    return estimator


def iwae_estimator(log_values: T.TensorLike,
                   axis: T.AxisOrAxes,
                   keepdims: bool = False,
                   neg_grad: bool = False) -> T.Tensor:
    _require_multi_samples('iwae estimator', axis)
    log_values = T.as_tensor(log_values)
    estimator = T.log_mean_exp(log_values, axis=axis, keepdims=keepdims)
    if neg_grad:
        estimator = -estimator
    return estimator
