from typing import *

from ..tensor import *

__all__ = [
    'sgvb_estimator', 'iwae_estimator',
]


@jit
def sgvb_estimator(values: Tensor,
                   axes: Optional[List[int]] = None,
                   keepdims: bool = False,
                   neg_grad: bool = False) -> Tensor:
    estimator = values
    if axes is not None:
        estimator = reduce_mean(estimator, axes=axes, keepdims=keepdims)
    if neg_grad:
        estimator = -estimator
    return estimator


@jit
def iwae_estimator(log_values: Tensor,
                   axes: Optional[List[int]] = None,
                   keepdims: bool = False,
                   neg_grad: bool = False) -> Tensor:
    if axes is None or len(axes) == 0:
        raise ValueError(
            '`iwae_estimator` requires to take multiple samples of the latent '
            'variables, thus the `axes` argument must be specified'
        )
    estimator = log_mean_exp(log_values, axes=axes, keepdims=keepdims)
    if neg_grad:
        estimator = -estimator
    return estimator
