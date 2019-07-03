from typing import *

from .. import tensor as T
from .utils import _require_multi_samples

__all__ = ['elbo_objective', 'monte_carlo_objective']


def elbo_objective(log_joint: T.TensorLike,
                   latent_log_prob: T.TensorLike,
                   axis: Optional[T.AxisOrAxes] = None,
                   keepdims: bool = False) -> T.Tensor:
    log_joint = T.as_tensor(log_joint)
    latent_log_prob = T.as_tensor(latent_log_prob)
    objective = log_joint - latent_log_prob
    if axis is not None:
        objective = T.mean(objective, axis=axis, keepdims=keepdims)
    return objective


def monte_carlo_objective(log_joint: T.TensorLike,
                          latent_log_prob: T.TensorLike,
                          axis: Optional[T.AxisOrAxes] = None,
                          keepdims: bool = False) -> T.Tensor:
    _require_multi_samples(axis, 'monte carlo objective')
    log_joint = T.as_tensor(log_joint)
    latent_log_prob = T.as_tensor(latent_log_prob)
    likelihood = log_joint - latent_log_prob
    objective = T.log_mean_exp(likelihood, axis=axis, keepdims=keepdims)
    return objective
