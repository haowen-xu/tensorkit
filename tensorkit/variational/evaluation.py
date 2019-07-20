from .. import tensor as T
from ..tensor import typing as Z
from .utils import _require_multi_samples

__all__ = ['importance_sampling_log_likelihood']


def importance_sampling_log_likelihood(log_joint: Z.TensorLike,
                                       latent_log_prob: Z.TensorLike,
                                       axis: Z.AxisOrAxes,
                                       keepdims=False):
    _require_multi_samples(axis, 'importance sampling log-likelihood')
    log_joint = T.as_tensor(log_joint)
    latent_log_prob = T.as_tensor(latent_log_prob)
    log_p = T.log_mean_exp(
        log_joint - latent_log_prob, axis=axis, keepdims=keepdims)
    return log_p
