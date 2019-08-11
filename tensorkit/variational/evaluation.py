from typing import *

from ..tensor import *

__all__ = ['importance_sampling_log_likelihood']


@jit
def importance_sampling_log_likelihood(log_joint: Tensor,
                                       latent_log_prob: Tensor,
                                       axes: Optional[List[int]] = None,
                                       keepdims: bool = False):
    if axes is None or len(axes) == 0:
        raise ValueError(
            '`importance_sampling_log_likelihood` requires to take '
            'multiple samples of the latent variables, '
            'thus the `axes` argument must be specified'
        )
    log_p = log_mean_exp(
        log_joint - latent_log_prob, axes=axes, keepdims=keepdims)
    return log_p
