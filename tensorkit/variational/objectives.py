from typing import *

from ..tensor import *

__all__ = ['elbo_objective', 'monte_carlo_objective']


@jit
def elbo_objective(log_joint: Tensor,
                   latent_log_prob: Tensor,
                   axes: Optional[List[int]] = None,
                   keepdims: bool = False) -> Tensor:
    objective = log_joint - latent_log_prob
    if axes is not None:
        objective = reduce_mean(objective, axes=axes, keepdims=keepdims)
    return objective


@jit
def monte_carlo_objective(log_joint: Tensor,
                          latent_log_prob: Tensor,
                          axes: Optional[List[int]] = None,
                          keepdims: bool = False) -> Tensor:
    if axes is None or len(axes) == 0:
        raise ValueError(
            '`monte_carlo_objective` requires to take multiple samples of the '
            'latent variables, thus the `axes` argument must be specified'
        )

    likelihood = log_joint - latent_log_prob
    objective = log_mean_exp(likelihood, axes=axes, keepdims=keepdims)
    return objective
