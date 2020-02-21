from typing import *

from ..tensor import jit, Tensor, reduce_mean, log_mean_exp
from .utils import apply_reduction

__all__ = ['elbo_objective', 'monte_carlo_objective']


@jit
def elbo_objective(log_joint: Tensor,
                   latent_log_joint: Tensor,
                   axis: Optional[List[int]] = None,
                   keepdims: bool = False,
                   reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                   ) -> Tensor:
    """
    Derive the ELBO objective.

    .. math::

        \\mathbb{E}_{\\mathbf{z} \\sim q_{\\phi}(\\mathbf{z}|\\mathbf{x})}\\big[
             \\log p_{\\theta}(\\mathbf{x},\\mathbf{z}) - \\log q_{\\phi}(\\mathbf{z}|\\mathbf{x})
        \\big]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_joint: :math:`\\log q(\\mathbf{z}|\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        reduction: "sum" to return the sum of the elbo,
            "mean" to return the mean of the elbo, or "none" to
            return the original element-wise elbo.
        keepdims: When `axis` is specified, whether or not to keep
            the reduced axis?  Defaults to :obj:`False`.

    Returns:
        The ELBO objective.  Do not use it for training.
    """
    objective = log_joint - latent_log_joint
    if axis is not None:
        objective = reduce_mean(objective, axis=axis, keepdims=keepdims)
    objective = apply_reduction(objective, reduction)
    return objective


@jit
def monte_carlo_objective(log_joint: Tensor,
                          latent_log_joint: Tensor,
                          axis: Optional[List[int]] = None,
                          keepdims: bool = False,
                          reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                          ) -> Tensor:
    """
    Derive the Monte-Carlo objective.

    .. math::

        \\mathcal{L}_{K}(\\mathbf{x};\\theta,\\phi) =
            \\mathbb{E}_{\\mathbf{z}^{(1:K)} \\sim q_{\\phi}(\\mathbf{z}|\\mathbf{x})}\\Bigg[
                \\log \\frac{1}{K} \\sum_{k=1}^K {
                    \\frac{p_{\\theta}(\\mathbf{x},\\mathbf{z}^{(k)})}
                         {q_{\\phi}(\\mathbf{z}^{(k)}|\\mathbf{x})}
                }
            \\Bigg]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_joint: :math:`\\log q(\\mathbf{z}|\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        reduction: "sum" to return the sum of the monte-carlo objective,
            "mean" to return the mean of the monte-carlo objective, or "none" to
            return the original element-wise monte-carlo objective.
        keepdims: When `axis` is specified, whether or not to keep
            the reduced axis?  Defaults to :obj:`False`.

    Returns:
        The Monte Carlo objective.  Do not use it for training.
    """
    if axis is None or len(axis) == 0:
        raise ValueError(
            '`monte_carlo_objective` requires to take multiple samples of the '
            'latent variables, thus the `axis` argument must be specified'
        )

    likelihood = log_joint - latent_log_joint
    objective = log_mean_exp(likelihood, axis=axis, keepdims=keepdims)
    objective = apply_reduction(objective, reduction)
    return objective
