from typing import *

from ..tensor import jit, Tensor, reduce_mean, log_mean_exp

__all__ = ['elbo_objective', 'monte_carlo_objective']


@jit
def elbo_objective(log_joint: Tensor,
                   latent_log_joint: Tensor,
                   axes: Optional[List[int]] = None,
                   keepdims: bool = False) -> Tensor:
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
        axes: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims: When `axes` is specified, whether or not to keep
            the reduced axes?  Defaults to :obj:`False`.

    Returns:
        The ELBO objective.  Do not use it for training.
    """
    objective = log_joint - latent_log_joint
    if axes is not None:
        objective = reduce_mean(objective, axes=axes, keepdims=keepdims)
    return objective


@jit
def monte_carlo_objective(log_joint: Tensor,
                          latent_log_joint: Tensor,
                          axes: Optional[List[int]] = None,
                          keepdims: bool = False) -> Tensor:
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
        axes: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims: When `axes` is specified, whether or not to keep
            the reduced axes?  Defaults to :obj:`False`.

    Returns:
        The Monte Carlo objective.  Do not use it for training.
    """
    if axes is None or len(axes) == 0:
        raise ValueError(
            '`monte_carlo_objective` requires to take multiple samples of the '
            'latent variables, thus the `axes` argument must be specified'
        )

    likelihood = log_joint - latent_log_joint
    objective = log_mean_exp(likelihood, axes=axes, keepdims=keepdims)
    return objective
