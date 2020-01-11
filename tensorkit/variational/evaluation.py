from typing import *

from ..tensor import jit, Tensor, log_mean_exp

__all__ = ['importance_sampling_log_likelihood']


@jit
def importance_sampling_log_likelihood(log_joint: Tensor,
                                       latent_log_joint: Tensor,
                                       axes: Optional[List[int]] = None,
                                       keepdims: bool = False):
    """
    Compute :math:`\\log p(\\mathbf{x})` by importance sampling.

    .. math::

        \\log p(\\mathbf{x}) =
            \\log \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})} \\Big[\\exp\\big(\\log p(\\mathbf{x},\\mathbf{z}) - \\log q(\\mathbf{z}|\\mathbf{x})\\big) \\Big]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_joint: :math:`\\log q(\\mathbf{z}|\\mathbf{x})`.
        axes: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        keepdims: When `axes` is specified, whether or not to keep
            the reduced axes?  Defaults to :obj:`False`.

    Returns:
        The computed :math:`\\log p(x)`.
    """
    if axes is None or len(axes) == 0:
        raise ValueError(
            '`importance_sampling_log_likelihood` requires to take '
            'multiple samples of the latent variables, '
            'thus the `axes` argument must be specified'
        )
    log_p = log_mean_exp(
        log_joint - latent_log_joint, axes=axes, keepdims=keepdims)
    return log_p
