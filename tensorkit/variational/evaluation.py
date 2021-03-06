from typing import *

from ..tensor import jit, Tensor, log_mean_exp
from .utils import apply_reduction

__all__ = ['importance_sampling_log_likelihood']


@jit
def importance_sampling_log_likelihood(log_joint: Tensor,
                                       latent_log_joint: Tensor,
                                       axis: Optional[List[int]] = None,
                                       keepdims: bool = False,
                                       reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                                       ) -> Tensor:
    """
    Compute :math:`\\log p(\\mathbf{x})` by importance sampling.

    .. math::

        \\log p(\\mathbf{x}) =
            \\log \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})} \\Big[\\exp\\big(\\log p(\\mathbf{x},\\mathbf{z}) - \\log q(\\mathbf{z}|\\mathbf{x})\\big) \\Big]

    Args:
        log_joint: Values of :math:`\\log p(\\mathbf{z},\\mathbf{x})`,
            computed with :math:`\\mathbf{z} \\sim q(\\mathbf{z}|\\mathbf{x})`.
        latent_log_joint: :math:`\\log q(\\mathbf{z}|\\mathbf{x})`.
        axis: The sampling dimensions to be averaged out.
            If :obj:`None`, no dimensions will be averaged out.
        reduction: "sum" to return the sum of the log-likelihood,
            "mean" to return the mean of the log-likelihood, or "none" to
            return the original element-wise log-likelihood.
        keepdims: When `axis` is specified, whether or not to keep
            the reduced axis?  Defaults to :obj:`False`.

    Returns:
        The computed :math:`\\log p(x)`.
    """
    if axis is None or len(axis) == 0:
        raise ValueError(
            '`importance_sampling_log_likelihood` requires to take '
            'multiple samples of the latent variables, '
            'thus the `axis` argument must be specified'
        )
    log_p = log_mean_exp(
        log_joint - latent_log_joint, axis=axis, keepdims=keepdims)
    log_p = apply_reduction(log_p, reduction)
    return log_p
