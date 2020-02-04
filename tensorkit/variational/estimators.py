from typing import *

from ..tensor import jit, Tensor, reduce_mean, log_mean_exp

__all__ = [
    'sgvb_estimator', 'iwae_estimator',
]


@jit
def sgvb_estimator(values: Tensor,
                   axis: Optional[List[int]] = None,
                   keepdims: bool = False,
                   negative: bool = False) -> Tensor:
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big]`,
    by SGVB (Kingma, D.P. and Welling, M., 2013) algorithm.

    .. math::

        \\nabla \\, \\mathbb{E}_{q(\\mathbf{z}|\\mathbf{x})}\\big[f(\\mathbf{x},\\mathbf{z})\\big] = \\nabla \\, \\mathbb{E}_{q(\\mathbf{\\epsilon})}\\big[f(\\mathbf{x},\\mathbf{z}(\\mathbf{\\epsilon}))\\big] = \\mathbb{E}_{q(\\mathbf{\\epsilon})}\\big[\\nabla f(\\mathbf{x},\\mathbf{z}(\\mathbf{\\epsilon}))\\big]

    Args:
        values: Values of the target function given `z` and `x`, i.e.,
            :math:`f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling axis to be reduced in outputs.
            If not specified, no axis will be reduced.
        keepdims: When `axis` is specified, whether or not to keep
            the reduced axis?  Defaults to :obj:`False`.
        negative: If :obj:`True`, returns negative of the gradient estimator,
            instead of the original gradient estimator derived from `values`.

    Returns:
        The surrogate for optimizing the original target.
        Maximizing/minimizing this surrogate via gradient descent will
        effectively maximize/minimize the original target.
    """
    estimator = values
    if axis is not None:
        estimator = reduce_mean(estimator, axis=axis, keepdims=keepdims)
    if negative:
        estimator = -estimator
    return estimator


@jit
def iwae_estimator(log_values: Tensor,
                   axis: Optional[List[int]] = None,
                   keepdims: bool = False,
                   negative: bool = False) -> Tensor:
    """
    Derive the gradient estimator for
    :math:`\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]`,
    by IWAE (Burda, Y., Grosse, R. and Salakhutdinov, R., 2015) algorithm.

    .. math::

        \\begin{aligned}
            &\\nabla\\,\\mathbb{E}_{q(\\mathbf{z}^{(1:K)}|\\mathbf{x})}\\Big[\\log \\frac{1}{K} \\sum_{k=1}^K f\\big(\\mathbf{x},\\mathbf{z}^{(k)}\\big)\\Big]
                = \\nabla \\, \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\log \\frac{1}{K} \\sum_{k=1}^K w_k\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\nabla \\log \\frac{1}{K} \\sum_{k=1}^K w_k\\Bigg] = \\\\
                & \\quad \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\frac{\\nabla \\frac{1}{K} \\sum_{k=1}^K w_k}{\\frac{1}{K} \\sum_{i=1}^K w_i}\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\frac{\\sum_{k=1}^K w_k \\nabla \\log w_k}{\\sum_{i=1}^K w_i}\\Bigg]
                = \\mathbb{E}_{q(\\mathbf{\\epsilon}^{(1:K)})}\\Bigg[\\sum_{k=1}^K \\widetilde{w}_k \\nabla \\log w_k\\Bigg]
        \\end{aligned}

    Args:
        log_values: Log values of the target function given `z` and `x`, i.e.,
            :math:`\\log f(\\mathbf{z},\\mathbf{x})`.
        axis: The sampling axis to be reduced in outputs.
            If not specified, no axis will be reduced.
        keepdims: When `axis` is specified, whether or not to keep
            the reduced axis?  Defaults to :obj:`False`.
        negative: If :obj:`True`, returns negative of the gradient estimator,
            instead of the original gradient estimator derived from `log_values`.

    Returns:
        The surrogate for optimizing the original target.
        Maximizing/minimizing this surrogate via gradient descent will
        effectively maximize/minimize the original target.
    """
    if axis is None or len(axis) == 0:
        raise ValueError(
            '`iwae_estimator` requires to take multiple samples of the latent '
            'variables, thus the `axis` argument must be specified'
        )
    estimator = log_mean_exp(log_values, axis=axis, keepdims=keepdims)
    if negative:
        estimator = -estimator
    return estimator
