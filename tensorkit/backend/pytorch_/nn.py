import torch

from .core import *

__all__ = [
    # activation functions
    'relu', 'leaky_relu', 'sigmoid', 'softmax', 'log_softmax',

    # cross entropy functions
    'binary_cross_entropy_with_logits', 'cross_entropy_with_logits',
    'sparse_cross_entropy_with_logits',
]


# ---- activation functions ----
@jit
def relu(x: Tensor) -> Tensor:
    return torch.relu(x)


@jit
def leaky_relu(x: Tensor, a: float = 0.01) -> Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=a)


@jit
def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


@jit
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return torch.softmax(x, dim=axis)


@jit
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return torch.log_softmax(x, dim=axis)


# ---- cross entropy functions ----
@jit
def binary_cross_entropy_with_logits(logits: Tensor,
                                     labels: Tensor,
                                     reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                                     negative: bool = False) -> Tensor:
    if labels.dtype != logits.dtype:
        labels = labels.to(dtype=logits.dtype)
    logits, labels = explicit_broadcast(logits, labels)
    ret = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if negative:
        ret = -ret
    return ret


@jit
def cross_entropy_with_logits(logits: Tensor,
                              labels: Tensor,
                              reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                              negative: bool = False) -> Tensor:

    if logits.shape[:-1] != labels.shape:
        logits_shape = list(logits.shape)
        labels_shape = list(labels.shape)
        b_shape = broadcast_shape(logits_shape[:-1], labels_shape)
        logits = broadcast_to(logits, b_shape + logits_shape[-1:])
        labels = broadcast_to(labels, b_shape)

    if len(logits.shape) < 2 or len(labels.shape) < 1:
        raise ValueError('`logits` must be at least 2d, and `labels` must '
                         'be at least 1d: logits.shape is {}, while '
                         'labels.shape is {}.'.
                         format(logits.shape, labels.shape))

    logits, front_shape = flatten_to_ndims(logits, 2)
    labels, _ = flatten_to_ndims(labels, 1)

    ret = torch.nn.functional.cross_entropy(
        logits, labels, reduction=reduction)
    if negative:
        ret = -ret

    if reduction == 'none':
        ret = unflatten_from_ndims(ret, front_shape)
    return ret


@jit
def sparse_cross_entropy_with_logits(logits: Tensor,
                                     labels: Tensor,
                                     reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                                     negative: bool = False) -> Tensor:
    if reduction != 'none' and reduction != 'sum' and reduction != 'mean':
        raise ValueError('`reduce` is not one of "none", "sum" and '
                         '"mean": got {}'.format(reduction))

    logits, labels = explicit_broadcast(logits, labels)
    logits_shape = logits.shape
    labels_shape = labels.shape

    if len(logits_shape) < 2 or len(labels_shape) < 2:
        raise ValueError('`logits` and `labels` must be at least 2d: '
                         'logits.shape is {}, while labels.shape '
                         'is {}.'.format(logits.shape, labels.shape))

    log_sum_exp_logits = torch.logsumexp(logits, dim=-1, keepdim=True)

    if negative:
        ret = labels * (logits - log_sum_exp_logits)
    else:
        ret = labels * (log_sum_exp_logits - logits)

    if reduction == 'sum':
        ret = torch.sum(ret)
    elif reduction == 'mean':
        ret = torch.mean(torch.sum(ret, dim=-1))
    else:
        ret = torch.sum(ret, dim=-1)

    return ret
