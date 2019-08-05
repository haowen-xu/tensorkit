from typing import Optional

import torch
import torch.nn.functional
from torch.nn import functional as F

from .core import *
from .core import _broadcast_shape, _broadcast_to, _explicit_broadcast
from .typing import *

__all__ = ['nn']


@jit
def _binary_cross_entropy_with_logits(logits: Tensor,
                                      labels: Tensor,
                                      reduction: str,
                                      negative: bool) -> Tensor:
    if labels.dtype != logits.dtype:
        labels = labels.to(dtype=logits.dtype)
    logits, labels = _explicit_broadcast(logits, labels)
    ret = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if negative:
        ret = -ret
    return ret


@jit
def _sparse_cross_entropy_with_logits(logits: Tensor,
                                      labels: Tensor,
                                      reduction: str = 'none',
                                      negative: bool = False) -> Tensor:
    if reduction != 'none' and reduction != 'sum' and reduction != 'mean':
        raise ValueError('`reduce` is not one of "none", "sum" and '
                         '"mean": got {}'.format(reduction))

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


class nn(object):

    # activation functions
    @staticmethod
    def relu(x: TensorLike) -> Tensor:
        return torch.relu(as_tensor(x))

    @staticmethod
    def leaky_relu(x: TensorLike, a: float = 0.01) -> Tensor:
        x = as_tensor(x)
        return F.leaky_relu(x, negative_slope=a)

    @staticmethod
    def sigmoid(x: TensorLike):
        return torch.sigmoid(as_tensor(x))

    @staticmethod
    def softmax(x: TensorLike, axis: int = -1) -> Tensor:
        return torch.softmax(as_tensor(x), dim=axis)

    @staticmethod
    def log_softmax(x: TensorLike, axis: int = -1) -> Tensor:
        return torch.log_softmax(as_tensor(x), dim=axis)

    # cross entropy functions
    @staticmethod
    def binary_cross_entropy_with_logits(logits: TensorLike,
                                         labels: TensorLike,
                                         reduction: str = 'none',
                                         negative: bool = False) -> Tensor:
        return _binary_cross_entropy_with_logits(
            as_tensor(logits), as_tensor(labels), reduction, negative)

    @staticmethod
    def cross_entropy_with_logits(logits: TensorLike,
                                  labels: TensorLike,
                                  reduction: str = 'none',
                                  negative: bool = False) -> Tensor:
        logits = as_tensor(logits)
        labels = as_tensor(labels)
        logits_shape = list(logits.shape)
        labels_shape = list(labels.shape)

        if len(logits_shape) < 2 or len(labels_shape) < 1:
            raise ValueError(f'`logits` must be at least 2d, and `labels` must '
                             f'be at least 1d: logits.shape is {logits.shape}, '
                             f'while labels.shape is {labels.shape}.')

        if logits_shape[:-1] != labels_shape:
            b_shape = _broadcast_shape(logits_shape[:-1], labels_shape)
            logits = _broadcast_to(logits, b_shape + logits_shape[-1:])
            labels = _broadcast_to(labels, b_shape)

        logits, front_shape = flatten_to_ndims(logits, 2)
        labels, _ = flatten_to_ndims(labels, 1)

        ret = F.cross_entropy(logits, labels, reduction=reduction)
        if negative:
            ret = -ret

        if reduction == 'none':
            ret = unflatten_from_ndims(ret, front_shape)
        return ret

    @staticmethod
    def sparse_cross_entropy_with_logits(logits: TensorLike,
                                         labels: TensorLike,
                                         reduction: str = 'none',
                                         negative: bool = False) -> Tensor:
        return _sparse_cross_entropy_with_logits(
            as_tensor(logits), as_tensor(labels), reduction, negative)

    # tensor transformations
    @staticmethod
    def one_hot(x: TensorLike,
                n_classes: int,
                dtype: Optional[DTypeLike] = None) -> Tensor:
        if dtype is not None:
            dtype = as_dtype(dtype)
            ret = F.one_hot(as_tensor(x), n_classes)
            if ret.dtype != dtype:
                ret = ret.to(dtype)
            return ret
        else:
            return F.one_hot(as_tensor(x), n_classes)
