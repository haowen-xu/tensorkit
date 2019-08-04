import torch
from torch.nn import functional as F

from .core import *
from .typing import *

__all__ = ['nn']


class nn(object):

    @staticmethod
    def relu(x: TensorLike) -> Tensor:
        return torch.relu(as_tensor(x))

    @staticmethod
    def sigmoid(x: TensorLike):
        return torch.sigmoid(as_tensor(x))

    @staticmethod
    def softmax(x: TensorLike, axis: int = -1) -> Tensor:
        return torch.softmax(as_tensor(x), dim=axis)

    @staticmethod
    def log_softmax(x: TensorLike, axis: int = -1) -> Tensor:
        return torch.log_softmax(as_tensor(x), dim=axis)

    @staticmethod
    def binary_cross_entropy_with_logits(logits: TensorLike,
                                         labels: TensorLike,
                                         reduction: str = 'none',
                                         negative: bool = False) -> Tensor:
        logits, labels = explicit_broadcast(logits, labels)
        ret = F.binary_cross_entropy_with_logits(
            logits, labels, reduction=reduction)
        if negative:
            ret = -ret
        return ret

    @staticmethod
    def cross_entropy_with_logits(logits: TensorLike,
                                  labels: TensorLike,
                                  reduction: str = 'none',
                                  negative: bool = False) -> Tensor:
        logits = as_tensor(logits)
        labels = as_tensor(labels)
        logits_shape = shape(logits)
        labels_shape = shape(labels)

        if len(logits_shape) < 2 or len(labels_shape) < 1:
            raise ValueError(f'`logits` must be at least 2d, and `labels` must '
                             f'be at least 1d: logits.shape is {logits}, while '
                             f'labels.shape is {labels}.')

        if logits_shape[:-1] != labels_shape:
            b_shape = broadcast_shape(logits_shape[:-1], labels_shape)
            logits = broadcast_to(logits, b_shape + logits_shape[-1:])
            labels = broadcast_to(labels, b_shape)

        logits, front_shape = flatten_to_ndims(logits, 2)
        labels, _ = flatten_to_ndims(labels, 1)

        ret = F.cross_entropy(logits, labels, reduction=reduction)
        if negative:
            ret = -ret

        ret = unflatten_from_ndims(ret, front_shape)
        return ret

    @staticmethod
    def sparse_cross_entropy_with_logits(logits: TensorLike,
                                         labels: TensorLike,
                                         reduction: str = 'none',
                                         negative: bool = False) -> Tensor:
        if reduction not in ('none', 'sum', 'mean'):
            raise ValueError(f'`reduce` is not one of "none", "sum" and '
                             f'"mean": got {reduction!r}')

        logits = as_tensor(logits)
        labels = as_tensor(labels)
        logits_shape = shape(logits)
        labels_shape = shape(labels)

        if len(logits_shape) < 2 or len(labels_shape) < 2:
            raise ValueError(f'`logits` and `labels` must be at least 2d: '
                             f'logits.shape is {logits}, while labels.shape '
                             f'is {labels}.')

        log_sum_exp_logits = log_sum_exp(logits, axis=-1, keepdims=True)

        if negative:
            ret = labels * (logits - log_sum_exp_logits)
        else:
            ret = labels * (log_sum_exp_logits - logits)

        if reduction == 'sum':
            ret = reduce_sum(ret)
        elif reduction == 'mean':
            ret = reduce_mean(reduce_sum(ret, axis=-1))
        else:
            ret = reduce_sum(ret, axis=-1)

        return ret

    @staticmethod
    def one_hot(x: TensorLike, n_classes: int) -> Tensor:
        return F.one_hot(x, n_classes)
