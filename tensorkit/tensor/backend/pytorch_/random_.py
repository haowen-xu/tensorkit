from typing import *

import torch

from ....settings_ import settings
from . import dtypes
from .core import *
from .nn_ import nn
from .typing import *

__all__ = ['random']


class random(object):

    CATEGORICAL_DTYPE = torch.LongTensor.dtype

    RandomState = torch.Generator

    @staticmethod
    def seed(seed: int):
        torch.manual_seed(seed)

    @staticmethod
    def normal(mean: TensorLike,
               std: TensorLike,
               *,
               random_state: Optional[RandomState] = None) -> Tensor:
        mean = as_tensor(mean)
        std = as_tensor(std)
        return torch.normal(mean=mean, std=std, generator=random_state)

    @staticmethod
    def randn(shape: Sequence[int],
              dtype: DTypeLike = settings.float_x,
              *,
              random_state: Optional[RandomState] = None) -> Tensor:
        kwargs = {'generator': random_state} if random_state is not None else {}
        return torch.randn(as_shape(shape), dtype=as_dtype(dtype), **kwargs)

    @staticmethod
    def bernoulli(*,
                  logits: Optional[TensorLike] = None,
                  probs: Optional[TensorLike] = None,
                  n_samples: int = None,
                  dtype: DTypeLike = dtypes.int32,
                  random_state: Optional[RandomState] = None):
        # validate arguments
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')
        if n_samples is not None and n_samples < 1:
            raise ValueError(f'`n_samples` must be at least 1: '
                             f'got {n_samples}')

        dtype = as_dtype(dtype)

        if logits is not None:
            logits = as_tensor(logits)
            probs = nn.sigmoid(logits)
        else:
            probs = as_tensor(probs)

        # do sample
        sample_shape = shape(probs)
        if n_samples is not None:
            sample_shape = sample_shape + (n_samples,)
            probs = expand(expand_dim(probs, -1), sample_shape)

        with torch.no_grad():
            out = torch.zeros(sample_shape, dtype=dtype)
            kwargs = {'generator': random_state} \
                if random_state is not None else {}
            ret = torch.bernoulli(probs, out=out, **kwargs)
            return ret

    @staticmethod
    def categorical(*,
                    logits: Optional[TensorLike] = None,
                    probs: Optional[TensorLike] = None,
                    n_samples: int = None,
                    dtype: DTypeLike = CATEGORICAL_DTYPE,
                    random_state: Optional[RandomState] = None):
        # validate arguments
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')
        if n_samples is not None and n_samples < 1:
            raise ValueError(f'`n_samples` must be at least 1: '
                             f'got {n_samples}')

        dtype = as_dtype(dtype)

        if logits is not None:
            logits = as_tensor(logits)
            probs = nn.softmax(logits)
        else:
            probs = as_tensor(probs)

        probs_rank = rank(probs)
        if probs_rank < 1:
            raise ValueError(f'The rank of `logits` or `probs` must be at '
                             f'least 1: got {probs_rank}')

        # do sample
        if probs_rank > 2:
            probs, front_shape = flatten_to_ndims(probs, 2)
        else:
            probs, front_shape = probs, None

        probs_shape = shape(probs)
        out_shape = probs_shape[:-1]
        if n_samples is None:
            out_shape += (1,)
        else:
            out_shape += (n_samples,)

        with torch.no_grad():
            kwargs = {'generator': random_state} \
                if random_state is not None else {}

            ret = torch.multinomial(probs, n_samples or 1, replacement=True,
                                    **kwargs)

            if n_samples is None:
                ret = squeeze(ret, -1)

            if front_shape is not None:
                ret = unflatten_from_ndims(ret, front_shape)

            if ret.dtype != dtype:
                ret = cast(ret, dtype)

            return ret
