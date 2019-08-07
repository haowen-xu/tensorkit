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
    def new_state(seed: Optional[int] = None):
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        return g

    @staticmethod
    def normal(mean: TensorLike,
               std: TensorLike,
               *,
               random_state: Optional[RandomState] = None) -> Tensor:
        mean, std = explicit_broadcast(mean, std)
        return torch.normal(mean=mean, std=std, generator=random_state)

    @staticmethod
    def randn(shape: Sequence[int],
              dtype: DTypeLike = settings.float_x,
              *,
              random_state: Optional[RandomState] = None) -> Tensor:
        shape = as_shape(shape)
        dtype = as_dtype(dtype)
        if random_state is not None:
            return torch.randn(shape, dtype=dtype, generator=random_state)
        else:
            return torch.randn(shape, dtype=dtype)

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

        with torch.no_grad():
            if logits is not None:
                logits = as_tensor(logits)
                probs = torch.sigmoid(logits)
            else:
                probs = as_tensor(probs)

            # do sample
            sample_shape = probs.shape
            if n_samples is not None:
                sample_shape = (n_samples,) + sample_shape
                probs = probs.unsqueeze(dim=0).expand(sample_shape)

            out = torch.zeros(sample_shape, dtype=dtype)
            return torch.bernoulli(probs, out=out, generator=random_state)

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

        with torch.no_grad():
            if logits is not None:
                logits = as_tensor(logits)
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = as_tensor(probs)

            probs_rank = len(probs.shape)
            if probs_rank < 1:
                raise ValueError(f'The rank of `logits` or `probs` must be at '
                                 f'least 1: got {probs_rank}')

            # do sample
            if probs_rank > 2:
                probs, front_shape = flatten_to_ndims(probs, 2)
            else:
                probs, front_shape = probs, None

            probs_shape = probs.shape
            out_shape = probs_shape[:-1]
            if n_samples is None:
                out_shape += (1,)
            else:
                out_shape += (n_samples,)

            ret = torch.multinomial(probs, n_samples or 1, replacement=True,
                                    generator=random_state)
            if n_samples is None:
                ret = torch.squeeze(ret, -1)
                if front_shape is not None:
                    ret = unflatten_from_ndims(ret, front_shape)
            else:
                if front_shape is not None:
                    ret = unflatten_from_ndims(ret, front_shape)
                ret = ret.permute((-1,) + tuple(range(len(ret.shape) - 1)))

            if ret.dtype != dtype:
                ret = ret.to(dtype)
            return ret
