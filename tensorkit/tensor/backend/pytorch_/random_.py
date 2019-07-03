from typing import *

import torch

from ....settings_ import settings
from .core import TensorLike, Tensor, as_tensor, DTypeLike, as_dtype

__all__ = ['random']


class random(object):

    RandomState = torch.Generator

    @classmethod
    def seed(cls, seed: int):
        torch.manual_seed(seed)

    @classmethod
    def normal(cls,
               mean: TensorLike,
               std: TensorLike,
               *,
               random_state: Optional[RandomState] = None) -> Tensor:
        mean = as_tensor(mean)
        std = as_tensor(std)
        return torch.normal(mean=mean, std=std, generator=random_state)

    @classmethod
    def randn(cls,
              shape: Sequence[int],
              dtype: DTypeLike = settings.float_x,
              *,
              random_state: Optional[RandomState] = None) -> Tensor:
        kwargs = {'generator': random_state} if random_state is not None else {}
        return torch.randn(tuple(shape), dtype=as_dtype(dtype), **kwargs)
