from typing import *

import torch

from .core import *

__all__ = ['qr', 'slogdet']


@jit
def qr(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return torch.qr(matrix)


@jit
def slogdet(matrix: Tensor) -> Tuple[Tensor, Tensor]:
    return torch.slogdet(matrix)
