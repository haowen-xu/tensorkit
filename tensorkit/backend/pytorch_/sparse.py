from typing import *

import numpy as np
import torch
import torch.sparse
from scipy.sparse import spmatrix, coo_matrix
from torch import Tensor

from ...settings_ import settings
from .core import (is_sparse_jit_enabled, jit, jit_method, jit_ignore,
                   current_device, as_tensor, SparseTensor,
                   to_numpy as to_numpy_)

__all__ = [
    'SPARSE_INDICES_DEFAULT_IS_COORD_FIRST',
    'MAKE_SPARSE_DEFAULT_FORCE_COALESCED',

    # jit decorators
    'sparse_jit', 'sparse_jit_method',

    # sparse tensor <=> other types
    'make_sparse', 'is_sparse_tensor',
    'from_dense', 'to_dense', 'from_numpy', 'to_numpy',
    'from_spmatrix', 'to_spmatrix',

    # sparse tensor operations
    'coalesce', 'is_coalesced', 'get_indices', 'get_values',
    'rank', 'length', 'shape', 'get_dtype', 'get_device',
    'to_dtype', 'to_device', 'eye', 'reduce_sum', 'matmul',

    # sparse tensor grad utilities
    'stop_grad',
]

SPARSE_INDICES_DEFAULT_IS_COORD_FIRST = True
MAKE_SPARSE_DEFAULT_FORCE_COALESCED = True

SPRASE_CONSTRUCTOR = {
    torch.int32: torch.sparse.IntTensor,
    torch.int64: torch.sparse.LongTensor,
    torch.float16: torch.sparse.HalfTensor,
    torch.float32: torch.sparse.FloatTensor,
    torch.float64: torch.sparse.DoubleTensor,
}

if is_sparse_jit_enabled():
    # Note: sparse tensor support for JIT is only experimental in PyTorch 1.3.1
    sparse_jit = jit
    sparse_jit_method = jit_method
else:
    sparse_jit = sparse_jit_method = jit_ignore


# ---- sparse tensor <=> other types ----
@jit_ignore
def make_sparse(indices: Tensor,
                values: Tensor,
                shape: List[int],
                dtype: Optional[Union[str, torch.dtype]] = None,
                coord_first: bool = SPARSE_INDICES_DEFAULT_IS_COORD_FIRST,
                force_coalesced: bool = MAKE_SPARSE_DEFAULT_FORCE_COALESCED
                ) -> torch.Tensor:
    if dtype is None:
        target_dtype = values.dtype
    elif isinstance(dtype, str):
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = dtype

    # get the sparse constructor
    if target_dtype not in SPRASE_CONSTRUCTOR:
        raise ValueError(f'`dtype` not supported: {target_dtype!r}')
    sparse_ctor = SPRASE_CONSTRUCTOR[target_dtype]

    # Ensure that the index tensor is 2d and is int64.
    # Transpose the `indices` tensor into `(K, N)`, instead of `(N, K)`.
    if indices.dim() != 2:
        raise ValueError(f'`indices` must be a 2d tensor: got shape {indices.shape}')

    if indices.dtype == torch.int32:
        indices = indices.to(torch.int64)
    elif indices.dtype != torch.int64:
        raise ValueError(f'`indices` must be a int32 or int64 tensor: '
                         f'got dtype {indices.dtype}')

    if not coord_first:
        indices = indices.transpose(0, 1)

    # now construct the sparse tensor
    if target_dtype != values.dtype:
        values = values.to(target_dtype)

    m = sparse_ctor(indices, values, shape, device=values.device)
    if force_coalesced:
        m = m.coalesce()
    return m


@jit_ignore
def is_sparse_tensor(input: Tensor) -> bool:
    return isinstance(input, Tensor) and input.is_sparse


@jit_ignore
def from_dense(input: Tensor,
               force_coalesced: bool = MAKE_SPARSE_DEFAULT_FORCE_COALESCED
               ) -> Tensor:
    mask = (input != 0)
    indices = torch.stack(torch.where(mask), dim=0)
    values = input[mask]
    return make_sparse(indices, values, shape=list(input.shape),
                       force_coalesced=force_coalesced)


@sparse_jit
def to_dense(input: Tensor) -> Tensor:
    return input.to_dense()


@jit_ignore
def from_numpy(input: np.ndarray,
               dtype: Optional[Union[str, torch.dtype]] = None,
               device: Optional[str] = None,
               force_coalesced: bool = MAKE_SPARSE_DEFAULT_FORCE_COALESCED,
               ) -> Tensor:
    mask = (input != 0)
    indices = as_tensor(np.stack(np.where(mask), axis=0), dtype=torch.int64,
                        device=device)
    values = as_tensor(input[mask], dtype=dtype, device=device)
    return make_sparse(indices, values, list(input.shape),
                       force_coalesced=force_coalesced)


@jit_ignore
def to_numpy(input: SparseTensor) -> np.ndarray:
    return to_numpy_(to_dense(input))


@jit_ignore
def from_spmatrix(input: spmatrix,
                  dtype: Optional[Union[str, torch.dtype]] = None,
                  device: Optional[str] = None,
                  force_coalesced: bool = MAKE_SPARSE_DEFAULT_FORCE_COALESCED,
                  ) -> Tensor:
    if not isinstance(input, coo_matrix):
        input = input.tocoo(copy=False)
    shape = list(input.shape)
    indices = as_tensor(np.stack([input.row, input.col], axis=0),
                        dtype=torch.int64, device=device)
    values = as_tensor(input.data, dtype=dtype, device=device, force_copy=True)
    return make_sparse(indices, values, shape, force_coalesced=force_coalesced)


@jit_ignore
def to_spmatrix(input: SparseTensor) -> spmatrix:
    if not input.is_coalesced():
        input = input.coalesce()
    indices = to_numpy_(input.indices())
    values = to_numpy_(input.values())
    return coo_matrix((values, (indices[0], indices[1])), shape=input.shape)


# ---- sparse tensor operations ----
@sparse_jit
def coalesce(input: Tensor) -> Tensor:
    if not input.is_coalesced():
        input = input.coalesce()
    return input


@sparse_jit
def is_coalesced(input: Tensor) -> bool:
    return input.is_coalesced()


@sparse_jit
def get_indices(input: Tensor,
                coord_first: bool = SPARSE_INDICES_DEFAULT_IS_COORD_FIRST
                ) -> Tensor:
    ret = input.indices()
    if not coord_first:
        ret = ret.transpose(0, 1)
    return ret


@sparse_jit
def get_values(input: Tensor) -> Tensor:
    return input.values()


@sparse_jit
def rank(input: Tensor) -> int:
    return input.dim()


@sparse_jit
def length(input: Tensor) -> int:
    return input.shape[0]


@sparse_jit
def shape(input: Tensor) -> List[int]:
    return list(input.shape)


@sparse_jit
def get_dtype(input: Tensor) -> str:
    if input.dtype == torch.float32:
        return 'float32'
    elif input.dtype == torch.int32:
        return 'int32'
    else:
        return {torch.int8: 'int8', torch.uint8: 'uint8', torch.int16: 'int16', torch.int64: 'int64', torch.float16: 'float16', torch.float64: 'float64', torch.bool: 'bool'}[input.dtype]


@sparse_jit
def get_device(input: Tensor) -> str:
    return str(input.device)


@sparse_jit
def to_dtype(input: Tensor, dtype: str) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if target_dtype != input.dtype:
        input = input.to(dtype=target_dtype)
    return input


@sparse_jit
def to_device(input: Tensor, device: str) -> Tensor:
    if str(input.device) != device:
        input = input.to(device=device)
    return input


@jit_ignore
def eye(n: int,
        m: Optional[int] = None,
        dtype: str = settings.float_x,
        device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if m is None:
        m = n
    if device is None:
        device = current_device()

    # construct the eye matrix
    k = max(n, m)
    indices = torch.arange(k, dtype=torch.int64, device=device)
    indices = torch.stack([indices, indices], dim=0)
    values = torch.ones([k], dtype=target_dtype, device=device)

    return torch.sparse.FloatTensor(indices, values, [n, m], device=values.device)


@sparse_jit
def reduce_sum(t: Tensor, axis: Optional[int] = None) -> Tensor:
    if not t.is_coalesced():
        t = t.coalesce()
    if axis is None:
        return torch.sparse.sum(t)
    else:
        return torch.sparse.sum(t, dim=(axis,))


@sparse_jit
def matmul(x: Tensor, y: Tensor) -> Tensor:  # only matmul(sparse, dense) is supported
    if not x.is_coalesced():
        x = x.coalesce()
    return torch.sparse.mm(x, y)


# ---- sparse tensor grad utilities ----
@jit
def stop_grad(input: Tensor) -> Tensor:
    return input.detach()
