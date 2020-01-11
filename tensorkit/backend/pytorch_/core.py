import math
from typing import *

import numpy as np
import torch
import torch.jit
import torch.nn.functional

from ...settings_ import settings

__all__ = [
    # typing
    'Tensor', 'Variable',

    # jit
    'jit', 'jit_ignore',

    # utilities
    'int_range',

    # dtypes
    'cast', 'cast_like', 'get_dtype', 'is_floating_point',

    # tensor constructors
    'as_tensor', 'from_numpy', 'float_scalar', 'int_scalar',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like',
    'arange', 'one_hot',

    # read / assign
    'to_numpy',

    # shape utils
    'shape', 'rank', 'reshape', 'repeat', 'expand', 'squeeze', 'expand_dim',
    'broadcast_shape', 'broadcast_to', 'explicit_broadcast', 'flatten_to_ndims',
    'unflatten_from_ndims',

    # split / join / indexing / gathering
    'index_select', 'concat',

    # math operators
    'abs', 'neg', 'square', 'exp', 'log', 'log1p', 'sin', 'cos',
    'erf', 'erfc', 'erfinv',
    'add', 'sub', 'mul', 'div', 'mod', 'pow', 'truediv', 'floordiv',
    'add_n',

    # reduce operators
    'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min',
    'log_sum_exp', 'log_mean_exp',
    # 'all', 'any',

    # logical operators
    'logical_not', 'logical_and', 'logical_or', 'logical_xor', 'multiply_mask',
    'where',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'minimum', 'maximum', 'clip',

    # gradient utilities
    'grad', 'is_null_grad', 'requires_grad', 'stop_grad',

    # debug utilities
    'is_all', 'is_finite', 'assert_finite',
]


# ---- typing ----
Tensor = torch.Tensor
Variable = torch.Tensor


# ---- jit ----
def jit(fn):
    if not settings.disable_jit:
        fn = torch.jit.script(fn)
    return fn


def jit_ignore(obj):
    if not settings.disable_jit:
        obj = torch.jit.ignore(obj)
    return obj


# ---- utilities ----
if settings.disable_jit:
    def int_range(start: int, end: int, step: int = 1) -> List[int]:
        return list(range(start, end, step))
else:
    @jit
    def int_range(start: int, end: int, step: int = 1) -> List[int]:
        ret: List[int] = []
        for i in range(start, end, step):
            ret.append(i)
        return ret


# ---- dtypes ----
@jit
def cast(x: Tensor, dtype: str) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    if target_dtype != x.dtype:
        x = x.to(dtype=target_dtype)
    return x


@jit
def cast_like(x: Tensor, dtype_as: Tensor) -> Tensor:
    if dtype_as.dtype != x.dtype:
        x = x.to(dtype=dtype_as.dtype)
    return x


@jit
def get_dtype(x: Tensor) -> str:
    if x.dtype == torch.float32:
        return 'float32'
    elif x.dtype == torch.int32:
        return 'int32'
    else:
        return {torch.int8: 'int8', torch.uint8: 'uint8', torch.int16: 'int16', torch.int64: 'int64', torch.float16: 'float16', torch.float64: 'float64', torch.bool: 'bool'}[x.dtype]


@jit
def is_floating_point(x: Tensor) -> bool:
    return x.is_floating_point()


# ---- tensor constructors ----
as_tensor = torch.as_tensor
"""
Use only the form ``(data) -> torch.Tensor``, or the form
``(data, dtype=another_tensor.dtype) -> torch.Tensor``.
"""


@jit_ignore
def from_numpy(data, dtype: Optional[Union[torch.dtype, str]] = None) -> Tensor:
    # check the dtype argument
    target_dtype = dtype
    if dtype is not None:
        if not isinstance(dtype, torch.dtype):
            if dtype == 'float32':
                target_dtype = torch.float32
            elif dtype == 'int32':
                target_dtype = torch.int32
            else:
                target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    # construct the tensor
    r = torch.from_numpy(np.asarray(data))

    # cast to desired type
    if target_dtype is not None and r.dtype != target_dtype:
        r = r.to(target_dtype)

    return r


@jit
def float_scalar(data: float, dtype: str = settings.float_x) -> Tensor:
    if dtype == 'float32':
        real_dtype = torch.float32
    else:
        real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]
    return torch.tensor(data, dtype=real_dtype)


@jit
def int_scalar(data: int, dtype: str = 'int32') -> Tensor:
    if dtype == 'int32':
        int_dtype = torch.int32
    else:
        int_dtype = {'int8': torch.int8, 'int16': torch.int16, 'int64': torch.int64}[dtype]
    return torch.tensor(data, dtype=int_dtype)


@jit
def zeros(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    return torch.zeros(shape, dtype=target_dtype)


@jit
def zeros_like(x: Tensor,
               dtype: Optional[str] = None,
               shape: Optional[List[int]] = None) -> Tensor:
    if dtype is not None:
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = x.dtype
    if shape is None:
        shape = list(x.shape)
    return torch.zeros(shape, dtype=target_dtype)


@jit
def ones(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    return torch.ones(shape, dtype=target_dtype)


@jit
def ones_like(x: Tensor,
              dtype: Optional[str] = None,
              shape: Optional[List[int]] = None) -> Tensor:
    if dtype is not None:
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = x.dtype
    if shape is None:
        shape = list(x.shape)
    return torch.ones(shape, dtype=target_dtype)


@jit
def full(shape: List[int],
         fill_value: float,
         dtype: str = settings.float_x) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    return torch.full(shape, fill_value, dtype=target_dtype)


@jit
def full_like(x: Tensor,
              fill_value: float,
              dtype: Optional[str] = None,
              shape: Optional[List[int]] = None) -> Tensor:
    if dtype is not None:
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = x.dtype
    if shape is None:
        shape = list(x.shape)
    return torch.full(shape, fill_value, dtype=target_dtype)


@jit
def arange(start: int, end: int, step: int = 1, dtype: str = 'int32') -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    return torch.arange(start, end, step, dtype=target_dtype)


@jit
def one_hot(x: Tensor,
            n_classes: int,
            dtype: str = 'int32') -> Tensor:
    ret = torch.nn.functional.one_hot(x, n_classes)
    ret = cast(ret, dtype)
    return ret


# ---- read / assign ----
@jit_ignore
def to_numpy(x: Tensor) -> np.ndarray:
    if not isinstance(x, Tensor):
        raise TypeError(f'Not a Tensor: got {x!r}')
    return x.detach().cpu().numpy()


# ---- shape utils ----
@jit
def shape(x: Tensor) -> List[int]:
    return list(x.shape)


@jit
def rank(x: Tensor) -> int:
    return len(x.shape)


@jit
def reshape(x: Tensor, shape: List[int]) -> Tensor:
    return x.reshape(shape)


@jit
def repeat(x: Tensor, repeats: List[int]) -> Tensor:
    x_shape = x.shape
    x_rank = len(x_shape)
    repeats_len = len(repeats)
    extra_len = repeats_len - x_rank

    # argument check
    if extra_len < 0:
        repeats = [1] * (len(x_shape) - len(repeats)) + repeats
        extra_len = 0

    # detect the repeat mode
    mode = 0  # 0 = return directly, 1 = expand, 2 = repeat
    if extra_len > 0:
        mode = 1

    for i in range(len(x_shape)):
        a = x_shape[i]
        b = repeats[i + extra_len]
        if b != 1:
            if a != 1:
                mode = 2
            else:
                mode = max(1, mode)

    # do repeat the tensor according to different mode
    if mode == 0:
        return x
    elif mode == 1:
        expands = repeats[:extra_len] + \
            list([-1 if a == 1 else a for a in repeats[extra_len:]])
        return x.expand(expands)
    else:
        return x.repeat(repeats)


@jit
def expand(x: Tensor, desired_shape: List[int]) -> Tensor:
    return x.expand(desired_shape)


@jit
def squeeze(x: Tensor, axes: Optional[List[int]] = None) -> Tensor:
    if axes is not None:
        if len(axes) == 1:
            return torch.squeeze(x, axes[0])
        else:
            old_shape = x.shape
            new_shape_mask = [True] * len(old_shape)
            for a in axes:
                if old_shape[a] == 1:
                    new_shape_mask[a] = False
                else:
                    raise ValueError('Axis {} cannot be squeezed, since its '
                                     'size is {} != 1'.format(a, old_shape[a]))
            new_shape = torch.jit.annotate(List[int], [])
            for i in range(len(old_shape)):
                if new_shape_mask[i]:
                    new_shape.append(old_shape[i])
            return x.reshape(new_shape)
    else:
        return torch.squeeze(x)


@jit
def expand_dim(x: Tensor, axis: int) -> Tensor:
    return x.unsqueeze(axis)


@jit
def broadcast_shape(x: List[int], y: List[int]) -> List[int]:
    common_len = min(len(x), len(y))

    right = torch.jit.annotate(List[int], [])
    for i in range(common_len):
        a = x[i - common_len]
        b = y[i - common_len]
        if a == 1:
            right.append(b)
        elif b == 1:
            right.append(a)
        elif a != b:
            raise ValueError('Shape x and y cannot broadcast against '
                             'each other: {} vs {}.'.format(x, y))
        else:
            right.append(a)

    if len(x) > common_len:
        left = x[:len(x)-common_len]
    else:
        left = y[:len(y)-common_len]
    return left + right


@jit
def _broadcast_to_sub(t: Tensor,
                      t_shape: List[int],
                      out_shape: List[int]) -> Tensor:
    t_rank = len(t_shape)
    out_rank = len(out_shape)

    if t_rank < out_rank:
        t_shape = [1] * (out_rank - t_rank) + t_shape

    t_repeats = torch.jit.annotate(List[int], [])
    should_repeat = False
    for i in range(out_rank):
        a = t_shape[i]
        b = out_shape[i]
        if a == 1 and b != 1:
            t_repeats.append(b)
            should_repeat = True
        else:
            t_repeats.append(1)

    if should_repeat:
        t = t.repeat(t_repeats)
    return t


@jit
def broadcast_to(x: Tensor, new_shape: List[int]) -> Tensor:
    x_shape = list(x.shape)
    x_rank = len(x_shape)
    new_rank = len(new_shape)

    if x_rank > new_rank:
        raise ValueError('`x` cannot be broadcast to `new_shape`: shape(x) {} '
                         'vs new_shape {}'.format(x_shape, new_shape))

    for i in range(x_rank):
        a = x_shape[-i - 1]
        b = new_shape[-i - 1]
        if a != 1 and a != b:
            raise ValueError('`x` cannot be broadcast to `new_shape`: '
                             'shape(x) {} vs new_shape {}'.
                             format(x_shape, new_shape))

    return _broadcast_to_sub(x, x_shape, new_shape)


@jit
def explicit_broadcast(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    out_shape = broadcast_shape(x_shape, y_shape)
    x = _broadcast_to_sub(x, x_shape, out_shape)
    y = _broadcast_to_sub(y, y_shape, out_shape)
    return x, y


@jit
def flatten_to_ndims(x: Tensor, ndims: int
                     ) -> Tuple[Tensor, Optional[List[int]]]:
    if ndims < 1:
        raise ValueError('`ndims` must be at least 1`: got ndims {}'.
                         format(ndims))
    if len(x.shape) < ndims:
        raise ValueError('rank(x) < ndims: x.shape is {}, while '
                         'ndims is {}'.format(x.shape, ndims))

    if ndims == len(x.shape):
        return x, None  # `None` to indicate x is not changed
    elif ndims == 1:
        front_shape = list(x.shape)
        return x.reshape((-1,)), front_shape
    else:
        x_shape = list(x.shape)
        offset = ndims - 1
        front_shape, back_shape = x_shape[: -offset], x_shape[-offset:]
        return x.reshape([-1] + back_shape), front_shape


@jit
def unflatten_from_ndims(x: Tensor, front_shape: Optional[List[int]]
                         ) -> Tensor:
    x_shape = list(x.shape)
    if front_shape is None:
        return x
    else:
        x_rank = len(x_shape)
        if x_rank < 1:
            raise ValueError(
                'Invalid input: rank(x) < 1, but front_shape is not None.')
        return x.reshape(list(front_shape) + x_shape[1:])


# ---- split / join / indexing / gathering ----
@jit
def index_select(x: Tensor, indices: Tensor, axis: int) -> Tensor:
    x_shape = x.shape
    i_shape = indices.shape

    if axis < 0:
        axis += len(x_shape)
    if axis < 0 or axis >= len(x_shape):
        raise ValueError('`axis` out of range: x.shape {} vs axis {}'.
                         format(x.shape, axis))

    if len(i_shape) == 0:
        y = torch.index_select(x, dim=axis, index=indices.reshape([1]))
        y = y.reshape(x_shape[:axis] + x_shape[axis + 1:])

    elif len(i_shape) == 1:
        y = torch.index_select(x, dim=axis, index=indices)

    else:
        y = torch.index_select(x, dim=axis, index=indices.flatten())
        y = y.reshape(x_shape[:axis] + i_shape + x_shape[axis + 1:])

    return y


@jit
def concat(tensors: List[Tensor], axis: int) -> Tensor:
    return torch.cat(tensors, dim=axis)


# ---- univariate element-wise math operations ----
abs = torch.abs
neg = torch.neg


@jit
def square(x: Tensor) -> Tensor:
    return x ** 2


exp = torch.exp
log = torch.log
log1p = torch.log1p

sin = torch.sin
cos = torch.cos

erf = torch.erf
erfc = torch.erfc
erfinv = torch.erfinv


# ---- bivariate element-wise math operations ----
add = torch.add
sub = torch.sub
mul = torch.mul
div = torch.div
mod = torch.fmod
pow = torch.pow


@jit
def truediv(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != y.dtype:
        raise TypeError('x and y must have the same dtype, got '
                        '{} != {}'.format(x.dtype, y.dtype))

    dtype = x.dtype
    if not x.is_floating_point():
        if dtype == torch.int8 or dtype == torch.uint8 or dtype == torch.int16:
            x = x.to(torch.float32)
            y = y.to(torch.float32)
        else:
            x = x.to(torch.float64)
            y = y.to(torch.float64)

    return x / y


@jit
def floordiv(x: Tensor, y: Tensor) -> Tensor:
    ret = torch.div(x, y)
    if ret.is_floating_point():
        ret = ret.floor()
    return ret


# ---- sequential math operations ----
@jit
def add_n(tensors: List[Tensor]) -> Tensor:
    if len(tensors) == 0:
        raise ValueError('`tensors` must not be empty.')
    ret = tensors[0]
    for i in range(len(tensors) - 1):
        ret = ret + tensors[i + 1]
    return ret


# ---- reduction operations ----
@jit
def reduce_sum(x: Tensor,
               axes: Optional[List[int]] = None,
               keepdims: bool = False) -> Tensor:
    if axes is None:
        if keepdims:
            return torch.sum(x).reshape([1] * len(x.shape))
        else:
            return torch.sum(x)
    else:
        return torch.sum(x, dim=axes, keepdim=keepdims)


@jit
def reduce_mean(x: Tensor,
                axes: Optional[List[int]] = None,
                keepdims: bool = False) -> Tensor:
    if axes is None:
        if keepdims:
            return torch.mean(x).reshape([1] * len(x.shape))
        else:
            return torch.mean(x)
    else:
        return torch.mean(x, dim=axes, keepdim=keepdims)


@jit
def reduce_max(x: Tensor,
               axes: Optional[List[int]] = None,
               keepdims: bool = False) -> Tensor:
    if axes is None:
        if keepdims:
            return torch.max(x).reshape([1] * len(x.shape))
        else:
            return torch.max(x)
    else:
        if len(axes) == 1:
            return torch.max(x, dim=axes[0], keepdim=keepdims)[0]
        else:
            for a in axes:
                x = torch.max(x, dim=a, keepdim=True)[0]
            if not keepdims:
                x = squeeze(x, axes)
            return x


@jit
def reduce_min(x: Tensor,
               axes: Optional[List[int]] = None,
               keepdims: bool = False) -> Tensor:
    if axes is None:
        if keepdims:
            return torch.min(x).reshape([1] * len(x.shape))
        else:
            return torch.min(x)
    else:
        if len(axes) == 1:
            return torch.min(x, dim=axes[0], keepdim=keepdims)[0]
        else:
            for a in axes:
                x = torch.min(x, dim=a, keepdim=True)[0]
            if not keepdims:
                x = squeeze(x, axes)
            return x


@jit
def log_sum_exp(x: Tensor,
                axes: Optional[List[int]] = None,
                keepdims: bool = False) -> Tensor:
    if axes is None:
        axes = int_range(0, len(x.shape))
        if keepdims:
            return torch.logsumexp(x, dim=axes, keepdim=True)
        else:
            return torch.logsumexp(x, dim=axes, keepdim=False)
    else:
        return torch.logsumexp(x, dim=axes, keepdim=keepdims)


@jit
def log_mean_exp(x: Tensor,
                 axes: Optional[List[int]] = None,
                 keepdims: bool = False) -> Tensor:
    x_max_keepdims = reduce_max(x, axes=axes, keepdims=True)
    if not keepdims:
        x_max = squeeze(x_max_keepdims, axes=axes)
    else:
        x_max = x_max_keepdims
    mean_exp = reduce_mean(
        torch.exp(x - x_max_keepdims), axes=axes, keepdims=keepdims)
    return x_max + torch.log(mean_exp)


# ---- logical operations ----
@jit
def logical_not(x: Tensor) -> Tensor:
    if x.dtype != torch.bool:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, x, x.dtype))
    return x == torch.tensor(False, dtype=torch.bool)


@jit
def logical_and(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != torch.bool:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, x, x.dtype))
    if y.dtype != torch.bool:
        raise TypeError('Expected y to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, y, y.dtype))

    return x & y


@jit
def logical_or(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != torch.bool:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, x, x.dtype))
    if y.dtype != torch.bool:
        raise TypeError('Expected y to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, y, y.dtype))

    return x | y


@jit
def logical_xor(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != torch.bool:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, x, x.dtype))
    if y.dtype != torch.bool:
        raise TypeError('Expected y to be {}, got {} of type '
                        '{} instead.'.format(torch.bool, y, y.dtype))

    return x ^ y


@jit
def multiply_mask(x: Tensor, mask: Tensor) -> Tensor:
    if mask.dtype != x.dtype:
        mask = mask.to(x.dtype)
    return x * mask


where = torch.where

# @jit
# def where(condition: Tensor, x: Tensor, y: Tensor) -> Tensor:
#     return torch.where(condition, x, y)


# ---- comparison operators ----
@jit
def equal(x: Tensor, y: Tensor) -> Tensor:
    return x == y


@jit
def not_equal(x: Tensor, y: Tensor) -> Tensor:
    return x != y


@jit
def less(x: Tensor, y: Tensor) -> Tensor:
    return x < y


@jit
def less_equal(x: Tensor, y: Tensor) -> Tensor:
    return x <= y


@jit
def greater(x: Tensor, y: Tensor) -> Tensor:
    return x > y


@jit
def greater_equal(x: Tensor, y: Tensor) -> Tensor:
    return x >= y


@jit
def minimum(x: Tensor, y: Tensor) -> Tensor:
    return torch.min(x, y)


@jit
def maximum(x: Tensor, y: Tensor) -> Tensor:
    return torch.max(x, y)


@jit
def clip(x: Tensor, x_min: float, x_max: float) -> Tensor:
    return torch.clamp(x, x_min, x_max)


# ---- gradient utilities ----
if settings.disable_jit:
    def grad(outputs: List[Tensor],
             inputs: List[Tensor],
             grad_outputs: Optional[List[Optional[Tensor]]] = None,
             keep_graph: Optional[bool] = None,
             create_graph: bool = False,
             allow_unused: bool = False) -> List[Optional[Tensor]]:
        return torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            retain_graph=keep_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )


    def is_null_grad(origin: Tensor, gradient: Optional[Tensor]) -> bool:
        return gradient is None
else:
    @jit
    def grad(outputs: List[Tensor],
             inputs: List[Tensor],
             grad_outputs: Optional[List[Optional[Tensor]]] = None,
             keep_graph: Optional[bool] = None,
             create_graph: bool = False,
             allow_unused: bool = False) -> List[Tensor]:
        grad_outs = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=grad_outputs,
            keep_graph=keep_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )

        if allow_unused:
            for i in range(len(grad_outs)):
                t = grad_outs[i]
                # seems to be the dtype of `undefined tensor`, but need more
                # investigation.
                if t.dtype == 16:
                    grad_outs[i] = torch.tensor(0., dtype=inputs[i].dtype)
        else:
            for i in range(len(grad_outs)):
                if grad_outs[i].dtype == 16:
                    raise RuntimeError(
                        'One of the differentiated Tensors '
                        'appears to not have been used in the graph. '
                        'Set allow_unused=True if this is the desired '
                        'behavior.'
                    )

        return grad_outs


    @jit
    def is_null_grad(origin: Tensor, gradient: Tensor) -> bool:
        return (gradient.shape == () and
                gradient.shape != origin.shape and
                gradient == 0.)


def requires_grad(t: Tensor,
                  requires: bool = True,
                  copy: bool = False) -> Tensor:
    if copy:
        return t.clone().requires_grad_(requires)
    else:
        t.requires_grad_(requires)
        return t


@jit
def stop_grad(x: Tensor) -> Tensor:
    return x.detach()


# ---- assertion utilities ----
@jit
def is_finite(x: Tensor) -> Tensor:
    if not x.is_floating_point():
        return torch.ones_like(x).to(torch.bool)
    return (x == x) & (x.abs() != math.inf)


@jit
def is_all(condition: Tensor) -> bool:
    return bool(torch.all(condition).item())


@jit
def assert_finite(x: Tensor, message: str) -> Tensor:
    if not is_all(is_finite(x)):
        raise ValueError('Infinity or NaN value encountered: {}'.
                         format(message))
    return x
