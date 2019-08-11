from typing import *

import numpy as np
import torch
import torch.nn.functional
from torch.jit import script

from ....settings_ import settings

__all__ = [
    # typing
    'Tensor', 'Variable',

    # jit
    'jit',

    # dtypes
    'float_x', 'boolean', 'index_dtype', 'dtype', 'is_floating_point',

    # tensor constructors
    'as_tensor', 'as_boolean', 'from_int', 'from_float',
    'from_ints_1d', 'from_floats_1d',
    'from_ints_2d', 'from_floats_2d',
    'from_ints_3d', 'from_floats_3d',
    'from_ints_4d', 'from_floats_4d',
    'zeros', 'ones', 'arange', 'cast',

    # shape utils
    'shape', 'rank', 'reshape', 'repeat', 'expand', 'squeeze', 'expand_dim',
    'broadcast_shape', 'broadcast_to', 'explicit_broadcast', 'flatten_to_ndims',
    'unflatten_from_ndims', 'index_select',

    # read / assign
    'to_numpy', 'to_numpy_bool',

    # math operators
    'truediv', 'floordiv', 'mod', 'square', 'add_n',

    # reduce operators
    'reduce_sum', 'reduce_mean', 'log_sum_exp', 'log_mean_exp',
    'reduce_max', 'reduce_min',

    # logical operators
    'logical_not', 'logical_and', 'logical_or', 'logical_xor',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'minimum', 'maximum', 'clip',

    # gradient utilities
    'requires_grad', 'clear_grad', 'back_prop', 'grad', 'detach',

    # activation functions
    'relu', 'leaky_relu', 'sigmoid', 'softmax', 'log_softmax',

    # cross entropy functions
    'binary_cross_entropy_with_logits', 'cross_entropy_with_logits',
    'sparse_cross_entropy_with_logits',

    # tensor transformations
    'one_hot',

    # random utilities
    'random_seed', 'random_normal', 'randn',
    'bernoulli', 'categorical',

    # template exported
    'int8', 'uint8', 'int16', 'int32', 'int64', 'float16', 'float32',
    'float64',
    'abs', 'neg', 'exp', 'log', 'log1p', 'sin', 'cos', 'add', 'sub', 'mul',
    'div', 'fmod', 'pow',
]


# ---- typing ----
# true types
Tensor = torch.Tensor
Variable = torch.Tensor


# ---- jit ----
def jit(fn):
    if not settings.disable_jit:
        fn = script(fn)
    return fn


# ---- jit specified utilities ----
if settings.disable_jit:
    def _int_range(n: int) -> List[int]:
        return list(range(n))
else:
    @jit
    def _int_range(n: int) -> List[int]:
        ret = torch.jit.annotate(List[int], [])
        for i in range(n):
            ret.append(i)
        return ret


# ---- dtypes ----
int8 = 'int8'
uint8 = 'uint8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
float16 = 'float16'
float32 = 'float32'
float64 = 'float64'
boolean = 'uint8'
index_dtype = 'int64'


def float_x() -> str:
    return settings.float_x


@jit
def cast(x: Tensor, dtype: str) -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    dtype = _mapper[dtype]
    if dtype != x.dtype:
        x = x.to(dtype=dtype)
    return x


@jit
def dtype(x: Tensor) -> str:
    _mapper = {
        torch.int8: 'int8',
        torch.uint8: 'uint8',
        torch.int16: 'int16',
        torch.int32: 'int32',
        torch.int64: 'int64',
        torch.float16: 'float16',
        torch.float32: 'float32',
        torch.float64: 'float64',
    }
    return _mapper[x.dtype]


@jit
def is_floating_point(x: Tensor) -> bool:
    return x.is_floating_point()


# ---- tensor constructors ----
def as_tensor(data, dtype: Optional[str] = None) -> Tensor:
    if dtype is not None:
        _mapper = {
            'int8': torch.int8,
            'uint8': torch.uint8,
            'int16': torch.int16,
            'int32': torch.int32,
            'int64': torch.int64,
            'float16': torch.float16,
            'float32': torch.float32,
            'float64': torch.float64,
        }
        dtype = _mapper[dtype]
    if isinstance(data, Tensor):
        if dtype is not None:
            data = torch.as_tensor(data, dtype=dtype)
        return data
    else:
        return torch.as_tensor(data, dtype=dtype)


def as_boolean(data) -> Tensor:
    return as_tensor(data).to(torch.bool).to(torch.uint8)


@jit
def from_int(data: int,
             dtype: str = 'int32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_ints_1d(data: List[int],
                 dtype: str = 'int32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_ints_2d(data: List[List[int]],
                 dtype: str = 'int32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_ints_3d(data: List[List[List[int]]],
                 dtype: str = 'int32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_ints_4d(data: List[List[List[List[int]]]],
                 dtype: str = 'int32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])



@jit
def from_float(data: float,
               dtype: str = 'float32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_floats_1d(data: List[float],
                   dtype: str = 'float32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_floats_2d(data: List[List[float]],
                   dtype: str = 'float32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_floats_3d(data: List[List[List[float]]],
                   dtype: str = 'float32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def from_floats_4d(data: List[List[List[List[float]]]],
                   dtype: str = 'float32') -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.tensor(data, dtype=_mapper[dtype])


@jit
def zeros(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.zeros(shape, dtype=_mapper[dtype])


@jit
def ones(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.ones(shape, dtype=_mapper[dtype])


@jit
def _arange_1(n: int,
              step: int = 1,
              dtype: str = int32) -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    dtype = _mapper[dtype]
    return torch.arange(0, n, step, dtype=dtype)


@jit
def _arange_2(start: int,
              end: int,
              step: int = 1,
              dtype: str = int32) -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    dtype = _mapper[dtype]
    return torch.arange(start, end, step, dtype=dtype)


@jit
def arange(start_or_end: int,
           end: Optional[int] = None,
           step: int = 1,
           dtype: str = int32) -> Tensor:
    if end is None:
        return _arange_1(start_or_end, step, dtype)
    else:
        return _arange_2(start_or_end, end, step, dtype)


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
        left = x[:-common_len]
    else:
        left = y[:-common_len]
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


# ---- read / assign ----
def to_numpy(x: Tensor) -> np.ndarray:
    if not isinstance(x, Tensor):
        raise TypeError(f'`x` is not a Tensor: got {x!r}')
    return x.data.numpy()


def to_numpy_bool(x: Tensor) -> np.ndarray:
    return to_numpy(x).astype(np.bool)


# ---- univariate element-wise math operations ----

@jit
def abs(x: Tensor) -> Tensor:
    return torch.abs(x)


@jit
def neg(x: Tensor) -> Tensor:
    return torch.neg(x)


@jit
def exp(x: Tensor) -> Tensor:
    return torch.exp(x)


@jit
def log(x: Tensor) -> Tensor:
    return torch.log(x)


@jit
def log1p(x: Tensor) -> Tensor:
    return torch.log1p(x)


@jit
def sin(x: Tensor) -> Tensor:
    return torch.sin(x)


@jit
def cos(x: Tensor) -> Tensor:
    return torch.cos(x)


@jit
def square(x: Tensor) -> Tensor:
    return x ** 2


# ---- bivariate element-wise math operations ----

@jit
def add(x: Tensor, y: Tensor) -> Tensor:
    return torch.add(x, y)


@jit
def sub(x: Tensor, y: Tensor) -> Tensor:
    return torch.sub(x, y)


@jit
def mul(x: Tensor, y: Tensor) -> Tensor:
    return torch.mul(x, y)


@jit
def div(x: Tensor, y: Tensor) -> Tensor:
    return torch.div(x, y)


@jit
def fmod(x: Tensor, y: Tensor) -> Tensor:
    return torch.fmod(x, y)


@jit
def pow(x: Tensor, y: Tensor) -> Tensor:
    return torch.pow(x, y)



@jit
def floordiv(x: Tensor, y: Tensor) -> Tensor:
    ret = torch.div(x, y)
    if ret.is_floating_point():
        ret = ret.floor()
    return ret


@jit
def truediv(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != y.dtype:
        raise TypeError('x and y must have the same dtype, got '
                        '{} != {}'.format(x.dtype, y.dtype))

    dtype = x.dtype
    if not x.is_floating_point():
        if dtype == torch.uint8 or dtype == torch.int16:
            x = x.to(torch.float32)
            y = y.to(torch.float32)
        else:
            x = x.to(torch.float64)
            y = y.to(torch.float64)

    return x / y


mod = fmod


# ---- sequential math element-wise operations ----
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
        axes = _int_range(len(x.shape))
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
def NOT_A_FUNCTION():
    pass


def logical_not(x: Tensor) -> Tensor:
    if x.dtype != torch.uint8:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, x, x.dtype))
    return ~x


@jit
def logical_and(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != torch.uint8:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, x, x.dtype))
    if y.dtype != torch.uint8:
        raise TypeError('Expected y to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, y, y.dtype))

    return x & y


@jit
def logical_or(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != torch.uint8:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, x, x.dtype))
    if y.dtype != torch.uint8:
        raise TypeError('Expected y to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, y, y.dtype))

    return x | y


@jit
def logical_xor(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != torch.uint8:
        raise TypeError('Expected x to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, x, x.dtype))
    if y.dtype != torch.uint8:
        raise TypeError('Expected y to be {}, got {} of type '
                        '{} instead.'.format(torch.uint8, y, y.dtype))

    return x ^ y


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
def requires_grad(x: Tensor) -> Tensor:
    return x.requires_grad_(True)


def clear_grad(x: Tensor) -> Tensor:
    x.grad.data.zero_()
    return x


def back_prop(x: Tensor) -> Tensor:
    x.backward()
    return x


def grad(x: Tensor) -> Optional[Tensor]:
    return x.grad


@jit
def detach(x: Tensor) -> Tensor:
    return x.detach()


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
                                     reduction: str = 'none',
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
                              reduction: str = 'none',
                              negative: bool = False) -> Tensor:
    logits_shape = list(logits.shape)
    labels_shape = list(labels.shape)

    if len(logits_shape) < 2 or len(labels_shape) < 1:
        raise ValueError('`logits` must be at least 2d, and `labels` must '
                         'be at least 1d: logits.shape is {}, while '
                         'labels.shape is {}.'.
                         format(logits.shape, labels.shape))

    if logits_shape[:-1] != labels_shape:
        b_shape = broadcast_shape(logits_shape[:-1], labels_shape)
        logits = broadcast_to(logits, b_shape + logits_shape[-1:])
        labels = broadcast_to(labels, b_shape)

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


# ---- tensor transformations ----
@jit
def one_hot(x: Tensor,
            n_classes: int,
            dtype: str = index_dtype) -> Tensor:
    ret = torch.nn.functional.one_hot(x, n_classes)
    ret = cast(ret, dtype)
    return ret


# ---- random utilities ----
def random_seed(seed: int):
    torch.manual_seed(seed)


@jit
def random_normal(mean: Tensor, std: Tensor) -> Tensor:
    mean, std = explicit_broadcast(mean, std)
    return torch.normal(mean=mean, std=std)


@jit
def randn(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    return torch.randn(shape, dtype=_mapper[dtype])


@jit
def _bernoulli_sub(probs: Tensor,
                   n_samples: Optional[int],
                   dtype: str = int32) -> Tensor:
    # validate arguments
    if n_samples is not None and n_samples < 1:
        raise ValueError('`n_samples` must be at least 1: got {}'.
                         format(n_samples))

    _mapper = {
        'int8': torch.int8,
        'uint8': torch.uint8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
    }
    dtype = _mapper[dtype]

    # do sample
    probs = probs.detach()
    sample_shape = probs.shape
    if n_samples is not None:
        sample_shape = (n_samples,) + sample_shape
        probs = probs.unsqueeze(dim=0).expand(sample_shape)
    out = torch.zeros(sample_shape, dtype=dtype)
    return torch.bernoulli(probs, out=out).detach()


@jit
def bernoulli(probs: Optional[Tensor] = None,
              logits: Optional[Tensor] = None,
              n_samples: Optional[int] = None,
              dtype: str = int32) -> Tensor:
    if (probs is None and logits is None) or \
            (probs is not None and logits is not None):
        raise ValueError(
            'Either `logits` or `probs` must be specified, but not both')

    if probs is not None:
        return _bernoulli_sub(probs=probs, n_samples=n_samples, dtype=dtype)
    elif logits is not None:
        probs = torch.sigmoid(logits)
        return _bernoulli_sub(probs=probs, n_samples=n_samples, dtype=dtype)
    else:
        # This branch should never touch, but PyTorch JIT engine cannot
        # recognize this.  So we add this branch.
        return torch.tensor(0)  # pragma: no cover


@jit
def _categorical_sub(probs: Tensor,
                     n_samples: Optional[int],
                     dtype: str) -> Tensor:
    # validate arguments
    if n_samples is not None and n_samples < 1:
        raise ValueError('`n_samples` must be at least 1: got {}'.
                         format(n_samples))

    probs_rank = len(probs.shape)
    if probs_rank < 1:
        raise ValueError(
            'The rank of `logits` or `probs` must be at least 1: '
            'got {}'.format(probs_rank)
        )

    # do sample
    probs = probs.detach()
    if probs_rank > 2:
        probs, front_shape = flatten_to_ndims(probs, 2)
    else:
        probs, front_shape = probs, None

    if n_samples is None:
        ret = torch.multinomial(probs, 1, replacement=True)
        ret = torch.squeeze(ret, -1)
        if front_shape is not None:
            ret = unflatten_from_ndims(ret, front_shape)
    else:
        ret = torch.multinomial(probs, n_samples, replacement=True)
        if front_shape is not None:
            ret = unflatten_from_ndims(ret, front_shape)
        ret = ret.permute([-1] + _int_range(len(ret.shape) - 1))

    ret = cast(ret, dtype)
    return ret.detach()


@jit
def categorical(probs: Optional[Tensor] = None,
                logits: Optional[Tensor] = None,
                n_samples: Optional[int] = None,
                dtype: str = index_dtype) -> Tensor:
    if (probs is None and logits is None) or \
            (probs is not None and logits is not None):
        raise ValueError(
            'Either `logits` or `probs` must be specified, but not both')

    if probs is not None:
        return _categorical_sub(probs=probs, n_samples=n_samples, dtype=dtype)
    elif logits is not None:
        probs = torch.softmax(logits, dim=-1)
        return _categorical_sub(probs=probs, n_samples=n_samples, dtype=dtype)
    else:
        # This branch should never touch, but PyTorch JIT engine cannot
        # recognize this.  So we add this branch.
        return torch.tensor(0)  # pragma: no cover
