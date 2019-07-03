import operator
from typing import *

import numpy as np
import torch
from mltk.utils import InheritanceDict

from ....settings_ import settings
from ....utils import validate_int_tuple_arg
from .dtypes import _DTYPES, _NUMPY_DTYPES


{%- set UNIVARIATE_OPS = [
    'abs', 'neg',
    'exp', 'log',
    'sin', 'cos',
] %}
{%- set BIVARIATE_OPS = [
    'add', 'sub', 'mul', 'fmod', 'pow',
] %}

__all__ = [
    # typing
    'Tensor', 'Variable', 'DType',
    'TensorTypes', 'TensorLike', 'DTypeLike', 'ShapeTuple', 'ShapeArgType',
    'AxisOrAxes',

    # tensor constructors
    'as_tensor', 'register_as_tensor', 'zeros', 'ones',

    # type utils
    'DType',
    'as_dtype', 'is_floating_point',
    'cast', 'dtype',

    # shape utils
    'shape', 'rank', 'reshape', 'tile', 'squeeze',

    # read / assign
    'read',

    # math operators
    'div', 'truediv', 'floordiv', 'mod', 'square',

    # reduce operators
    'sum', 'mean', 'max', 'min',

    # bits operators
    'invert', 'and_', 'or_', 'xor',

    # logical operators
    'boolean', 'to_boolean',
    'logical_not', 'logical_and', 'logical_or', 'logical_xor',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',

    # gradient utilities
    'stop_gradient',

    # template exported
    {{format_all_list(UNIVARIATE_OPS + BIVARIATE_OPS)}},
]

# ---- typing ----
Tensor = torch.Tensor
Variable = torch.Tensor
DType = torch.dtype

TensorTypes = Union[Tensor, Variable]
TensorLike = Union[TensorTypes, 'TensorWrapper']
DTypeLike = Union[str, np.dtype, DType]
ShapeTuple = Tuple[int, ...]
ShapeArgType = Sequence[int]
AxisOrAxes = Union[int, Tuple[int, ...]]


# ---- dtypes ----
def as_dtype(dtype: DTypeLike) -> DType:
    try:
        if isinstance(dtype, str):
            return _DTYPES[dtype]
        elif isinstance(dtype, np.dtype):
            return _NUMPY_DTYPES[dtype]
        elif isinstance(dtype, DType):
            return dtype
    except KeyError:
        pass

    raise ValueError('Not a valid dtype: {type!r}')


def iinfo(dtype: DTypeLike) -> torch.iinfo:
    return torch.iinfo(as_dtype(dtype))


def finfo(dtype: DTypeLike) -> torch.finfo:
    return torch.finfo(as_dtype(dtype))


def is_floating_point(dtype: DTypeLike) -> bool:
    return as_dtype(dtype).is_floating_point


# ---- tensor constructors ----
AsTensorFunc = Callable[[Any, Optional[DType]], Tensor]
_as_tensor_convertors: InheritanceDict[AsTensorFunc] = InheritanceDict()


def as_tensor(data, dtype: Optional[DTypeLike] = None):
    if isinstance(data, Tensor) and dtype is None:
        return data

    if dtype is not None:
        dtype = as_dtype(dtype)

    try:
        convertor = _as_tensor_convertors[type(data)]
    except KeyError:
        convertor = lambda t, dtype: torch.as_tensor(t, dtype=dtype)

    return convertor(data, dtype)


def register_as_tensor(type_: type, convertor: AsTensorFunc):
    _as_tensor_convertors[type_] = convertor


def zeros(shape: ShapeArgType, dtype: DTypeLike = settings.float_x) -> Tensor:
    dtype = as_dtype(dtype)
    return torch.zeros(tuple(shape), dtype=dtype)


def ones(shape: ShapeArgType, dtype: DTypeLike = settings.float_x) -> Tensor:
    dtype = as_dtype(dtype)
    return torch.ones(tuple(shape), dtype=dtype)


# ---- type utils ----
def cast(x: TensorLike, dtype: DType) -> Tensor:
    return as_tensor(x).to(dtype=as_dtype(dtype))


def dtype(x: TensorLike) -> DType:
    return as_tensor(x).dtype


# ---- shape utils ----
def shape(x: TensorLike) -> ShapeTuple:
    return tuple(as_tensor(x).shape)


def rank(x: TensorLike) -> int:
    return len(shape(x))


def reshape(x: TensorLike, shape: ShapeArgType) -> Tensor:
    return as_tensor(x).reshape(tuple(shape))


def tile(x: TensorLike, repeats: Iterable[int]) -> Tensor:
    return as_tensor(x).repeat(tuple(repeats))


def squeeze(x: TensorLike, axis: Optional[AxisOrAxes] = None) -> Tensor:
    axis = validate_int_tuple_arg('axis', axis, nullable=True)
    x = as_tensor(x)

    if axis is not None:
        if len(axis) == 1:
            x = torch.squeeze(x, axis)
        else:
            old_shape = shape(x)
            new_shape_mask = [True] * len(old_shape)
            for a in axis:
                if old_shape[a] == 1:
                    new_shape_mask[a] = False
                else:
                    raise ValueError(f'Axis {a} cannot be squeezed, since its '
                                     f'size is {old_shape[a]} != 1')
            new_shape = [s for i, s in enumerate(old_shape)
                         if new_shape_mask[i]]
            x = reshape(x, tuple(new_shape))
    else:
        x = torch.squeeze(x)

    return x


# ---- read / assign ----
def read(x: TensorLike) -> Tensor:
    x = as_tensor(x)
    return x.data


# ---- math operators ----
{%- for name in UNIVARIATE_OPS %}

def {{ name }}(x: TensorLike) -> Tensor:
    return torch.{{ name }}(as_tensor(x))
{% endfor %}
{%- for name in BIVARIATE_OPS %}

def {{ name }}(x: TensorLike, y: TensorLike) -> Tensor:
    return torch.{{ name }}(as_tensor(x), as_tensor(y))
{% endfor %}

def floordiv(x: TensorLike, y: TensorLike) -> Tensor:
    ret = torch.div(as_tensor(x), as_tensor(y))
    if ret.is_floating_point():
        ret = ret.floor()
    return ret


def truediv(x: TensorLike, y: TensorLike) -> Tensor:
    x = as_tensor(x)
    y = as_tensor(y)

    if x.dtype != y.dtype:
        raise TypeError(f'x and y must have the same dtype, got '
                        f'{x.dtype} != {y.dtype}')

    dtype = x.dtype
    if not is_floating_point(dtype):
        if dtype in (torch.uint8, torch.int16):
            x = x.to(torch.float32)
            y = x.to(torch.float32)
        else:
            x = x.to(torch.float64)
            y = y.to(torch.float64)

    return x / y


div = truediv
mod = fmod


def square(x: TensorLike) -> Tensor:
    x = as_tensor(x)
    return x ** 2


# ---- reduction operations ----
def _reduce_sum_mean(op,
                     x: TensorLike,
                     axis: Optional[AxisOrAxes],
                     keepdims: bool):
    x = as_tensor(x)

    if axis is None:
        if keepdims:
            return reshape(op(x), [1] * rank(x))
        else:
            return op(x)
    else:
        return op(x, dim=axis, keepdim=keepdims)


def sum(x: TensorLike,
        axis: Optional[AxisOrAxes] = None,
        keepdims: bool = False) -> Tensor:
    return _reduce_sum_mean(torch.sum, x, axis, keepdims)


def mean(x: TensorLike,
         axis: Optional[AxisOrAxes] = None,
         keepdims: bool = False) -> Tensor:
    return _reduce_sum_mean(torch.mean, x, axis, keepdims)


def _reduce_max_min(op,
                    x: TensorLike,
                    axis: Optional[AxisOrAxes],
                    keepdims: bool):
    axis = validate_int_tuple_arg('axis', axis, nullable=True)
    x = as_tensor(x)

    if axis is None:
        if keepdims:
            x = reshape(op(x), [1] * rank(x))
        else:
            x = op(x)
    else:
        if axis:
            if len(axis) == 1:
                x = op(x, dim=axis[0], keepdim=keepdims)[0]
            else:
                for a in axis:
                    x = op(x, dim=a, keepdim=True)[0]
                if not keepdims:
                    x = squeeze(x, axis)
    return x


def max(x: TensorLike,
        axis: Optional[AxisOrAxes] = None,
        keepdims: bool = False) -> Tensor:
    return _reduce_max_min(torch.max, x, axis, keepdims)


def min(x: TensorLike,
        axis: Optional[AxisOrAxes] = None,
        keepdims: bool = False) -> Tensor:
    return _reduce_max_min(torch.min, x, axis, keepdims)


# ---- bits operators ----
def invert(x: TensorLike) -> Tensor:
    return ~as_tensor(x)


def and_(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) & as_tensor(y)


def or_(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) | as_tensor(y)


def xor(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) ^ as_tensor(y)


# ---- logical operators ----
boolean = torch.uint8


def to_boolean(x: TensorLike) -> Tensor:
    return as_tensor(x).to(torch.bool).to(boolean)


def logical_not(x: TensorLike) -> Tensor:
    x = as_tensor(x)
    if x.dtype != boolean:
        raise TypeError(f'Expected x to be {boolean}, got {x!r} of type '
                        f'{x.dtype} instead.')
    return ~x


def _logical_bi_op(op, x: TensorLike, y: TensorLike) -> Tensor:
    x = as_tensor(x)
    y = as_tensor(y)

    if x.dtype != boolean:
        raise TypeError(f'Expected x to be {boolean}, got {x!r} of type '
                        f'{x.dtype} instead.')
    if y.dtype != boolean:
        raise TypeError(f'Expected y to be {boolean}, got {y!r} of type '
                        f'{y.dtype} instead.')

    return op(x, y)


def logical_and(x: TensorLike, y: TensorLike) -> Tensor:
    return _logical_bi_op(operator.and_, x, y)


def logical_or(x: TensorLike, y: TensorLike) -> Tensor:
    return _logical_bi_op(operator.or_, x, y)


def logical_xor(x: TensorLike, y: TensorLike) -> Tensor:
    return _logical_bi_op(operator.xor, x, y)


# ---- comparison operators ----
def equal(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) == as_tensor(y)


def not_equal(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) != as_tensor(y)


def less(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) < as_tensor(y)


def less_equal(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) <= as_tensor(y)


def greater(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) > as_tensor(y)


def greater_equal(x: TensorLike, y: TensorLike) -> Tensor:
    return as_tensor(x) >= as_tensor(y)


# ---- gradient utilities ----
def stop_gradient(x: Tensor) -> Tensor:
    return x.detach()


# ---- TensorWrapper ----
from ...tensor_wrapper import TensorWrapper

register_as_tensor(
    TensorWrapper,
    (lambda t, dtype=None: t.as_tensor(dtype))
)
