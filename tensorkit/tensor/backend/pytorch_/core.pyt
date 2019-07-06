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
    'exp', 'log', 'log1p',
    'sin', 'cos',
] %}
{%- set BIVARIATE_OPS = [
    'add', 'sub', 'mul', 'fmod', 'pow',
] %}

__all__ = [
    # typing
    'Tensor', 'Variable', 'DType',
    'TensorTypes', 'TensorLike', 'DTypeLike', 'ShapeTuple', 'ShapeArgType',
    'AxisOrAxes', 'as_shape',

    # dtypes
    'as_dtype', 'float_x', 'iinfo', 'finfo', 'is_floating_point',

    # tensor constructors
    'as_tensor', 'register_as_tensor', 'zeros', 'ones', 'cast', 'dtype',

    # shape utils
    'shape', 'rank', 'reshape', 'repeat', 'expand', 'squeeze', 'expand_dim',
    'broadcast_shape', 'broadcast_to', 'explicit_broadcast', 'flatten_to_ndims',
    'undo_flatten_to_ndims',

    # read / assign
    'read',

    # math operators
    'div', 'truediv', 'floordiv', 'mod', 'square',
    'add_n',

    # reduce operators
    'reduce_sum', 'reduce_mean', 'log_sum_exp', 'log_mean_exp',
    'reduce_max', 'reduce_min',

    # bits operators
    'invert', 'and_', 'or_', 'xor',

    # logical operators
    'boolean', 'to_boolean',
    'logical_not', 'logical_and', 'logical_or', 'logical_xor',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'minimum', 'maximum', 'clip',

    # gradient utilities
    'stop_gradient',

    # tensor wrapper
    'TensorWrapper', 'register_tensor_wrapper_class',

    # template exported
    {{format_all_list(UNIVARIATE_OPS + BIVARIATE_OPS)}},
]

# ---- typing ----
Tensor = torch.Tensor
Variable = torch.Tensor
DType = torch.dtype


TensorTypes = Union[Tensor, Variable]
TensorLike = Union[TensorTypes, 'TensorWrapper', np.ndarray]
DTypeLike = Union[str, np.dtype, DType]
ShapeTuple = Tuple[int, ...]
ShapeArgType = Sequence[int]
AxisOrAxes = Union[int, Tuple[int, ...]]


def as_shape(s: ShapeArgType) -> ShapeTuple:
    return tuple(s)


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

    raise ValueError(f'Not a valid dtype: {type!r}')


def float_x() -> DType:
    return as_dtype(settings.float_x)


def iinfo(dtype: DTypeLike) -> torch.iinfo:
    return torch.iinfo(as_dtype(dtype))


def finfo(dtype: DTypeLike) -> torch.finfo:
    return torch.finfo(as_dtype(dtype))


def is_floating_point(dtype: DTypeLike) -> bool:
    return as_dtype(dtype).is_floating_point


def cast(x: TensorLike, dtype: DType) -> Tensor:
    return as_tensor(x).to(dtype=as_dtype(dtype))


def dtype(x: TensorLike) -> DType:
    return as_tensor(x).dtype


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
    return torch.zeros(as_shape(shape), dtype=dtype)


def ones(shape: ShapeArgType, dtype: DTypeLike = settings.float_x) -> Tensor:
    dtype = as_dtype(dtype)
    return torch.ones(as_shape(shape), dtype=dtype)


# ---- shape utils ----
def shape(x: TensorLike) -> ShapeTuple:
    return tuple(as_tensor(x).shape)


def rank(x: TensorLike) -> int:
    return as_tensor(x).numel()


def reshape(x: TensorLike, shape: ShapeArgType) -> Tensor:
    return as_tensor(x).reshape(as_shape(shape))


def repeat(x: TensorLike, repeats: Iterable[int]) -> Tensor:
    # TODO: avoid copying if not necessary
    return as_tensor(x).repeat(tuple(repeats))


def expand(x: TensorLike, desired_shape: ShapeArgType) -> Tensor:
    desired_shape = as_shape(desired_shape)
    return as_tensor(x).expand(desired_shape)


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


def expand_dim(x: TensorLike, axis: int) -> Tensor:
    return as_tensor(x).unsqueeze(axis)


def broadcast_shape(x: ShapeArgType, y: ShapeArgType) -> ShapeTuple:
    x = tuple(x)
    y = tuple(y)
    common_len = min(len(x), len(y))

    right = []
    for a, b in zip(x[-common_len:], y[-common_len:]):
        if a == 1:
            right.append(b)
        elif b == 1:
            right.append(a)
        elif a != b:
            raise ValueError(f'Shape x and y cannot broadcast against '
                             f'each other: {x} vs {y}.')
        else:
            right.append(a)

    left = x[:-common_len] or y[:-common_len]
    return left + tuple(right)


def _broadcast_to_internal(t, t_shape, out_shape):
    if len(t_shape) < len(out_shape):
        t_shape = (1,) * (len(out_shape) - len(t_shape)) + t_shape
        t = reshape(t, t_shape)
    t_repeats = tuple(b if a == 1 else 1
                      for a, b in zip(t_shape, out_shape))
    if any(s != 1 for s in t_repeats):
        t = tile(t, t_repeats)
    return t


def broadcast_to(x: TensorLike, new_shape: ShapeArgType) -> Tensor:
    x = as_tensor(x)
    x_shape = shape(x)
    new_shape = tuple(new_shape)

    # check whether or not x can broadcast to shape
    can_broadcast = len(x_shape) <= len(new_shape)
    if can_broadcast:
        for a, b in zip(reversed(x_shape), new_shape):
            if a != 1 and a != b:
                can_broadcast = False
                break
    if not can_broadcast:
        raise ValueError(f'`x` cannot be broadcast to `shape`: '
                         f'shape(x) {x_shape} vs shape {shape}')

    return _broadcast_to_internal(x, x_shape, new_shape)


def explicit_broadcast(x: TensorLike,
                       y: TensorLike
                       ) -> Tuple[Tensor, Tensor]:
    x = as_tensor(x)
    y = as_tensor(y)
    x_shape = shape(x)
    y_shape = shape(y)
    out_shape = broadcast_shape(x_shape, y_shape)
    x = _broadcast_to_internal(x, x_shape, out_shape)
    y = _broadcast_to_internal(y, y_shape, out_shape)
    return x, y


def flatten_to_ndims(x: TensorLike, ndims: int
                     ) -> Tuple[Tensor, Optional[ShapeTuple]]:
    x = as_tensor(x)
    x_shape = shape(x)
    if len(x_shape) == ndims:
        return x, None
    elif ndims < 1:
        raise ValueError(f'`ndims >= 1` must hold when `rank(x) >= 1`: '
                         f'got ndims {ndims!r}')

    x_rank = len(x_shape)
    if x_rank < ndims:
        raise ValueError(f'rank(x) < ndims: x.shape is {x_shape}, while '
                         f'ndims is {ndims}')
    else:
        front_shape, back_shape = x_shape[: -ndims], x_shape[-ndims:]
        return reshape(x, (-1,) + back_shape), front_shape


def undo_flatten_to_ndims(x: TensorLike, front_shape: Optional[ShapeTuple]
                          ) -> Tensor:
    x = as_tensor(x)
    x_shape = shape(x)

    if front_shape is None:
        return x
    else:
        x_rank = len(x_shape)
        if x_rank < 1:
            raise ValueError('Invalid input: rank(x) < 1, but front_shape is '
                             'not None.')
        return reshape(x, front_shape + x_shape[1:])


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


def add_n(tensors: Iterable[TensorLike]) -> Tensor:
    tensors = tuple(map(as_tensor, tensors))
    if not tensors:
        raise ValueError('`tensors` must not be emtpy.')
    ret = tensors[0]
    for t in tensors[1:]:
        ret += t
    return ret


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


def reduce_sum(x: TensorLike,
        axis: Optional[AxisOrAxes] = None,
        keepdims: bool = False) -> Tensor:
    return _reduce_sum_mean(torch.sum, x, axis, keepdims)


def reduce_mean(x: TensorLike,
         axis: Optional[AxisOrAxes] = None,
         keepdims: bool = False) -> Tensor:
    return _reduce_sum_mean(torch.mean, x, axis, keepdims)


def log_sum_exp(x: TensorLike,
                axis: Optional[AxisOrAxes] = None,
                keepdims: bool = False) -> Tensor:
    return _reduce_sum_mean(torch.logsumexp, x, axis, keepdims)


def log_mean_exp(x: TensorLike,
                 axis: Optional[AxisOrAxes] = None,
                 keepdims: bool = False) -> Tensor:
    axis = validate_int_tuple_arg('axis', axis, nullable=True)
    x = as_tensor(x)
    x_max_keepdims = reduce_max(x, axis=axis, keepdims=True)
    if not keepdims:
        x_max = squeeze(x_max_keepdims, axis=axis)
    else:
        x_max = x_max_keepdims
    mean_exp = reduce_mean(
        exp(x - x_max_keepdims), axis=axis, keepdims=keepdims)
    return x_max + log(mean_exp)


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


def reduce_max(x: TensorLike,
        axis: Optional[AxisOrAxes] = None,
        keepdims: bool = False) -> Tensor:
    return _reduce_max_min(torch.max, x, axis, keepdims)


def reduce_min(x: TensorLike,
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


def minimum(x: TensorLike, y: TensorLike) -> Tensor:
    return torch.min(as_tensor(x), as_tensor(y))


def maximum(x: TensorLike, y: TensorLike) -> Tensor:
    return torch.max(as_tensor(x), as_tensor(y))


def clip(x: TensorLike, x_min: float, x_max: float) -> Tensor:
    return torch.clamp(as_tensor(x), x_min, x_max)


# ---- gradient utilities ----
def stop_gradient(x: TensorLike) -> Tensor:
    return as_tensor(x).detach()


# ---- TensorWrapper ----
class TensorWrapper(object):

    @property
    def tensor(self) -> 'Tensor':
        raise NotImplementedError()

    def as_tensor(self, dtype: 'DType' = None) -> 'Tensor':
        t = self.tensor
        if dtype is not None:
            t = cast(t, dtype=dtype)
        return t

    # mimic `tf.Tensor` interface
    def __dir__(self):
        ret = list(set(dir(self.tensor) + list(object.__dir__(self))))
        return ret

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __setattr__(self, name, value):
        if name.startswith('_self_'):
            object.__setattr__(self, name, value)
        elif hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.tensor, name, value)

    def __delattr__(self, name):
        if name.startswith('_self_'):
            object.__delattr__(self, name)
        elif hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.tensor, name)

    def __iter__(self):
        return iter(self.tensor)

    def __bool__(self):
        return bool(self.tensor)

    # overloading arithmetic operations
    def __abs__(self):
        return abs(self.tensor)

    def __neg__(self):
        return neg(self.tensor)

    def __add__(self, other):
        return add(self.tensor, other)

    def __radd__(self, other):
        return add(other, self.tensor)

    def __sub__(self, other):
        return sub(self.tensor, other)

    def __rsub__(self, other):
        return sub(other, self.tensor)

    def __mul__(self, other):
        return mul(self.tensor, other)

    def __rmul__(self, other):
        return mul(other, self.tensor)

    def __div__(self, other):
        return div(self.tensor, other)

    def __rdiv__(self, other):
        return div(other, self.tensor)

    def __truediv__(self, other):
        return truediv(self.tensor, other)

    def __rtruediv__(self, other):
        return truediv(other, self.tensor)

    def __floordiv__(self, other):
        return floordiv(self.tensor, other)

    def __rfloordiv__(self, other):
        return floordiv(other, self.tensor)

    def __mod__(self, other):
        return mod(self.tensor, other)

    def __rmod__(self, other):
        return mod(other, self.tensor)

    def __pow__(self, other):
        return pow(self.tensor, other)

    def __rpow__(self, other):
        return pow(other, self.tensor)

    # logical operations
    def __invert__(self):
        return invert(self.tensor)

    def __and__(self, other):
        return and_(self.tensor, other)

    def __rand__(self, other):
        return and_(other, self.tensor)

    def __or__(self, other):
        return or_(self.tensor, other)

    def __ror__(self, other):
        return or_(other, self.tensor)

    def __xor__(self, other):
        return xor(self.tensor, other)

    def __rxor__(self, other):
        return xor(other, self.tensor)

    # boolean operations
    def __lt__(self, other):
        return less(self.tensor, other)

    def __le__(self, other):
        return less_equal(self.tensor, other)

    def __gt__(self, other):
        return greater(self.tensor, other)

    def __ge__(self, other):
        return greater_equal(self.tensor, other)

    # slicing and indexing
    def __getitem__(self, item):
        return (as_tensor(self.tensor))[item]


def register_tensor_wrapper_class(cls: Type[TensorWrapper]):
    # nothing should be done, all is okay
    pass


register_as_tensor(
    TensorWrapper,
    (lambda t, dtype=None: t.as_tensor(dtype))
)
