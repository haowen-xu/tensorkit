import operator
from typing import *

import numpy as np
import torch
from mltk.utils import InheritanceDict
from torch.jit import script

from ....settings_ import settings
from .dtypes import _DTYPES, _NUMPY_DTYPES, int32
from .typing import *

{%- set UNIVARIATE_OPS = [
    'abs', 'neg',
    'exp', 'log', 'log1p',
    'sin', 'cos',
] %}
{%- set BIVARIATE_OPS = [
    'add', 'sub', 'mul', 'fmod', 'pow',
] %}

__all__ = [
    # jit
    'jit',

    # typing
    'Tensor', 'Variable', 'DType', 'Shape', 'as_shape',

    # dtypes
    'as_dtype', 'float_x', 'iinfo', 'finfo', 'is_floating_point',

    # tensor constructors
    'as_tensor', 'register_as_tensor', 'zeros', 'ones', 'arange',
    'cast', 'dtype',

    # shape utils
    'shape', 'rank', 'reshape', 'repeat', 'expand', 'squeeze', 'expand_dim',
    'broadcast_shape', 'broadcast_to', 'explicit_broadcast', 'flatten_to_ndims',
    'unflatten_from_ndims', 'index_select',

    # read / assign
    'to_numpy', 'to_numpy_bool',

    # math operators
    'div', 'truediv', 'floordiv', 'mod', 'square',
    'add_n',

    # reduce operators
    'reduce_sum', 'reduce_mean', 'log_sum_exp', 'log_mean_exp',
    'reduce_max', 'reduce_min',

    # logical operators
    'boolean', 'to_boolean',
    'logical_not', 'logical_and', 'logical_or', 'logical_xor',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'minimum', 'maximum', 'clip',

    # gradient utilities
    'requires_grad', 'clear_grad', 'back_prop', 'grad', 'detach',

    # tensor wrapper
    'TensorWrapper', 'register_tensor_wrapper_class',

    # template exported
    {{format_all_list(UNIVARIATE_OPS + BIVARIATE_OPS)}},
]


# ---- jit ----
def jit(fn):
    if not settings.disable_jit:
        fn = script(fn)
    return fn


# ---- typing ----
def as_shape(s: ShapeLike) -> Shape:
    return Shape(s)


# ---- dtypes ----
def as_dtype(dtype: DTypeLike) -> DType:
    if isinstance(dtype, DType):
        return dtype
    elif dtype in _DTYPES:
        return _DTYPES[dtype]
    elif dtype in _NUMPY_DTYPES:
        return _NUMPY_DTYPES[dtype]
    else:
        raise ValueError(f'Not a valid dtype: {dtype!r}')


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


def zeros(shape: ShapeLike, dtype: DTypeLike = settings.float_x) -> Tensor:
    dtype = as_dtype(dtype)
    return torch.zeros(as_shape(shape), dtype=dtype)


def ones(shape: ShapeLike, dtype: DTypeLike = settings.float_x) -> Tensor:
    dtype = as_dtype(dtype)
    return torch.ones(as_shape(shape), dtype=dtype)


def arange(start_or_end: Number,
           end: Optional[Number] = None,
           step: Optional[Number] = None,
           *,
           dtype: DTypeLike = int32) -> Tensor:
    if end is None and step is None:
        return torch.arange(start_or_end, dtype=dtype)
    elif step is None:
        return torch.arange(start_or_end, end, dtype=dtype)
    elif end is None:
        return torch.arange(0, start_or_end, step, dtype=dtype)
    else:
        return torch.arange(start_or_end, end, step, dtype=dtype)


# ---- shape utils ----
def shape(x: TensorLike) -> Shape:
    return as_tensor(x).shape


def rank(x: TensorLike) -> int:
    return len(as_tensor(x).shape)


def reshape(x: TensorLike, shape: ShapeLike) -> Tensor:
    return as_tensor(x).reshape(as_shape(shape))


@jit
def _repeat(x: Tensor, repeats: List[int]) -> Tensor:
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


def repeat(x: TensorLike, repeats: Iterable[int]) -> Tensor:
    return _repeat(as_tensor(x), list(repeats))


def expand(x: TensorLike, desired_shape: ShapeLike) -> Tensor:
    desired_shape = as_shape(desired_shape)
    return as_tensor(x).expand(desired_shape)


@jit
def _squeeze_slow_branch(x: Tensor, axis: List[int]) -> Tensor:
    old_shape = x.shape
    new_shape_mask = [True] * len(old_shape)
    for a in axis:
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


def squeeze(x: TensorLike, axis: Optional[AxisOrAxes] = None) -> Tensor:
    if axis is not None:
        axis = list(axis) if isinstance(axis, (tuple, list)) else [axis]
    x = as_tensor(x)

    if axis is not None:
        if len(axis) == 1:
            return torch.squeeze(x, axis[0])
        else:
            return _squeeze_slow_branch(x, list(axis))
    else:
        return torch.squeeze(x)


def expand_dim(x: TensorLike, axis: int) -> Tensor:
    return as_tensor(x).unsqueeze(axis)


@jit
def _broadcast_shape(x: List[int], y: List[int]) -> List[int]:
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


def broadcast_shape(x: ShapeLike, y: ShapeLike) -> Shape:
    return Shape(_broadcast_shape(list(x), list(y)))


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
def _broadcast_to(x: Tensor, new_shape: List[int]) -> Tensor:
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


def broadcast_to(x: TensorLike, new_shape: ShapeLike) -> Tensor:
    return _broadcast_to(as_tensor(x), list(new_shape))


def explicit_broadcast(x: TensorLike,
                       y: TensorLike
                       ) -> Tuple[Tensor, Tensor]:
    x = as_tensor(x)
    y = as_tensor(y)
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    out_shape = _broadcast_shape(x_shape, y_shape)
    x = _broadcast_to_sub(x, x_shape, out_shape)
    y = _broadcast_to_sub(y, y_shape, out_shape)
    return x, y


@jit
def _flatten_to_ndims(x: Tensor, ndims: int) -> Tuple[Tensor, List[int]]:
    if ndims < 1:
        raise ValueError('`ndims >= 1` must hold when `rank(x) >= 1`: '
                         'got ndims {}'.format(ndims))
    if len(x.shape) < ndims:
        raise ValueError('rank(x) < ndims: x.shape is {}, while '
                         'ndims is {}'.format(x.shape, ndims))

    if ndims == 1:
        front_shape = list(x.shape)
        return x.reshape((-1,)), front_shape
    else:
        offset = ndims - 1
        front_shape, back_shape = x.shape[: -offset], x.shape[-offset:]
        return x.reshape([-1] + list(back_shape)), front_shape


def flatten_to_ndims(x: TensorLike, ndims: int
                     ) -> Tuple[Tensor, Optional[Shape]]:
    x = as_tensor(x)
    if len(x.shape) == ndims:
        return x, None
    else:
        out, front_shape = _flatten_to_ndims(x, ndims)
        return out, Shape(front_shape)


def unflatten_from_ndims(x: TensorLike, front_shape: Optional[Shape]
                          ) -> Tensor:
    x = as_tensor(x)
    x_shape = x.shape

    if front_shape is None:
        return x
    else:
        x_rank = len(x_shape)
        if x_rank < 1:
            raise ValueError('Invalid input: rank(x) < 1, but front_shape is '
                             'not None.')
        return x.reshape(front_shape + x_shape[1:])


# ---- split / join / indexing / gathering ----
@jit
def _index_select(x: Tensor, indices: Tensor, axis: int) -> Tensor:
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


def index_select(x: TensorLike, indices: TensorLike, axis: int = 0) -> Tensor:
    return _index_select(as_tensor(x), as_tensor(indices), axis)


# ---- read / assign ----
def to_numpy(x: TensorLike) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    x = as_tensor(x)
    return x.data.numpy()


def to_numpy_bool(x: TensorLike) -> np.ndarray:
    return to_numpy(x).astype(np.bool)


# ---- univariate element-wise math operations ----
{%- for name in UNIVARIATE_OPS %}

def {{ name }}(x: TensorLike) -> Tensor:
    return torch.{{ name }}(as_tensor(x))
{% endfor %}

def square(x: TensorLike) -> Tensor:
    x = as_tensor(x)
    return x ** 2


# ---- bivariate element-wise math operations ----
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
            y = y.to(torch.float32)
        else:
            x = x.to(torch.float64)
            y = y.to(torch.float64)

    return x / y


div = truediv
mod = fmod


# ---- sequential math element-wise operations ----
@jit
def _add_n(tensors):
    # type: (List[Tensor]) -> Tensor
    if len(tensors) == 0:
        raise ValueError('`tensors` must not be empty.')
    ret = tensors[0]
    for i in range(len(tensors) - 1):
        ret = ret + tensors[i + 1]
    return ret


def add_n(tensors: Iterable[TensorLike]) -> Tensor:
    return _add_n(list(map(as_tensor, tensors)))


# ---- reduction operations ----
{%- for op_name in ['sum', 'mean'] %}

@jit
def _reduce_{{ op_name }}_sub(x, keepdims):
    # type: (Tensor, bool) -> Tensor
    if keepdims:
        return torch.{{op_name}}(x).reshape([1] * len(x.shape))
    else:
        return torch.{{op_name}}(x)


def reduce_{{ op_name }}(x: TensorLike,
            {{ ' ' * (op_name | length) }}axis: Optional[AxisOrAxes] = None,
            {{ ' ' * (op_name | length) }}keepdims: bool = False) -> Tensor:
    x = as_tensor(x)
    if axis is None:
        return _reduce_{{ op_name }}_sub(x, keepdims=keepdims)
    else:
        return torch.{{ op_name }}(x, dim=axis, keepdim=keepdims)
{% endfor %}
{%- for op_name in ['max', 'min'] %}

@jit
def _reduce_{{ op_name }}_1(x, keepdims):
    # type: (Tensor, bool) -> Tensor
    if keepdims:
        return torch.{{op_name}}(x).reshape([1] * len(x.shape))
    else:
        return torch.{{op_name}}(x)


@jit
def _reduce_{{ op_name }}_2(x, axis, keepdims):
    # type: (Tensor, List[int], bool) -> Tensor
    if len(axis) == 1:
        return torch.{{ op_name }}(x, dim=axis[0], keepdim=keepdims)[0]
    else:
        for a in axis:
            x = torch.{{ op_name }}(x, dim=a, keepdim=True)[0]
        if not keepdims:
            x = _squeeze_slow_branch(x, axis)
        return x


def reduce_{{ op_name }}(x: TensorLike,
            {{ ' ' * (op_name | length) }}axis: Optional[AxisOrAxes] = None,
            {{ ' ' * (op_name | length) }}keepdims: bool = False) -> Tensor:
    if axis is not None:
        axis = list(axis) if isinstance(axis, (tuple, list)) else [axis]
    x = as_tensor(x)

    if axis is None:
        return _reduce_{{ op_name }}_1(x, keepdims)
    else:
        return _reduce_{{ op_name }}_2(x, axis, keepdims)
{% endfor %}

def log_sum_exp(x: TensorLike,
                axis: Optional[AxisOrAxes] = None,
                keepdims: bool = False) -> Tensor:
    x = as_tensor(x)
    if axis is None:
        axis = list(range(len(x.shape)))
        if keepdims:
            return torch.logsumexp(x, dim=axis, keepdim=True)
        else:
            return torch.logsumexp(x, dim=axis, keepdim=False)
    else:
        return torch.logsumexp(x, dim=axis, keepdim=keepdims)


def log_mean_exp(x: TensorLike,
                 axis: Optional[AxisOrAxes] = None,
                 keepdims: bool = False) -> Tensor:
    if axis is not None:
        axis = list(axis) if isinstance(axis, (tuple, list)) else [axis]
    x = as_tensor(x)
    x_max_keepdims = reduce_max(x, axis=axis, keepdims=True)
    if not keepdims:
        x_max = squeeze(x_max_keepdims, axis=axis)
    else:
        x_max = x_max_keepdims
    mean_exp = reduce_mean(
        torch.exp(x - x_max_keepdims), axis=axis, keepdims=keepdims)
    return x_max + torch.log(mean_exp)


# ---- logical operations ----
boolean = torch.uint8


def to_boolean(x: TensorLike) -> Tensor:
    return as_tensor(x).to(torch.bool).to(boolean)


def logical_not(x: TensorLike) -> Tensor:
    x = as_tensor(x)
    if x.dtype != boolean:
        raise TypeError(f'Expected x to be {boolean}, got {x!r} of type '
                        f'{x.dtype} instead.')
    return ~x
{% for op_name in ['and', 'or', 'xor'] %}

def logical_{{ op_name }}(x: TensorLike, y: TensorLike) -> Tensor:
    x = as_tensor(x)
    y = as_tensor(y)

    if x.dtype != boolean:
        raise TypeError(f'Expected x to be {boolean}, got {x!r} of type '
                        f'{x.dtype} instead.')
    if y.dtype != boolean:
        raise TypeError(f'Expected y to be {boolean}, got {y!r} of type '
                        f'{y.dtype} instead.')

    return x {{ {'and': '&', 'or': '|', 'xor': '^'}[op_name] }} y
{% endfor %}

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


def detach(x: Tensor) -> Tensor:
    return x.detach()


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

    def __len__(self):
        return len(self.tensor)

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
        return logical_not(self.tensor)

    def __and__(self, other):
        return logical_and(self.tensor, other)

    def __rand__(self, other):
        return logical_and(other, self.tensor)

    def __or__(self, other):
        return logical_or(self.tensor, other)

    def __ror__(self, other):
        return logical_or(other, self.tensor)

    def __xor__(self, other):
        return logical_xor(self.tensor, other)

    def __rxor__(self, other):
        return logical_xor(other, self.tensor)

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
        if isinstance(item, TensorWrapper):
            # special hack: otherwise pytorch will try to call
            # `len(item)`.  And if item is a scalar index, it will fail.
            item = item.tensor
        return self.tensor[item]


def register_tensor_wrapper_class(cls: Type[TensorWrapper]):
    if not isinstance(cls, type) or not issubclass(cls, TensorWrapper):
        raise TypeError(f'`{cls}` is not a class, or not a subclass of '
                        f'`TensorWrapper`')


register_as_tensor(
    TensorWrapper,
    (lambda t, dtype=None: t.as_tensor(dtype))
)
