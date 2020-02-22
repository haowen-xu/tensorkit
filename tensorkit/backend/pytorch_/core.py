import math
from contextlib import contextmanager
from typing import *

import numpy as np
import torch
import torch.jit
import torch.nn.functional

from ...settings_ import settings, JitMode

__all__ = [
    # constants
    'IS_CHANNEL_LAST', 'EPSILON', 'CPU_DEVICE',

    # typing
    'Tensor', 'Variable', 'Module',

    # ordinary module base classes
    # jit
    'is_function_jit_enabled', 'is_module_jit_enabled',
    'jit', 'jit_ignore', 'jit_method',

    # device
    'get_device', 'to_device', 'current_device', 'use_device',
    'gpu_device_list', 'first_gpu_device',

    # utilities
    'int_range', 'identity',

    # cast
    'cast', 'cast_like',

    # dtypes
    'get_dtype', 'is_floating_point', 'is_floating_point_dtype',

    # tensor constructors
    'as_tensor', 'from_numpy',
    'float_scalar', 'float_scalar_like', 'int_scalar', 'int_scalar_like',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like',
    'arange', 'one_hot',

    # to_numpy
    'to_numpy',

    # variable and initialize
    'variable',

    # assignment to variable
    'fill', 'fill_zeros', 'assign', 'assign_data',

    # shape utils
    'shape', 'rank', 'reshape', 'repeat', 'expand', 'squeeze', 'expand_dim',
    'swap_axes', 'transpose',
    'broadcast_shape', 'broadcast_to', 'explicit_broadcast', 'flatten_to_ndims',
    'unflatten_from_ndims', 'reshape_tail',

    # split / join / indexing / gathering ...
    'index_select', 'concat', 'split', 'stack', 'unstack', 'slice', 'slice_axis',
    'pad', 'pad_axis', 'shift', 'shift_axis',

    # math operators
    'floor', 'ceil', 'abs', 'neg', 'square', 'exp', 'log', 'log1p', 'sin',
    'cos', 'tan', 'tanh',
    'erf', 'erfc', 'erfinv',
    'add', 'sub', 'mul', 'div', 'mod', 'pow', 'sqrt', 'truediv', 'floordiv',
    'add_n',

    # reduce operators
    'reduce_sum', 'reduce_mean', 'reduce_max', 'reduce_min',
    'argmax', 'argmin', 'log_sum_exp', 'log_mean_exp',
    # 'all', 'any',
    'calculate_mean_and_var', 'norm_except_axis',

    # logical operators
    'logical_not', 'logical_and', 'logical_or', 'logical_xor', 'multiply_mask',
    'where',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'minimum', 'maximum', 'clip', 'maybe_clip',

    # sort operators
    'sort', 'argsort',

    # matrix operators
    'matmul', 'matrix_inverse',

    # gradient utilities
    'grad', 'is_null_grad', 'requires_grad', 'stop_grad', 'no_grad',

    # debug utilities
    'is_all', 'is_finite', 'assert_finite',
]


# ---- constants ----
IS_CHANNEL_LAST = False
"""Whether or not the channel axis is the last axis for convolutional operations?"""

EPSILON = 1e-6
"""The small infinitesimal constant to avoid diving by zero of taking logarithm of zero."""

CPU_DEVICE = 'cpu'
"""The constant that represents the local CPU device."""


# ---- typing ----
Tensor = torch.Tensor
Variable = torch.Tensor
Module = torch.nn.Module


# ---- jit ----
def is_function_jit_enabled() -> bool:
    return settings.jit_mode is not None and settings.jit_mode != JitMode.NONE


def is_module_jit_enabled() -> bool:
    return settings.jit_mode is not None and settings.jit_mode == JitMode.ALL


def jit(fn):
    if is_function_jit_enabled():
        fn = torch.jit.script(fn)
    return fn


def jit_ignore(fn):
    if is_function_jit_enabled() or is_module_jit_enabled():
        fn = torch.jit.ignore(fn)
    return fn


def jit_method(fn):
    if is_module_jit_enabled():
        fn = torch.jit.export(fn)
    return fn


# ---- device ----
@jit
def get_device(t: Tensor) -> str:
    return str(t.device)


@jit
def to_device(t: Tensor, device: str) -> Tensor:
    if str(t.device) != device:
        t = t.to(device=device)
    return t


_current_device = [CPU_DEVICE]


@jit_ignore
def current_device() -> str:
    return _current_device[0]


@contextmanager
def use_device(device: str):
    if not torch.cuda.is_available():
        if device != CPU_DEVICE:
            raise RuntimeError('GPU is not available.')
        yield
    else:
        old_device = _current_device[0]
        try:
            _current_device[0] = device
            yield
        finally:
            _current_device[0] = old_device


def gpu_device_list() -> List[str]:
    return [f'cuda:{index}' for index in range(torch.cuda.device_count())]


def first_gpu_device(fallback_to_cpu: bool = True) -> str:
    gpu_list = gpu_device_list()
    if not gpu_list:
        if not fallback_to_cpu:  # pragma: no cover
            raise RuntimeError('No GPU is available.')
        else:
            return CPU_DEVICE
    return gpu_list[0]


# ---- utilities ----
if not is_function_jit_enabled():
    def int_range(start: int, end: int, step: int = 1) -> List[int]:
        return list(range(start, end, step))
else:
    @jit
    def int_range(start: int, end: int, step: int = 1) -> List[int]:
        ret: List[int] = []
        for i in range(start, end, step):
            ret.append(i)
        return ret


# ---- cast dtype and device ----
@jit
def cast(input: Tensor,
         dtype: Optional[str] = None,
         device: Optional[str] = None) -> Tensor:
    if dtype is None:
        target_dtype = input.dtype
    else:
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if target_dtype != input.dtype and device is not None:
        output = input.to(dtype=target_dtype, device=device)
    elif target_dtype != input.dtype:
        output = input.to(dtype=target_dtype)
    elif device is not None:
        output = input.to(device=device)
    else:
        output = input

    return output


@jit
def cast_like(input: Tensor, like: Tensor) -> Tensor:
    return input.to(dtype=like.dtype, device=like.device)


# ---- dtypes ----
@jit
def get_dtype(input: Tensor) -> str:
    if input.dtype == torch.float32:
        return 'float32'
    elif input.dtype == torch.int32:
        return 'int32'
    else:
        return {torch.int8: 'int8', torch.uint8: 'uint8', torch.int16: 'int16', torch.int64: 'int64', torch.float16: 'float16', torch.float64: 'float64', torch.bool: 'bool'}[input.dtype]


@jit
def is_floating_point(input: Tensor) -> bool:
    return input.is_floating_point()


def is_floating_point_dtype(dtype: str) -> bool:
    try:
        if dtype == 'float32':
            real_dtype = torch.float32
        else:
            real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]
        return True
    except KeyError:
        return False


# ---- tensor constructors ----
@jit_ignore
def as_tensor(data,
              dtype: Optional[Union[torch.dtype, str]] = None,
              device: Optional[str] = None,
              force_copy: bool = False) -> Tensor:
    """
    Construct a new tensor from `data`.

    This method will copy `data` only when it is required to do so, or
    when `force_copy` is set to :obj:`True`.

    Args:
        data: The tensor data.  It might be a Python number, a NumPy array,
            another tensor, a :class:`~tensorkit.StochasticTensor`, or anything
            else that the backend supports.
        dtype: The expected dtype of the constructed tensor.
        device: The device where to place new tensors and variables.
        force_copy: Force to copy `data` even if it is not necessary.
            The gradient propagation will not be stopped from the copied tensor
            to the original tensor.  The caller may need to use `T.stop_grad()`
            if necessary.

            It should not be necessary to copy the given `data`, if `data`
            is already another tensor with `dtype`; or if `data` is a NumPy
            array with compatible `dtype`, and the backend supports to share
            memory between a tensor and a NumPy array.

    Returns:
        The constructed tensor.
    """
    from tensorkit import StochasticTensor

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

    # check the device argument
    if device is None:
        device = current_device()

    # if `data` is already a tensor
    if isinstance(data, StochasticTensor):
        data = data.tensor

    if isinstance(data, Tensor):
        # input `data` may be `StochasticTensor`, `Tensor` or `numpy.ndarray`
        kwargs = {}
        if data.dtype != target_dtype:
            kwargs['dtype'] = target_dtype
        if str(data.device) != device:
            kwargs['device'] = device
        if kwargs:
            data = data.to(**kwargs)
        if force_copy:
            data = data.clone()
        return data

    # or if `data` is other types
    ret = torch.as_tensor(data, dtype=target_dtype, device=device)
    if force_copy:
        ret = ret.clone()
    return ret


@jit_ignore
def from_numpy(data,
               dtype: Optional[Union[torch.dtype, str]] = None,
               device: Optional[str] = None) -> Tensor:
    """
    Construct a new tensor from given numpy array `data`.

    Args:
        data: The numpy array, which will always be copied, even if the backend
            supports share memory between a numpy array and a tensor.
        dtype: The expected dtype of the constructed tensor.
        device: Where to put the new tensor.

    Returns:
        The constructed tensor.
    """
    if device is None:
        device = current_device()
    return as_tensor(data, dtype=dtype, device=device, force_copy=True)


@jit
def float_scalar(data: float,
                 dtype: str = settings.float_x,
                 device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        real_dtype = torch.float32
    else:
        real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]

    if device is None:
        device = current_device()
    return torch.tensor(data, dtype=real_dtype, device=device)


@jit
def float_scalar_like(data: float, like: Tensor) -> Tensor:
    return torch.tensor(data, dtype=like.dtype, device=like.device)


@jit
def int_scalar(data: int,
               dtype: str = 'int32',
               device: Optional[str] = None) -> Tensor:
    if dtype == 'int32':
        int_dtype = torch.int32
    else:
        int_dtype = {'int8': torch.int8, 'int16': torch.int16, 'int64': torch.int64}[dtype]

    if device is None:
        device = current_device()
    return torch.tensor(data, dtype=int_dtype, device=device)


@jit
def int_scalar_like(data: int, like: Tensor) -> Tensor:
    return torch.tensor(data, dtype=like.dtype, device=like.device)


@jit
def zeros(shape: List[int],
          dtype: str = settings.float_x,
          device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if device is None:
        device = current_device()
    return torch.zeros(shape, dtype=target_dtype, device=device)


@jit
def zeros_like(input: Tensor,
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
        target_dtype = input.dtype
    if shape is None:
        shape = list(input.shape)
    return torch.zeros(shape, dtype=target_dtype, device=input.device)


@jit
def ones(shape: List[int],
         dtype: str = settings.float_x,
         device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if device is None:
        device = current_device()
    return torch.ones(shape, dtype=target_dtype, device=device)


@jit
def ones_like(input: Tensor,
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
        target_dtype = input.dtype
    if shape is None:
        shape = list(input.shape)
    return torch.ones(shape, dtype=target_dtype, device=input.device)


@jit
def full(shape: List[int],
         fill_value: float,
         dtype: str = settings.float_x,
         device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if device is None:
        device = current_device()
    return torch.full(shape, fill_value, dtype=target_dtype, device=device)


@jit
def full_like(input: Tensor,
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
        target_dtype = input.dtype
    if shape is None:
        shape = list(input.shape)
    return torch.full(shape, fill_value, dtype=target_dtype, device=input.device)


@jit
def arange(start: int, end: int, step: int = 1, dtype: str = 'int32',
           device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if device is None:
        device = current_device()
    return torch.arange(start, end, step, dtype=target_dtype, device=device)


@jit
def one_hot(input: Tensor,
            n_classes: int,
            dtype: str = 'int32',
            axis: int = -1) -> Tensor:
    r = input.dim()
    if axis < -(r + 1) or axis > r:
        raise ValueError('`axis` out of range: the minimum allowed axis is {}, '
                         'while the maximum allowed axis is {}.'.
                         format(-(r + 1), r))

    ret = torch.nn.functional.one_hot(input, n_classes)
    if axis != -1 and axis != r:
        if axis < 0:
            axis = axis + ret.dim()
        new_axis: List[int] = int_range(0, r)
        new_axis.insert(axis, r)
        ret = ret.permute(new_axis)
    ret = cast(ret, dtype)
    return ret


# ---- to_numpy ----
@jit_ignore
def to_numpy(input: Tensor) -> np.ndarray:
    if not isinstance(input, Tensor):
        raise TypeError(f'Not a Tensor: got {input!r}')
    return input.detach().cpu().numpy()


# ---- variable and initializer ----
def variable(shape: List[int],
             dtype: Union[str, torch.dtype] = settings.float_x,
             device: Optional[str] = None,
             initializer: Optional[
                 Union[
                     int, float, np.ndarray, Tensor,
                     Callable[[Variable], None]
                 ]
             ] = None,
             requires_grad: bool = True,
             force_copy: bool = True) -> Variable:
    """
    Create a new variable.

    Args:
        shape: Shape of the variable.
        dtype: Dtype of the variable.
        device: The device where to place new tensors and variables.
        initializer: The variable initializer.  It may be a scalar (which
            will be filled into the new variable), an array or another
            `Tensor` with the same shape as specified `shape`, or a callable
            function that can be used to initialize the variable.
        requires_grad: Whether or not that the variable requires gradient
            during back-propagation?  Defaults to :obj:`True`.
        force_copy: Whether or not to force copy the data from `initializer`,
            even if the backend supports sharing memory?
            Defaults to :obj:`True`.

    Returns:
        The created variable.
    """
    if isinstance(dtype, str):
        if dtype == 'float32':
            target_dtype = torch.float32
        elif dtype == 'int32':
            target_dtype = torch.int32
        else:
            target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]
    else:
        target_dtype = dtype

    if device is None:
        device = current_device()

    if isinstance(initializer, (int, float)):
        ret = torch.full(shape, float(initializer), dtype=target_dtype,
                         device=device, requires_grad=requires_grad)
    elif isinstance(initializer, np.ndarray) and initializer.shape == ():
        ret = torch.full(shape, initializer.tolist(), dtype=target_dtype,
                         device=device, requires_grad=requires_grad)
    elif isinstance(initializer, (np.ndarray, Tensor)):
        if list(initializer.shape) != shape:
            raise ValueError(f'`initializer.shape` != `shape`: '
                             f'{list(initializer.shape)} vs {shape}')
        if isinstance(initializer, Tensor):
            initializer = to_numpy(initializer)
        ret = as_tensor(initializer, dtype=target_dtype,
                        device=device, force_copy=force_copy)
        if requires_grad:
            ret.requires_grad_(True)
    elif isinstance(initializer, Callable):
        ret = zeros(shape, device=device, dtype=dtype)
        with torch.no_grad():
            initializer(ret)
        if requires_grad:
            ret.requires_grad_(True)
    elif initializer is None:
        ret = torch.zeros(shape, dtype=target_dtype, device=device,
                          requires_grad=requires_grad)
    else:
        raise TypeError(f'Unsupported initializer: {initializer!r}')

    return ret


# ---- assignment ----
@jit
def fill(dst: Tensor, fill_value: float) -> Tensor:
    dst.fill_(fill_value)
    return dst


@jit
def fill_zeros(dst: Tensor) -> Tensor:
    dst.zero_()
    return dst


@jit
def assign(dst: Tensor, src: Tensor) -> Tensor:
    if src.shape != dst.shape:
        raise ValueError('`dst.shape` != `src.shape`: {} vs {}'.
                         format(shape(dst), shape(src)))
    dst.copy_(src.detach())
    return dst


@jit_ignore
def assign_data(dst: Tensor, src) -> Tensor:
    src = as_tensor(src, force_copy=True).detach()
    if src.shape != dst.shape:
        raise ValueError('`dst.shape` != `src.shape`: {} vs {}'.
                         format(shape(dst), shape(src)))
    dst.data = src
    return dst


# ---- shape utils ----
@jit
def shape(input: Tensor) -> List[int]:
    return list(input.shape)


@jit
def rank(input: Tensor) -> int:
    return input.dim()


@jit
def reshape(input: Tensor, shape: List[int]) -> Tensor:
    return input.reshape(shape)


@jit
def repeat(input: Tensor, repeats: List[int]) -> Tensor:
    x_shape = input.shape
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
        return input
    elif mode == 1:
        expands = repeats[:extra_len] + \
            list([-1 if a == 1 else a for a in repeats[extra_len:]])
        return input.expand(expands)
    else:
        return input.repeat(repeats)


@jit
def expand(input: Tensor, desired_shape: List[int]) -> Tensor:
    return input.expand(desired_shape)


@jit
def squeeze(input: Tensor, axis: Optional[List[int]] = None) -> Tensor:
    if axis is not None:
        if len(axis) == 1:
            return torch.squeeze(input, axis[0])
        else:
            old_shape = input.shape
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
            return input.reshape(new_shape)
    else:
        return torch.squeeze(input)


@jit
def expand_dim(input: Tensor, axis: int) -> Tensor:
    return input.unsqueeze(axis)


@jit
def swap_axes(input: Tensor, axis1: int, axis2: int) -> Tensor:
    return input.transpose(axis1, axis2)


@jit
def transpose(input: Tensor, axis: List[int]) -> Tensor:
    return input.permute(axis)


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
def broadcast_to(input: Tensor, new_shape: List[int]) -> Tensor:
    x_shape = list(input.shape)
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

    return _broadcast_to_sub(input, x_shape, new_shape)


@jit
def explicit_broadcast(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    out_shape = broadcast_shape(x_shape, y_shape)
    x = _broadcast_to_sub(x, x_shape, out_shape)
    y = _broadcast_to_sub(y, y_shape, out_shape)
    return x, y


@jit
def flatten_to_ndims(input: Tensor, ndims: int
                     ) -> Tuple[Tensor, Optional[List[int]]]:
    if ndims < 1:
        raise ValueError('`ndims` must be at least 1`: got ndims {}'.
                         format(ndims))
    if len(input.shape) < ndims:
        raise ValueError('rank(x) < ndims: x.shape is {}, while '
                         'ndims is {}'.format(input.shape, ndims))

    if ndims == len(input.shape):
        return input, None  # `None` to indicate x is not changed
    elif ndims == 1:
        front_shape = list(input.shape)
        return input.reshape((-1,)), front_shape
    else:
        x_shape = list(input.shape)
        offset = ndims - 1
        front_shape, back_shape = x_shape[: -offset], x_shape[-offset:]
        return input.reshape([-1] + back_shape), front_shape


@jit
def unflatten_from_ndims(input: Tensor, front_shape: Optional[List[int]]
                         ) -> Tensor:
    x_shape = list(input.shape)
    if front_shape is None:
        return input
    else:
        x_rank = len(x_shape)
        if x_rank < 1:
            raise ValueError(
                'Invalid input: rank(x) < 1, but front_shape is not None.')
        return input.reshape(list(front_shape) + x_shape[1:])


@jit
def reshape_tail(input: Tensor, ndims: int, shape: List[int]) -> Tensor:
    input_shape = list(input.shape)
    input_rank = len(input_shape)
    if input_rank < ndims:
        raise ValueError(
            '`input` must be at least `ndims`-dimensional: '
            '`shape(input)` is {}, while `ndims` is {}'.
            format(input_shape, ndims)
        )
    left_shape = input_shape[: input_rank - ndims]
    return input.reshape(left_shape + shape)


# ---- split / join / indexing / gathering ----
@jit
def index_select(input: Tensor, indices: Tensor, axis: int) -> Tensor:
    x_shape = input.shape
    i_shape = indices.shape

    if axis < 0:
        axis += len(x_shape)
    if axis < 0 or axis >= len(x_shape):
        raise ValueError('`axis` out of range: x.shape {} vs axis {}'.
                         format(input.shape, axis))

    if len(i_shape) == 0:
        y = torch.index_select(input, dim=axis, index=indices.reshape([1]))
        y = y.reshape(x_shape[:axis] + x_shape[axis + 1:])

    elif len(i_shape) == 1:
        y = torch.index_select(input, dim=axis, index=indices)

    else:
        y = torch.index_select(input, dim=axis, index=indices.flatten())
        y = y.reshape(x_shape[:axis] + i_shape + x_shape[axis + 1:])

    return y


@jit
def concat(tensors: List[Tensor], axis: int) -> Tensor:
    return torch.cat(tensors, dim=axis)


@jit
def split(input: Tensor, sections: List[int], axis: int) -> List[Tensor]:
    return torch.split(input, sections, axis)


@jit
def stack(input: List[Tensor], axis: int) -> Tensor:
    return torch.stack(input, axis)


@jit
def unstack(input: Tensor, axis: int) -> List[Tensor]:
    size = input.shape[axis]
    outputs: List[Tensor] = list(torch.split(input, [1] * size, dim=axis))
    for i in range(len(outputs)):
        outputs[i] = torch.squeeze(outputs[i], dim=axis)
    return outputs


@jit
def slice_axis(input: Tensor,
               axis: int,
               start: int,
               length: Optional[int] = None) -> Tensor:
    if length is None:
        if start < 0:
            length = -start
        else:
            length = input.shape[axis] - start
    return torch.narrow(input, axis, start, length)


@jit
def slice(input: Tensor,
          slice_start: List[int],
          slice_length: Optional[List[Optional[int]]] = None
          ) -> Tensor:
    slice_count = len(slice_start)
    if slice_count > input.dim():
        raise ValueError(
            '`len(slice_start)` must be less or equal to `rank(input)`: '
            'got input shape {}, slice_start {}, slice_length {}.'.
            format(shape(input), slice_start, slice_length)
        )
    if slice_length is None:
        output = input
        for i in range(-1, -(slice_count + 1), -1):
            output = slice_axis(output, i, slice_start[i])
    else:
        if slice_count != len(slice_length):
            raise ValueError('`len(slice_start)` != `len(slice_length)`: '
                             'got slice_start {}, slice_length {}.'.
                             format(slice_start, slice_length))
        output = input
        for i in range(-1, -(slice_count + 1), -1):
            output = slice_axis(output, i, slice_start[i], slice_length[i])
    return output


@jit
def pad_axis(input: Tensor,
             axis: int,
             padding: Tuple[int, int],
             value: float = 0.) -> Tensor:
    r = input.dim()
    if axis < -r or axis >= r:
        raise ValueError('`axis` out of range: expected to be >= {} and '
                         '< {}, got {}.'.format(-r, r, axis))
    if axis < 0:
        axis = axis + r
    pad: List[int] = []
    for i in range(r - 1, axis, -1):
        pad.extend((0, 0))
    pad.extend(padding)
    return torch.nn.functional.pad(input, pad=pad, value=value)


@jit
def pad(input: Tensor,
        padding: List[Tuple[int, int]],
        value: float = 0.) -> Tensor:
    if len(padding) > input.dim():
        raise ValueError(
            'The length of `padding` must not be larger than `rank(input)`: '
            '`padding` is {}, while `shape(input)` is {}'.
            format(padding, shape(input))
        )
    pad: List[int] = []
    for i in range(len(padding) - 1, -1, -1):
        pad.extend(padding[i])
    return torch.nn.functional.pad(input, pad=pad, value=value)


@jit
def shift_axis(input: Tensor,
               axis: int,
               shift: int,
               fill_value: float = 0.) -> Tensor:
    size = input.shape[axis]
    if shift < -size or shift > size:
        raise ValueError('`shift` out of range: expected to be >= {} '
                         'and <= {}.'.format(-size, size))
    if shift < 0:
        output = pad_axis(
            torch.narrow(input, axis, -shift, size + shift),
            axis,
            (0, -shift),
            fill_value
        )
    elif shift > 0:
        output = pad_axis(
            torch.narrow(input, axis, 0, size - shift),
            axis,
            (shift, 0),
            fill_value
        )
    else:
        output = input
    return output


@jit
def shift(input: Tensor,
          shift: List[int],
          fill_value: float = 0.) -> Tensor:
    shift_length = len(shift)
    if shift_length > input.dim():
        raise ValueError('`len(shift) <= rank(input)` does not hold: '
                         'got `shift` {}, and `shape(input)` {}.'.
                         format(shift, shape(input)))

    padding: List[int] = []
    output = input
    need_pad: bool = False

    for axis in range(-1, -(shift_length + 1), -1):
        s = shift[axis]
        size = input.shape[axis]
        if s < -size or s > size:
            raise ValueError(
                '`shift` out of range at axis {}: expected to be >= {} '
                'and <= {}.'.format(axis, -size, size)
            )
        if s < 0:
            padding.append(0)
            padding.append(-s)
            output = torch.narrow(output, axis, -s, size + s)
            need_pad = True
        elif s > 0:
            padding.append(s)
            padding.append(0)
            output = torch.narrow(output, axis, 0, size - s)
            need_pad = True
        else:
            padding.append(0)
            padding.append(0)
        axis -= 1

    if need_pad:
        output = torch.nn.functional.pad(
            output, padding, mode='constant', value=fill_value)

    return output


# ---- univariate element-wise math operations ----
@jit
def identity(input: Tensor) -> Tensor:
    return input


floor = torch.floor
ceil = torch.ceil
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
tan = torch.tan

tanh = torch.tanh

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
sqrt = torch.sqrt


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
def reduce_sum(input: Tensor,
               axis: Optional[List[int]] = None,
               keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.sum(input).reshape([1] * len(input.shape))
        else:
            return torch.sum(input)
    else:
        if len(axis) == 0:
            return input
        else:
            return torch.sum(input, dim=axis, keepdim=keepdims)


@jit
def reduce_mean(input: Tensor,
                axis: Optional[List[int]] = None,
                keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.mean(input).reshape([1] * len(input.shape))
        else:
            return torch.mean(input)
    else:
        if len(axis) == 0:
            return input
        else:
            return torch.mean(input, dim=axis, keepdim=keepdims)


@jit
def reduce_max(input: Tensor,
               axis: Optional[List[int]] = None,
               keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.max(input).reshape([1] * len(input.shape))
        else:
            return torch.max(input)
    else:
        if len(axis) == 0:
            return input
        elif len(axis) == 1:
            return torch.max(input, dim=axis[0], keepdim=keepdims)[0]
        else:
            for a in axis:
                input = torch.max(input, dim=a, keepdim=True)[0]
            if not keepdims:
                input = squeeze(input, axis)
            return input


@jit
def reduce_min(input: Tensor,
               axis: Optional[List[int]] = None,
               keepdims: bool = False) -> Tensor:
    if axis is None:
        if keepdims:
            return torch.min(input).reshape([1] * len(input.shape))
        else:
            return torch.min(input)
    else:
        if len(axis) == 0:
            return input
        elif len(axis) == 1:
            return torch.min(input, dim=axis[0], keepdim=keepdims)[0]
        else:
            for a in axis:
                input = torch.min(input, dim=a, keepdim=True)[0]
            if not keepdims:
                input = squeeze(input, axis)
            return input


@jit
def argmax(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return torch.argmax(input, dim=axis, keepdim=keepdims)


@jit
def argmin(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return torch.argmin(input, dim=axis, keepdim=keepdims)


@jit
def log_sum_exp(input: Tensor,
                axis: Optional[List[int]] = None,
                keepdims: bool = False) -> Tensor:
    if axis is None:
        axis = int_range(0, len(input.shape))
        if keepdims:
            return torch.logsumexp(input, dim=axis, keepdim=True)
        else:
            return torch.logsumexp(input, dim=axis, keepdim=False)
    else:
        if len(axis) == 0:
            raise ValueError('`axis` must not be an empty list.')
        return torch.logsumexp(input, dim=axis, keepdim=keepdims)


@jit
def log_mean_exp(input: Tensor,
                 axis: Optional[List[int]] = None,
                 keepdims: bool = False) -> Tensor:
    if axis is not None:
        if len(axis) == 0:
            raise ValueError('`axis` must not be an empty list.')
    x_max_keepdims = reduce_max(input, axis=axis, keepdims=True)
    if not keepdims:
        x_max = squeeze(x_max_keepdims, axis=axis)
    else:
        x_max = x_max_keepdims
    mean_exp = reduce_mean(
        torch.exp(input - x_max_keepdims), axis=axis, keepdims=keepdims)
    return x_max + torch.log(mean_exp)


@jit
def calculate_mean_and_var(input: Tensor,
                           axis: Optional[List[int]] = None,
                           keepdims: bool = False,
                           unbiased: bool = True) -> Tuple[Tensor, Tensor]:
    # compute mean & var
    mean = reduce_mean(input, axis=axis, keepdims=True)
    var = reduce_mean((input - mean) ** 2, axis=axis, keepdims=keepdims)
    if not keepdims:
        mean = mean.reshape(var.shape)

    reduce_size = input.numel() // mean.numel()
    if reduce_size < 2:
        raise RuntimeError(
            'Variance can only be calculated with at least 2 samples.')

    # obtain unbiased estimator from the biased estimator by multiply n / (n-1)
    if unbiased:
        var = var * (float(reduce_size) / (reduce_size - 1.))

    return mean, var


@jit
def norm_except_axis(input: Tensor,
                     axis: Optional[List[int]],
                     p: float = 2,
                     keepdims: bool = False) -> Tensor:
    """
    Calculate the Lp-norm of a tensor except for specified axis.

    Args:
        input: The input tensor.
        axis: The axis to keep for computing Lp-norm.
            All other axis will be reduced.  Defaults to :obj:`None`,
            where no axis will be kept.
        p: The `p` of the `Lp` norm.  Defaults to 2.
        keepdims: Whether or not to keep the reduced dimensions?
            Defaults to :obj:`False`.

    Returns:
        The Lp-norm of the tensor.
    """
    r = rank(input)
    if axis is None:
        axis_reduce = None
    elif len(axis) == 1:
        # compute the axis to reduce in a fast manner
        a = axis[0]
        if a < -r or a >= r:
            raise ValueError(
                f'`axis` out of range: `axis` is {axis}, '
                f'while the shape of `input` is {shape(input)}.')
        if a < 0:
            a = a + r
        axis_reduce = int_range(0, a) + int_range(a + 1, r)
    else:
        # compute the axis to reduce in a slow manner
        axis_mask: List[bool] = [True] * r
        for a in axis:
            if a < -r or a >= r:
                raise ValueError(
                    f'`axis` out of range: `axis` is {axis}, '
                    f'while the shape of `input` is {shape(input)}.')
            axis_mask[a] = False
        axis_reduce: List[int] = []
        for i in range(r):
            if axis_mask[i]:
                axis_reduce.append(i)

    if p == 2:
        return sqrt(reduce_sum(input ** 2, axis=axis_reduce, keepdims=keepdims))
    elif p == 1:
        return reduce_sum(abs(input), axis=axis_reduce, keepdims=keepdims)
    else:
        p_inv = 1. / p
        return pow(
            reduce_sum(pow(abs(input), p), axis=axis_reduce, keepdims=keepdims),
            p_inv
        )


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
def multiply_mask(input: Tensor, mask: Tensor) -> Tensor:
    if mask.dtype != input.dtype:
        mask = mask.to(input.dtype)
    return input * mask


where = torch.where


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


@jit
def maybe_clip(x: Tensor,
               x_min: Optional[float] = None,
               x_max: Optional[float] = None) -> Tensor:
    if x_min is not None and x_max is not None:
        return clip(x, x_min, x_max)
    elif x_min is not None:
        return torch.max(x, torch.as_tensor(x_min, dtype=x.dtype, device=x.device))
    elif x_max is not None:
        return torch.min(x, torch.as_tensor(x_max, dtype=x.dtype, device=x.device))
    else:
        return x


# ---- sort operators ----
@jit
def sort(input: Tensor, axis: int = -1, descending: bool = False) -> Tensor:
    output, indices = torch.sort(input, dim=axis, descending=descending)
    return output


@jit
def argsort(input: Tensor, axis: int = -1, descending: bool = False) -> Tensor:
    indices = torch.argsort(input, dim=axis, descending=descending)
    return indices


# ---- matrix operators ----
@jit
def matmul(x: Tensor, y: Tensor) -> Tensor:
    return torch.matmul(x, y)


@jit
def matrix_inverse(matrix: Tensor) -> Tensor:
    return torch.inverse(matrix)


# ---- gradient utilities ----
if not is_function_jit_enabled() or not torch.__version__.startswith('1.3.'):
    @jit
    def grad(outputs: List[Tensor],
             inputs: List[Tensor],
             grad_outputs: Optional[List[Optional[Tensor]]] = None,
             retain_graph: Optional[bool] = None,
             create_graph: bool = False,
             allow_unused: bool = False) -> List[Optional[Tensor]]:
        grad_outs = list(
            torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
                retain_graph=retain_graph,
                create_graph=create_graph,
                allow_unused=allow_unused,
            )
        )

        if not allow_unused:
            for i in range(len(grad_outs)):
                if grad_outs[i] is None:
                    raise RuntimeError(
                        'One of the differentiated Tensors '
                        'appears to not have been used in the graph. '
                        'Set allow_unused=True if this is the desired '
                        'behavior.'
                    )

        return grad_outs


    def is_null_grad(origin: Tensor, grad: Optional[Tensor]) -> bool:
        return grad is None

else:
    @jit
    def grad(outputs: List[Tensor],
             inputs: List[Tensor],
             grad_outputs: Optional[List[Optional[Tensor]]] = None,
             retain_graph: Optional[bool] = None,
             create_graph: bool = False,
             allow_unused: bool = False) -> List[Tensor]:
        grad_outs = list(
            torch.autograd.grad(
                outputs=outputs,
                inputs=inputs,
                grad_outputs=grad_outputs,
                keep_graph=retain_graph,
                create_graph=create_graph,
                allow_unused=allow_unused,
            )
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


def requires_grad(input: Tensor,
                  requires: bool = True,
                  copy: bool = False) -> Tensor:
    if copy:
        return input.clone().requires_grad_(requires)
    else:
        input.requires_grad_(requires)
        return input


@jit
def stop_grad(input: Tensor) -> Tensor:
    return input.detach()


no_grad = torch.no_grad


# ---- assertion utilities ----
@jit
def is_finite(input: Tensor) -> Tensor:
    if not input.is_floating_point():
        return torch.ones_like(input).to(torch.bool)
    return (input == input) & (input.abs() != math.inf)


@jit
def is_all(condition: Tensor) -> bool:
    return bool(torch.all(condition).item())


@jit
def assert_finite(input: Tensor, message: str) -> Tensor:
    if not is_all(is_finite(input)):
        raise ValueError('Infinity or NaN value encountered: {}'.
                         format(message))
    return input
