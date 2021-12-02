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
    'float_list', 'float_list_like', 'int_list', 'int_list_like',
    'zeros', 'zeros_like', 'ones', 'ones_like', 'full', 'full_like',
    'arange', 'one_hot', 'eye', 'diag',

    # to_numpy
    'to_numpy',

    # variable and initialize
    'variable',

    # assignment to variable
    'fill', 'fill_zeros', 'assign', 'assign_data',
    'assign_add', 'swap_assign',

    # tensor copy
    'copy', 'copy_as_variable',

    # shape utils
    'length', 'shape', 'rank', 'reshape', 'repeat', 'expand', 'squeeze',
    'squeeze_axis', 'expand_dim', 'swap_axes', 'transpose',
    'get_broadcast_shape', 'broadcast_to_shape', 'strict_broadcast_to_shape',
    'broadcast_to', 'strict_broadcast_to', 'explicit_broadcast',
    'broadcast_concat', 'flatten_to_ndims', 'unflatten_from_ndims', 'reshape_tail',

    # split / join / indexing / gathering ...
    'index_select', 'concat', 'split', 'stack', 'unstack', 'slice', 'slice_axis',
    'pad', 'pad_axis', 'shift', 'shift_axis', 'flip', 'flip_axis', 'roll',
    'roll_axis', 'embedding',

    # math operators
    'floor', 'ceil', 'abs', 'neg', 'square', 'exp', 'log', 'log1p', 'sin',
    'cos', 'tan', 'tanh',
    'erf', 'erfc', 'erfinv',
    'add', 'sub', 'mul', 'div', 'mod', 'pow', 'sqrt', 'truediv', 'floordiv',
    'add_n',

    # reduce operators
    'reduce_sum', 'reduce_sum_axis', 'reduce_mean', 'reduce_mean_axis',
    'reduce_max', 'reduce_max_axis', 'reduce_min', 'reduce_min_axis',
    'argmax', 'argmin', 'log_sum_exp', 'log_sum_exp_axis',
    'log_mean_exp', 'log_mean_exp_axis',
    # 'all', 'any',
    'calculate_mean_and_var', 'l1_norm', 'l2_norm', 'norm',
    'norm_except_axis', 'global_norm',

    # logical operators
    'logical_not', 'logical_and', 'logical_or', 'logical_xor', 'multiply_mask',
    'where',

    # comparison operators (resulting in `boolean` dtype)
    'equal', 'not_equal', 'less', 'less_equal', 'greater', 'greater_equal',
    'minimum', 'maximum',
    'clip', 'clip_right', 'clip_left', 'maybe_clip', 'clip_by_norm',
    'clip_by_global_norm',

    # sort operators
    'sort', 'argsort',

    # matrix operators
    'matmul', 'matrix_inverse',

    # gradient utilities
    'grad', 'is_null_grad', 'requires_grad', 'stop_grad', 'no_grad',

    # debug utilities
    'is_all', 'is_any', 'is_finite', 'assert_finite',
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
        input = input.to(dtype=target_dtype, device=device)
    elif target_dtype != input.dtype:
        input = input.to(dtype=target_dtype)
    elif device is not None:
        input = input.to(device=device)

    return input


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
        from_dev = str(data.device)
        if data.dtype != target_dtype and from_dev != device:
            data = data.to(device=device, dtype=target_dtype)
        elif data.dtype != target_dtype:
            data = data.to(target_dtype)
        elif from_dev != device:
            data = data.to(device=device)

        if force_copy:
            data = data.clone()
        return data

    # or if `data` is other types
    if force_copy:
        ret = torch.tensor(data, dtype=target_dtype, device=device)
    else:
        ret = torch.as_tensor(data, dtype=target_dtype, device=device)
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
def float_list(data: List[float],
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
def float_list_like(data: List[float], like: Tensor) -> Tensor:
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
def int_list(data: List[int],
             dtype: str = 'int32',
             device: Optional[str] = None):
    if dtype == 'int32':
        int_dtype = torch.int32
    else:
        int_dtype = {'int8': torch.int8, 'int16': torch.int16, 'int64': torch.int64}[dtype]

    if device is None:
        device = current_device()
    return torch.tensor(data, dtype=int_dtype, device=device)


@jit
def int_list_like(data: List[int], like: Tensor) -> Tensor:
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


@jit
def eye(n: int,
        m: Optional[int] = None,
        dtype: str = settings.float_x,
        device: Optional[str] = None):
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

    return torch.eye(n, m, dtype=target_dtype, device=device)


@jit
def diag(v: Tensor, k: int = 0) -> Tensor:
    return torch.diag(v, k)


# ---- to_numpy ----
@jit_ignore
def to_numpy(input: Tensor, force_copy: bool = False) -> np.ndarray:
    if not isinstance(input, Tensor):
        raise TypeError(f'Not a Tensor: got {input!r}')
    r = input.detach().cpu().numpy()
    if force_copy:
        r = np.copy(r)
    return r


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
                         format(dst.shape, src.shape))
    dst.copy_(src.detach())
    return dst


@jit_ignore
def assign_data(dst: Tensor, src) -> Tensor:
    src = as_tensor(src, force_copy=True).detach()
    if src.shape != dst.shape:
        raise ValueError('`dst.shape` != `src.shape`: {} vs {}'.
                         format(dst.shape, src.shape))
    dst.data = src
    return dst


@jit
def assign_add(dst: Tensor, src: Tensor) -> Tensor:
    dst.add_(src.detach())
    return dst


@jit
def swap_assign(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    tmp = torch.empty_like(x)
    tmp.copy_(x.detach())
    x.data.copy_(y.detach())
    y.data.copy_(tmp.detach())
    return x, y


# ---- tensor copy ----
@jit
def copy(input: Tensor, device: Optional[str] = None, requires_grad: bool = False) -> Tensor:
    if not requires_grad:
        input = input.detach()
    input = input.clone()
    if device is not None:
        if device != str(input.device):
            input = input.to(device=device)
    input = input.requires_grad_(requires_grad)
    return input


@jit
def copy_as_variable(input: Tensor,
                     device: Optional[str] = None,
                     requires_grad: bool = True) -> Variable:
    return copy(input, device, requires_grad=requires_grad)


# ---- shape utils ----
@jit
def length(input: Tensor) -> int:
    return input.shape[0]


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
    in_shape = list(input.shape)
    in_shape_len, repeats_len = len(in_shape), len(repeats)
    max_length = max(in_shape_len, repeats_len)
    in_shape = [1] * (max_length - in_shape_len) + in_shape
    repeats = [1] * (max_length - repeats_len) + repeats

    mode = max([
        (1 if repeats[i] != 1 else 0) + (1 if in_shape[i] != 1 else 0)
        for i in range(max_length)
    ])

    if mode == 0 and in_shape_len == max_length:
        return input
    elif mode < 2:
        extra_len = max_length - in_shape_len
        expands = repeats[:extra_len] + \
            [-1 if a == 1 else a for a in repeats[extra_len:]]
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
            new_shape: List[int] = []
            for i in range(len(old_shape)):
                if new_shape_mask[i]:
                    new_shape.append(old_shape[i])
            return input.reshape(new_shape)
    else:
        return torch.squeeze(input)


@jit
def squeeze_axis(input: Tensor, axis: int) -> Tensor:
    return torch.squeeze(input, axis)


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
def get_broadcast_shape(x: List[int], y: List[int]) -> List[int]:
    o = (
        torch.zeros(x, dtype=torch.float32) +
        torch.zeros(y, dtype=torch.float32)
    )
    return list(o.shape)

    # x_len, y_len = len(x), len(y)
    # max_length = max(x_len, y_len)
    # x_ex = [1] * (max_length - x_len) + x
    # y_ex = [1] * (max_length - y_len) + y
    # for i in range(max_length):
    #     a, b = x_ex[i], y_ex[i]
    #     if b != a and b != 1:
    #         if a != 1:
    #             raise ValueError('Shape x and y cannot broadcast against '
    #                              'each other: {} vs {}.'.format(x, y))
    #         else:
    #             x_ex[i] = b
    # return x_ex


@jit
def broadcast_to_shape(input: Tensor, new_shape: List[int]) -> Tensor:
    # TODO: do we have a better way to calculate the final shape and do broadcast here?
    if list(input.shape) != new_shape:
        input = input + torch.zeros(new_shape, dtype=input.dtype, device=input.device)
    return input


@jit
def strict_broadcast_to_shape(input: Tensor, new_shape: List[int]) -> Tensor:
    output = broadcast_to_shape(input, new_shape)
    if list(output.shape) != new_shape:
        raise ValueError(
            '`input` cannot be broadcast to `new_shape`: shape(input) {} '
            'vs new_shape {}'.format(shape(input), new_shape))
    return output


@jit
def broadcast_concat(x: Tensor, y: Tensor, axis: int) -> Tensor:
    # check the shapes of inputs
    x_shape = list(x.shape)
    y_shape = list(y.shape)
    x_rank = len(x_shape)
    y_rank = len(y_shape)
    final_rank = max(x_rank, y_rank)  # the final rank

    # validate `axis`.  Note that `final_rank == 0` will be rejected,
    # since no integer can be simultaneously >= 0 and < 0
    if axis < -final_rank or axis >= final_rank:
        raise ValueError(
            '`axis` out of range: got {}, but expected to be >= {} and < {}'.
            format(axis, -final_rank, final_rank))
    elif axis >= 0:
        # normalize `axis` to be negative
        axis = axis - final_rank

    # calculate the broadcast shape along other axis
    if x_rank < y_rank:
        x_shape = [1] * (y_rank - x_rank) + x_shape
    elif x_rank > y_rank:
        y_shape = [1] * (x_rank - y_rank) + y_shape

    x_shape[axis] = 1
    y_shape[axis] = 1
    b_shape = get_broadcast_shape(x_shape, y_shape)

    # expand x shape
    if b_shape != x_shape:
        if -x_rank <= axis:
            b_shape[axis] = x.shape[axis]
        else:
            b_shape[axis] = 1
        x = x.expand(b_shape)

    # expand y shape
    if b_shape != y_shape:
        if -y_rank <= axis:
            b_shape[axis] = y.shape[axis]
        else:
            b_shape[axis] = 1
        y = y.expand(b_shape)

    # now concat two tensors
    return torch.cat([x, y], dim=axis)


@jit
def broadcast_to(input: Tensor, target: Tensor) -> Tensor:
    if input.shape != target.shape:
        input = input + torch.zeros(target.shape, dtype=input.dtype, device=input.device)
    return input


@jit
def strict_broadcast_to(input: Tensor, target: Tensor) -> Tensor:
    output = broadcast_to(input, target)
    if output.shape != target.shape:
        raise ValueError(
            '`input` cannot be broadcast to `target`: shape(input) {} '
            'vs shape(target) {}'.format(shape(input), shape(target)))
    return output


@jit
def explicit_broadcast(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    ret = torch.broadcast_tensors(x, y)
    return ret[0], ret[1]


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
        for i in range(-1, -(slice_count + 1), -1):
            input = slice_axis(input, i, slice_start[i])
    else:
        if slice_count != len(slice_length):
            raise ValueError('`len(slice_start)` != `len(slice_length)`: '
                             'got slice_start {}, slice_length {}.'.
                             format(slice_start, slice_length))
        for i in range(-1, -(slice_count + 1), -1):
            input = slice_axis(input, i, slice_start[i], slice_length[i])
    return input


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
        input = pad_axis(
            torch.narrow(input, axis, -shift, size + shift),
            axis,
            (0, -shift),
            fill_value
        )
    elif shift > 0:
        input = pad_axis(
            torch.narrow(input, axis, 0, size - shift),
            axis,
            (shift, 0),
            fill_value
        )
    return input


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
            input = torch.narrow(input, axis, -s, size + s)
            need_pad = True
        elif s > 0:
            padding.append(s)
            padding.append(0)
            input = torch.narrow(input, axis, 0, size - s)
            need_pad = True
        else:
            padding.append(0)
            padding.append(0)
        axis -= 1

    if need_pad:
        input = torch.nn.functional.pad(
            input, padding, mode='constant', value=fill_value)

    return input


@jit
def flip_axis(input: Tensor, axis: int) -> Tensor:
    return torch.flip(input, [axis])


@jit
def flip(input: Tensor, axis: List[int]) -> Tensor:
    return torch.flip(input, axis)


@jit
def roll_axis(input: Tensor, shift: int, axis: int) -> Tensor:
    return torch.roll(input, shift, axis)


@jit
def roll(input: Tensor, shift: List[int], axis: List[int]) -> Tensor:
    if len(axis) != 0:
        input = torch.roll(input, shift, axis)
    return input


@jit
def embedding(weight: Tensor, indices: Tensor) -> Tensor:
    # ensure `input` is int64
    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)

    # `torch.embedding` only supports 2d `weight`, thus we must reshape to 2d.
    if weight.dim() != 2:
        if weight.dim() < 2:
            raise ValueError('`weight` must be at least 2d: got shape {}'.
                             format(weight.shape))
        back_shape: Optional[List[int]] = list(weight.shape[1:])
        weight = weight.reshape((weight.shape[0], -1))
    else:
        back_shape: Optional[List[int]] = None

    # do embedding lookup
    weight = torch.embedding(weight, indices)

    # reshape back to match the shape of `weight`
    if back_shape is not None:
        weight = weight.reshape(list(indices.shape) + back_shape)
    return weight


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
def reduce_sum_axis(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
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
def reduce_mean_axis(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
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
def reduce_max_axis(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return torch.max(input, dim=axis, keepdim=keepdims)[0]


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
def reduce_min_axis(input: Tensor, axis: int, keepdims: bool = False) -> Tensor:
    return torch.min(input, dim=axis, keepdim=keepdims)[0]


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
def log_sum_exp_axis(input: Tensor,
                     axis: int,
                     keepdims: bool = False) -> Tensor:
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
def log_mean_exp_axis(input: Tensor,
                      axis: int,
                      keepdims: bool = False) -> Tensor:
    x_max_keepdims = reduce_max_axis(input, axis=axis, keepdims=True)
    if not keepdims:
        x_max = torch.squeeze(x_max_keepdims, dim=axis)
    else:
        x_max = x_max_keepdims
    mean_exp = reduce_mean_axis(
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
def l1_norm(input: Tensor,
            axis: Optional[List[int]] = None,
            keepdims: bool = False) -> Tensor:
    return reduce_sum(torch.abs(input), axis=axis, keepdims=keepdims)


@jit
def l2_norm(input: Tensor,
            axis: Optional[List[int]] = None,
            keepdims: bool = False) -> Tensor:
    return torch.sqrt(reduce_sum(input ** 2, axis=axis, keepdims=keepdims))


@jit
def norm(input: Tensor,
         axis: Optional[List[int]] = None,
         p: float = 2,
         keepdims: bool = False) -> Tensor:
    """
    Calculate the Lp-norm of a tensor for specified axis.

    Args:
        input:
        axis: The axis to reduce for computing Lp-norm.
            If :obj:`None`, all axis will be reduced.
        p: The `p` of the `Lp` norm.  Defaults to 2.
        keepdims: Whether or not to keep the reduced dimensions?
            Defaults to :obj:`False`.

    Returns:
        The Lp-norm of the tensor.
    """
    if p == 2:
        return l2_norm(input, axis, keepdims)
    elif p == 1:
        return l1_norm(input, axis, keepdims)
    else:
        p_inv = 1. / p
        return pow(
            reduce_sum(pow(abs(input), p), axis=axis, keepdims=keepdims),
            p_inv
        )


@jit
def norm_except_axis(input: Tensor,
                     axis: int,
                     p: float = 2,
                     keepdims: bool = False) -> Tensor:
    """
    Calculate the Lp-norm of a tensor except for specified axis.

    Args:
        input: The input tensor.
        axis: The axis to keep for computing Lp-norm.
            All other axis will be reduced.
        p: The `p` of the `Lp` norm.  Defaults to 2.
        keepdims: Whether or not to keep the reduced dimensions?
            Defaults to :obj:`False`.

    Returns:
        The Lp-norm of the tensor.
    """
    r = rank(input)
    if axis < -r or axis >= r:
        raise ValueError(
            f'`axis` out of range: `axis` is {axis}, '
            f'while the shape of `input` is {shape(input)}.')
    if axis < 0:
        axis = axis + r
    axis_reduce: List[int] = []
    for a in range(0, r):
        if a != axis:
            axis_reduce.append(a)
    return norm(input, axis_reduce, p, keepdims)


@jit
def global_norm(inputs: List[Tensor]) -> Tensor:
    """
    Calculates the global norm of `inputs`.

    .. math::

        global_norm = \\sqrt{\\sum_{i=1}^n \\left|x_i\\right|_2^2}

    Args:
        inputs: The tensors

    Returns:
        The global norm.
    """
    if len(inputs) == 0:
        return float_scalar(0.)
    else:
        ret: Tensor = torch.sum(inputs[0] ** 2)
        for t in inputs[1:]:
            ret = ret + torch.sum(t ** 2)
        return torch.sqrt(ret)


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
def clip(x: Tensor, min_val: float, max_val: float) -> Tensor:
    return torch.clamp(x, min_val, max_val)


@jit
def clip_right(x: Tensor, max_val: float) -> Tensor:
    return torch.min(x, float_scalar_like(max_val, x))


@jit
def clip_left(x: Tensor, min_val: float) -> Tensor:
    return torch.max(x, float_scalar_like(min_val, x))


@jit
def clip_by_norm(input: Tensor,
                 clip_norm: float,
                 axis: Optional[List[int]] = None) -> Tensor:
    input_norm = l2_norm(input, axis=axis, keepdims=True)
    scale = torch.min(
        clip_norm / input_norm,
        float_scalar_like(1.0, input)
    )
    return input * scale


@jit
def clip_by_global_norm(inputs: List[Tensor],
                        clip_norm: float) -> List[Tensor]:
    input_global_norm = global_norm(inputs)
    scale = torch.min(
        clip_norm / input_global_norm,
        float_scalar_like(1.0, input_global_norm)
    )
    return [input * scale for input in inputs]


@jit
def maybe_clip(x: Tensor,
               min_val: Optional[float] = None,
               max_val: Optional[float] = None) -> Tensor:
    if min_val is not None and max_val is not None:
        return clip(x, min_val, max_val)
    elif min_val is not None:
        return torch.max(x, torch.as_tensor(min_val, dtype=x.dtype, device=x.device))
    elif max_val is not None:
        return torch.min(x, torch.as_tensor(max_val, dtype=x.dtype, device=x.device))
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
def is_any(condition: Tensor) -> bool:
    return bool(torch.any(condition).item())


@jit
def assert_finite(input: Tensor, message: str) -> Tensor:
    if not is_all(is_finite(input)):
        raise ValueError('Infinity or NaN value encountered: {}'.
                         format(message))
    return input
