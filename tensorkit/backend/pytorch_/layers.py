import math
import types
from functools import partial, wraps
from logging import getLogger
from typing import *

import mltk
import numpy as np
import torch
from mltk import NOT_SET
from torch import nn as torch_nn
from torch.jit import script as torch_script
from torch.nn import ModuleList

from ...typing_ import *
from ...arg_check import *
from . import init
from .core import *

__all__ = [
    # constants
    'DEFAULT_GATE_BIAS', 'DEFAULT_WEIGHT_INIT', 'DEFAULT_BIAS_INIT',

    # utils
    'jit_compile', 'jit_compile_children', 'is_jit_layer', 'layer_to_device',
    'add_parameter', 'get_parameter', 'iter_parameters', 'iter_named_parameters',
    'add_buffer', 'get_buffer', 'iter_buffers', 'iter_named_buffers',
    'get_parameters', 'get_buffers',
    'set_train_mode', 'set_eval_mode',

    # parameter store modules
    'ParamStore', 'SimpleParamStore', 'NullParamStore',
    'NormedWeightStore', 'NormedAndScaledWeightStore',
    'get_weight_store', 'get_bias_store',

    # identity layer
    'Identity',

    # base layers and composition layers
    'BaseLayer', 'ModuleList', 'Sequential',

    # linear layers
    'CoreLinear', 'Linear',
    'LinearConv1d', 'LinearConv2d', 'LinearConv3d',
    'LinearConvTranspose1d', 'LinearConvTranspose2d', 'LinearConvTranspose3d',

    # normalizer layers
    'BatchNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',
    'has_batch_norm', 'is_batch_norm', 'batch_norm_reset',
    'batch_norm_full_init',

    # dropout layers
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',

    # embedding layers
    'Embedding', 'Embedding1d', 'Embedding2d', 'Embedding3d',
]


# ---- constants ----
DEFAULT_GATE_BIAS: float = 2.0
DEFAULT_WEIGHT_INIT = init.kaming_uniform
DEFAULT_BIAS_INIT = init.zeros


# ---- utils ----
def jit_compile(m: Module,
                filter: Optional[Callable[[Module], bool]] = None,
                filter_key: Optional[Callable[[Module, str], bool]] = None) -> Module:
    """
    Compile `m` and all children modules of `m`  with JIT.

    Args:
        m: The module.
        filter: ``(m: Module) -> bool``
            Should return True if `m` should be compiled.
            The `m` module itself will not be affected by this filter.
        filter_key: ``(parent: Module, attribute: str) -> bool``.
            Should return True if the child `attribute` of `parent` should be
            compiled. The `m` module itself will not be affected by this filter.

    Returns:
        The compiled `m` module.
    """
    if is_module_jit_enabled():
        if isinstance(m, Module) and not is_jit_layer(m):
            m = jit_compile_children(m, filter, filter_key)
            m = torch_script(m)
    return m


def jit_compile_children(m: Module,
                         filter: Optional[Callable[[Module], bool]] = None,
                         filter_key: Optional[Callable[[Module, str], bool]] = None) -> Module:
    """
    Compile all children modules of `m` in-place with JIT.

    Args:
        m: The parent module.
        filter: ``(m: Module) -> bool``
            Should return True if `m` should be compiled.
        filter_key: ``(parent: Module, attribute: str) -> bool``.
            Should return True if the child `attribute` of `parent` should be
            compiled.

    Returns:
        The `m` module itself.
    """
    filter = filter or (lambda module: True)
    filter_key = filter_key or (lambda parent, attribute: True)

    if is_module_jit_enabled():
        if hasattr(m, 'custom_compile_module'):
            m.custom_compile_module()
        else:
            m_allowed_attributes = set(
                list(getattr(m, 'annotations', [])) +
                list(getattr(m, '__constants__', []))
            )

            for attr in dir(m):
                val = getattr(m, attr, None)
                if (not attr.startswith('_') or attr in m_allowed_attributes) and \
                        isinstance(val, Module) and \
                        not is_jit_layer(val) and \
                        filter(val) and \
                        filter_key(m, attr):
                    if isinstance(val, ModuleList):
                        module_list = ModuleList([
                            jit_compile(c, filter, filter_key)
                            for c in val
                        ])
                        setattr(m, attr, jit_compile(module_list, filter, filter_key))
                    else:
                        val = jit_compile(val, filter, filter_key)
                        setattr(m, attr, val)
    return m


def is_jit_layer(layer: Module) -> bool:
    """Check whether or not `layer` is a JIT compiled layer."""
    return isinstance(layer, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule))


def layer_to_device(layer: Module, device: Optional[str] = None) -> Module:
    """
    Move the specified module or layer to the given device.
    The module or layer may be changed in-place.

    Args:
        layer: The module or layer to be moved.
        device: The device, to where move the module or layer.
            If not specified, will move to ``T.current_device()``.

    Returns:
        The layer instance.
    """
    if device is None:
        device = current_device()
    layer = layer.to(device=torch.device(device))
    return layer


def add_parameter(layer: Module,
                  name: str,
                  value: Optional[Tensor],
                  requires_grad: bool = True
                  ) -> Optional[Variable]:
    if value is not None:
        v = torch.nn.Parameter(value, requires_grad=requires_grad)
    else:
        v = value
    layer.register_parameter(name, v)
    return v


def get_parameter(layer: Module, name: str) -> Optional[Variable]:
    return getattr(layer, name)


def iter_parameters(layer: Module, recursive: bool = True
                    ) -> Iterator[Variable]:
    return layer.parameters(recurse=recursive)


def get_parameters(layer: Module, recursive: bool = True) -> List[Variable]:
    return list(iter_parameters(layer, recursive))


def iter_named_parameters(layer: Module, recursive: bool = True
                          ) -> Iterator[Tuple[str, Variable]]:
    return layer.named_parameters(recurse=recursive)


def add_buffer(layer: Module,
               name: str,
               value: Optional[Tensor]
               ) -> Optional[Tensor]:
    layer.register_buffer(name, value)
    return value


def get_buffer(layer: Module, name: str) -> Optional[Tensor]:
    return getattr(layer, name)


def iter_buffers(layer: Module, recursive: bool = True
                 ) -> Iterator[Tensor]:
    return layer.buffers(recurse=recursive)


def get_buffers(layer: Module, recursive: bool = True) -> List[Variable]:
    return list(iter_buffers(layer, recursive))


def iter_named_buffers(layer: Module, recursive: bool = True
                       ) -> Iterator[Tuple[str, Tensor]]:
    return layer.named_buffers(recurse=recursive)


def set_train_mode(layer: Module, training: bool = True):
    layer.train(training)
    return layer


def set_eval_mode(layer: Module):
    layer.train(False)
    return layer


def with_layer_args(cls):
    def make_wrapper(type_, method):
        @wraps(method)
        def wrapper(*args, **kwargs):
            from tensorkit.layers import get_default_layer_args
            layer_args = get_default_layer_args()
            default_kwargs = layer_args.get_kwargs(type_)
            if default_kwargs:
                getLogger(__name__).debug(
                    'Build instance of layer %r with default kwargs %r',
                    type_, default_kwargs
                )
                for k, v in default_kwargs.items():
                    kwargs.setdefault(k, v)
            return method(*args, **kwargs)

        type_.__with_layer_args_decorated__ = True
        return wrapper

    if not isinstance(cls, type) or not issubclass(cls, Module):
        raise TypeError(f'`with_layer_args` can only be applied on a Module class.')

    cls.__init__ = make_wrapper(cls, cls.__init__)
    return cls


# ---- weight wrapper: a simple weight, or a normed weight ----
class NullParamStore(Module):
    # This module is actually not used in any context.
    # It is just a place-holder module, to gain JIT support.

    __constants__ = ('device',)

    device: str

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = device or current_device()

    def forward(self) -> Tensor:  # pragma: no cover
        zero_shape: List[int] = []
        return zeros(zero_shape, dtype='float32', device=self.device)


class ParamStore(Module):
    """
    Base class for a component that stores a trainable parameter,
    or a set of trainable parameters that can be used to derive
    virtually "a parameter" (e.g., weight-normed `weight`).
    """

    __constants__ = ('shape',)

    shape: List[int]

    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = list(map(int, shape))

    def extra_repr(self) -> str:
        return f'shape={self.shape}'

    def forward(self) -> Tensor:
        return self.get()

    def get(self) -> Tensor:
        raise NotImplementedError()

    def set(self, value: TensorOrData) -> None:
        raise NotImplementedError()


class SimpleParamStore(ParamStore):
    """A module that carries a direct variable as the parameter."""

    value: Tensor

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 device: Optional[str] = None):
        device = device or current_device()
        super().__init__(shape)
        add_parameter(self, 'value', variable(
            shape, initializer=initializer, device=device))

    @jit_method
    def get(self) -> Tensor:
        return self.value

    @jit_ignore
    def set(self, value: TensorOrData) -> None:
        with no_grad():
            assign_data(self.value, value)


@jit
def weight_norm_decompose(weight: Tensor,
                          reduce_axis: List[int],
                          epsilon: float
                          ) -> Tuple[Tensor, Tensor]:
    """
    Decompose `weight` by "weight-norm", i.e., into direction part `v`
    and norm part `v_norm`.

    Args:
        weight: The weight to be decomposed.
        reduce_axis: The axis to be reduced when calculating the weight norm.
        epsilon: Infinitesimal constant to avoid dividing zero.

    Returns:
        A tuple of `(v, v_norm)`.
    """
    v_norm = torch.sqrt(reduce_sum(weight ** 2, axis=reduce_axis, keepdims=True))
    v = weight / torch.max(v_norm, float_scalar_like(epsilon, v_norm))
    return v, v_norm


class _BaseNormedWeightStore(ParamStore):

    __constants__ = ParamStore.__constants__ + ('reduce_axis', 'epsilon')

    reduce_axis: List[int]
    epsilon: float

    def __init__(self,
                 shape: List[int],
                 axis: int = 1,
                 epsilon: float = EPSILON):
        shape = list(map(int, shape))
        r = len(shape)
        if axis < -r or axis >= r:
            raise ValueError(f'`axis` out of range: `axis` {axis} vs '
                             f'shape` {shape!r}.')
        if axis < 0:
            axis += r

        super().__init__(shape)
        self.reduce_axis = [a for a in range(0, r) if a != axis]
        self.epsilon = epsilon


class NormedWeightStore(_BaseNormedWeightStore):
    """A module that carries the weight-normed `weight`, without `g`."""

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 axis: int = 1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        weight = variable(shape, initializer=initializer, device=device)
        super().__init__(shape, axis, epsilon)
        with no_grad():
            v, _ = weight_norm_decompose(weight, self.reduce_axis, epsilon)
        add_parameter(self, 'v', v)

    @jit_method
    def get(self) -> Tensor:
        v, _ = weight_norm_decompose(self.v, self.reduce_axis, self.epsilon)
        return v

    @jit_ignore
    def set(self, value: TensorOrData) -> None:
        with no_grad():
            v, _ = weight_norm_decompose(
                as_tensor(value, dtype=get_dtype(self.v), device=get_device(self.v)),
                self.reduce_axis,
                self.epsilon,
            )
            assign_data(self.v, v)


class NormedAndScaledWeightStore(_BaseNormedWeightStore):
    """A module that carries the weight-normed `weight`, with `v` and `g`."""

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 axis: int = 1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        weight = variable(shape, initializer=initializer, device=device)
        super().__init__(shape, axis, epsilon)
        with no_grad():
            v, g = weight_norm_decompose(weight, self.reduce_axis, epsilon)
        add_parameter(self, 'v', v)
        add_parameter(self, 'g', g)

    @jit_method
    def get(self) -> Tensor:
        v, _ = weight_norm_decompose(self.v, self.reduce_axis, self.epsilon)
        return self.g * v

    @jit_ignore
    def set(self, value: TensorOrData) -> None:
        with no_grad():
            v, g = weight_norm_decompose(
                as_tensor(value, dtype=get_dtype(self.v), device=get_device(self.v)),
                self.reduce_axis,
                self.epsilon,
            )
            assign_data(self.v, v)
            assign_data(self.g, g)


def get_weight_store(shape: List[int],
                     initializer: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                     norm_axis: int = 1,
                     weight_norm: WeightNormArgType = False,
                     device: Optional[str] = None,
                     ) -> ParamStore:
    """
    Create a module which carries the `weight` parameter.

    Args:
        shape: The shape of the weight.
        initializer: The initializer for the weight.
        norm_axis: The axis, along with to normalize the weight.
        weight_norm: The mode of weight norm.
            Use `NormedAndScaledWeightStore` if `True` or `WeightNormMode.FULL`.
            Use `NormedWeightStore` if `WeightNormMode.NO_SCALE`.
            Use `WeightStore` if `False` or `WeightNormMode.NONE`.
        device: The device where to place new tensors and variables.

    Returns:
        The weight object.
    """
    device = device or current_device()
    if weight_norm is True or weight_norm == WeightNormMode.FULL:
        return NormedAndScaledWeightStore(shape, initializer, norm_axis, device)
    elif weight_norm == WeightNormMode.NO_SCALE:
        return NormedWeightStore(shape, initializer, norm_axis, device)
    elif weight_norm is False or weight_norm == WeightNormMode.NONE:
        return SimpleParamStore(shape, initializer, device)
    else:
        raise ValueError(f'Invalid value for argument `weight_norm`: '
                         f'{weight_norm!r}.')


def get_bias_store(shape: List[int],
                   initializer: TensorInitArgType = DEFAULT_BIAS_INIT,
                   use_bias: bool = True,
                   device: Optional[str] = None
                   ) -> Optional[ParamStore]:
    """
    Create a module that carries the `bias` parameter.

    Args:
        shape: The shape of the bias.
        initializer: The initializer for the bias.
        use_bias: Whether or not to use the bias?
            If `False`, will return :obj:`None`.
        device: The device where to place new tensors and variables.

    Returns:
        The bias object, or :obj:`None` if `use_bias` is False.
    """
    device = device or current_device()
    if use_bias:
        return SimpleParamStore(shape, initializer, device)


# ---- identity layer ----
class Identity(Module):

    def forward(self, input: Tensor) -> Tensor:
        return input


# ---- base layers and composition layers ----
class BaseLayerMeta(type):

    def __new__(cls, name, parents, dct):
        if torch.__version__ != '1.3.1':
            # strange bug, that PyTorch >= 1.4.0 does not support annotations
            # with type `Module` or `ModuleList`
            if '__annotations__' in dct:
                annotations = dct['__annotations__']
                annotation_keys = list(annotations)
                for attr in annotation_keys:
                    if annotations[attr] in (Module, ModuleList):
                        annotations.pop(attr)

        kclass = super().__new__(cls, name, parents, dct)
        kclass = with_layer_args(kclass)
        return kclass


class BaseLayer(Module, metaclass=BaseLayerMeta):

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        if attr in getattr(Module, '__annotations__', ()):
            return False
        # if callable(value) or isinstance(value, types.MethodType):
        #     return False
        return True

    def extra_repr(self) -> str:
        buf = []
        attributes = list(getattr(self, '__constants__', ()))

        for attr in getattr(self, '__annotations__', ()):
            if attr not in attributes:
                attributes.append(attr)

        for attr in attributes:
            if attr.startswith('_'):
                continue
            attr_val = getattr(self, attr, None)
            if attr_val is None or \
                    isinstance(attr_val, Module) or \
                    isinstance(attr_val, Tensor) or \
                    is_jit_layer(attr_val):
                continue
            if self._is_attr_included_in_repr(attr, attr_val):
                buf.append(f'{attr}={attr_val!r}')

        return ', '.join(buf)


class Sequential(torch_nn.Sequential, metaclass=BaseLayerMeta):

    def __init__(self, *layers: Union[Module, Sequence[Module]]):
        from tensorkit.layers import flatten_nested_layers
        super().__init__(*flatten_nested_layers(layers))


# ---- linear layers ----
class CoreLinear(BaseLayer):
    """Base class for the core linear layers."""

    __constants__ = ('use_bias',)

    weight_store: Module
    bias_store: Module
    use_bias: bool

    def __init__(self,
                 weight_shape: List[int],
                 bias_shape: List[int],
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        device = device or current_device()
        weight_store = get_weight_store(
            weight_shape, initializer=weight_init, weight_norm=weight_norm,
            device=device
        )
        bias_store = get_bias_store(
            bias_shape, initializer=bias_init, use_bias=use_bias, device=device)
        if bias_store is None:
            bias_store = NullParamStore(device=device)

        if data_init is not None:
            if not isinstance(data_init, init.DataDependentInitializer) and \
                    (isinstance(data_init, type) or callable(data_init)):
                data_init = data_init()

            if not isinstance(data_init, init.DataDependentInitializer):
                raise TypeError(f'Unsupported data dependent initializer: '
                                f'{data_init!r}')

        super().__init__()
        self.weight_store = weight_store
        self.bias_store = bias_store
        self.use_bias = use_bias

        if data_init is not None:
            data_init.register(self)

    def _is_attr_included_in_repr(self, attr: str, value: Any) -> bool:
        return attr != 'use_bias' or not value

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.extra_repr()})'


class Linear(CoreLinear):

    __constants__ = CoreLinear.__constants__ + (
        'in_features', 'out_features',
    )

    in_features: int
    out_features: int

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        in_features = validate_positive_int('in_features', in_features)
        out_features = validate_positive_int('out_features', out_features)

        self.in_features = in_features
        self.out_features = out_features

        super().__init__(
            weight_shape=[out_features, in_features],
            bias_shape=[out_features],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
            device=device,
        )

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_store()
        if self.use_bias:
            bias: Optional[Tensor] = self.bias_store()
        else:
            bias: Optional[Tensor] = None
        return torch.nn.functional.linear(input, weight, bias)


@jit_ignore
def _get_manual_and_conv_padding(padding: List[Tuple[int, int]]):
    if all(p1 == p2 for p1, p2 in padding):
        return [], [p1 for p1, _ in padding],
    else:
        return sum([[p1, p2] for p1, p2 in reversed(padding)], []), [0] * len(padding)


class LinearConvNd(CoreLinear):

    __constants__ = CoreLinear.__constants__ + (
        'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
        'dilation', '_manual_padding', '_conv_padding', '_is_manual_padding',
    )

    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[Tuple[int, int]]
    dilation: List[int]
    _manual_padding: List[int]
    _conv_padding: List[int]
    _is_manual_padding: bool

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 dilation: Union[int, Sequence[int]] = 1,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        spatial_ndims = self._get_spatial_ndims()
        in_channels = validate_positive_int('in_channels', in_channels)
        out_channels = validate_positive_int('out_channels', out_channels)
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)
        _manual_padding, _conv_padding = _get_manual_and_conv_padding(padding)

        super().__init__(
            weight_shape=[out_channels, in_channels] + kernel_size,
            bias_shape=[out_channels],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
            device=device,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self._manual_padding = _manual_padding
        self._conv_padding = _conv_padding
        self._is_manual_padding = len(_manual_padding) > 0

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_store()
        if self.use_bias:
            bias: Optional[Tensor] = self.bias_store()
        else:
            bias: Optional[Tensor] = None
        if self._is_manual_padding:
            input = torch.nn.functional.pad(
                input, self._manual_padding, mode='constant', value=0.)
        return self._conv_transform(input, weight, bias)

    @jit_method
    def _conv_transform(self,
                        input: Tensor,
                        weight: Tensor,
                        bias: Optional[Tensor]
                        ) -> Tensor:
        raise NotImplementedError()

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class LinearConv1d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    @jit_method
    def _conv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        return torch.nn.functional.conv1d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self._conv_padding, dilation=self.dilation
        )


class LinearConv2d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    @jit_method
    def _conv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        return torch.nn.functional.conv2d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self._conv_padding, dilation=self.dilation
        )


class LinearConv3d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    @jit_method
    def _conv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        return torch.nn.functional.conv3d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self._conv_padding, dilation=self.dilation
        )


class LinearConvTransposeNd(CoreLinear):

    __constants__ = CoreLinear.__constants__ + (
        'in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
        'dilation', 'output_padding',
        '_manual_padding', '_conv_padding', '_is_manual_padding', '_unpad_axis',
    )

    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[Tuple[int, int]]
    dilation: List[int]
    output_padding: List[int]
    _manual_padding: List[int]
    _conv_padding: List[int]
    _is_manual_padding: bool
    _unpad_axis: List[int]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]] = 1,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 output_padding: Union[int, Sequence[int]] = 0,
                 dilation: Union[int, Sequence[int]] = 1,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        spatial_ndims = self._get_spatial_ndims()
        in_channels = validate_positive_int('in_channels', in_channels)
        out_channels = validate_positive_int('out_channels', out_channels)
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)
        output_padding = validate_output_padding(
            output_padding, stride, dilation, spatial_ndims)
        _manual_padding, _conv_padding = _get_manual_and_conv_padding(padding)

        super().__init__(
            weight_shape=[in_channels, out_channels] + kernel_size,
            bias_shape=[out_channels],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
            device=device,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self._manual_padding = _manual_padding
        self._conv_padding = _conv_padding
        self._is_manual_padding = len(_manual_padding) > 0
        self._unpad_axis = list(range(-1, -(spatial_ndims + 1), -1))

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    @jit_method
    def _deconv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_store()
        if self.use_bias:
            bias: Optional[Tensor] = self.bias_store()
        else:
            bias: Optional[Tensor] = None
        input = self._deconv_transform(input, weight, bias)
        if self._is_manual_padding:
            for a in self._unpad_axis:
                p1, p2 = self.padding[a]
                length = input.shape[a] - p1 - p2
                if length < 0:
                    raise ValueError('Too large `padding` at axis {}: ({}, {})'.
                                     format(a, p1, p2))
                input = torch.narrow(input, a, p1, length)
        return input


class LinearConvTranspose1d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    @jit_method
    def _deconv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        return torch.nn.functional.conv_transpose1d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self._conv_padding, output_padding=self.output_padding,
            dilation=self.dilation
        )


class LinearConvTranspose2d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    @jit_method
    def _deconv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        return torch.nn.functional.conv_transpose2d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self._conv_padding, output_padding=self.output_padding,
            dilation=self.dilation
        )


class LinearConvTranspose3d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    @jit_method
    def _deconv_transform(self,
                          input: Tensor,
                          weight: Tensor,
                          bias: Optional[Tensor]
                          ) -> Tensor:
        return torch.nn.functional.conv_transpose3d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self._conv_padding, output_padding=self.output_padding,
            dilation=self.dilation
        )


# ---- normalizer layers ----
class BatchNorm(torch_nn.BatchNorm1d, metaclass=BaseLayerMeta):
    """Batch normalization for dense layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if rank(input) != 2:
            raise ValueError('`BatchNorm` only supports 2d input, '
                             'but the input shape is {}'.format(shape(input)))


class BatchNorm1d(torch_nn.BatchNorm1d, metaclass=BaseLayerMeta):
    """Batch normalization for 1D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if rank(input) != 3:
            raise ValueError('`BatchNorm1d` only supports 3d input, '
                             'but the input shape is {}'.format(shape(input)))


class BatchNorm2d(torch_nn.BatchNorm2d, metaclass=BaseLayerMeta):
    """Batch normalization for 2D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 4:
            raise ValueError('`BatchNorm2d` only supports 4d input, '
                             'but the input shape is {}'.format(shape(input)))


class BatchNorm3d(torch_nn.BatchNorm3d, metaclass=BaseLayerMeta):
    """Batch normalization for 3D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 device: Optional[str] = None,
                 epsilon: float = EPSILON):
        device = device or current_device()
        super().__init__(num_features, eps=epsilon, momentum=momentum)
        if device != CPU_DEVICE:
            self.to(device=device)

    def _check_input_dim(self, input: Tensor):
        if rank(input) != 5:
            raise ValueError('`BatchNorm2d` only supports 5d input, '
                             'but the input shape is {}'.format(shape(input)))


def has_batch_norm(module: Module, recursive: bool = True) -> bool:
    """
    Check whether or not `module` is a Batch Norm module (if `recursive` is
    False), or any children of `module` is Batch Norm module (if `recursive`
    is True).

    JIT compiled Batch Norm module is not recognized by this method.

    Args:
        module: The module to be checked.
        recursive: Whether or not to check the module recursively.

    Returns:
        Whether or not `module` is a Batch Norm module.
    """
    ret = [False]

    def fn(m):
        ret[0] = ret[0] or isinstance(
            m,
            (torch_nn.BatchNorm1d, torch_nn.BatchNorm2d, torch_nn.BatchNorm3d)
        )

    if recursive:
        module.apply(fn)
    else:
        fn(module)

    return ret[0]


def is_batch_norm(module: Module) -> bool:
    """
    Check whether or not `module` is a Batch Norm module.
    JIT compiled Batch Norm module is not recognized by this method.

    Args:
        module: The module to be checked.

    Returns:
        Whether or not `module` is a Batch Norm module.
    """
    return has_batch_norm(module, recursive=False)


def batch_norm_reset(model: Module):
    """
    Reset the running statistics of Batch Norm modules in a model recursively.
    JIT compiled Batch Norm modules are not supported.

    Args:
        model: The root module of the module.

    Returns:
        The `model` itself.
    """
    def fn(m: Module):
        if is_batch_norm(m):
            m.running_mean = torch.zeros_like(
                m.running_mean, device=m.running_mean.device)
            m.running_var = torch.ones_like(
                m.running_var, device=m.running_var.device)
    model.apply(fn)
    return model


def batch_norm_full_init(model: Module,
                         data_generator,
                         step_fn: Callable[[Sequence[Tensor]], None],
                         loop=None) -> Module:
    """
    Run a full epoch to initialize the mean and variance of BatchNorm layers
    in the given `model` recursively.  JIT compiled BatchNorm layers are not
    supported.

    Args:
        model: The model, whose BatchNorm children to be initialized.
        data_generator: The epoch data generator.
        step_fn: The callback function, which executes each mini-batch.
        loop: Optional mltk loop object.

    Returns:
        The `model` itself.

    Notes:
        Remember to set the train mode of `model` before calling this method.
    """

    # backup the momentum of all BatchNorm layers
    def backup_momentum(m):
        if is_batch_norm(m):
            orig_momentum[m] = m.momentum

    def restore_momentum(m):
        if m in orig_momentum:
            m.momentum = orig_momentum[m]

    orig_momentum = {}
    model.apply(backup_momentum)
    batch_norm_reset(model)

    # run the full epoch to update the BatchNorm stats
    if orig_momentum:
        try:
            def step(*batch_data):
                batch_size = mltk.utils.get_array_shape(batch_data[0])[0]
                total_size[0] += batch_size
                for m in orig_momentum.keys():
                    m.momentum = float(batch_size) / (total_size[0])
                step_fn(*batch_data)

            total_size = [0]
            batch_norm_reset(model)
            if isinstance(loop, mltk.TrainLoop):
                for _ in loop.iter_epochs(count=1):
                    loop.run_batches(step, data_generator)
            elif loop is not None:
                loop.run(step, data_generator)
            else:
                for b in data_generator:
                    step(*b)

        finally:
            # restore the momentum
            model.apply(restore_momentum)

    return model


# ---- dropout layers ----
class Dropout(torch_nn.Dropout, metaclass=BaseLayerMeta):
    pass


class Dropout1d(BaseLayer):
    """Randomly zero out entire channels of the 1d convolution input."""

    __constants__ = ('p', '_keep_prob')

    p: float
    _keep_prob: float

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._keep_prob = 1. - p

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() < 2:  # pragma: no cover
            raise ValueError('`input` must be at least 2d, but the '
                             'input shape is {}.'.format(shape(input)))

        device = input.device
        if self.training:
            noise_shape = input.shape[:-1] + (1,)
            noise = torch.zeros(noise_shape, dtype=input.dtype, device=device)
            keep_prob = torch.as_tensor(self._keep_prob, dtype=input.dtype, device=device)
            noise = torch.bernoulli(keep_prob.expand(noise_shape), out=noise)
            noise = noise.detach()
            input = input * noise / keep_prob
        return input


class Dropout2d(torch_nn.Dropout2d, metaclass=BaseLayerMeta):
    pass


class Dropout3d(torch_nn.Dropout3d, metaclass=BaseLayerMeta):
    pass


# ---- embedding layers ----
class Embedding(BaseLayer):

    def __init__(self,
                 n_embeddings: int,
                 embedding_size: Union[int, List[int]],
                 initializer: TensorInitArgType = NOT_SET,
                 freeze: bool = False,
                 device: Optional[str] = None,
                 ):
        n_embeddings = int(n_embeddings)

        if hasattr(embedding_size, '__iter__'):
            embedding_size = list(map(int, embedding_size))
        else:
            embedding_size = [int(embedding_size)]
        if not embedding_size:
            raise ValueError(f'`embedding_size` must not be empty.')

        if initializer is NOT_SET:
            std = 1. / math.sqrt(embedding_size[0])
            a = math.sqrt(3.0) * std  # such that U(-a, a) will have standard deviation `std`
            initializer = partial(init.uniform, low=-a, high=a)

        super().__init__()
        w = variable(shape=[n_embeddings] + embedding_size, device=device,
                     initializer=initializer)
        add_parameter(self, 'weight', w, requires_grad=not freeze)

    def forward(self, input: Tensor) -> Tensor:
        return embedding(self.weight, input)


class EmbeddingNd(Embedding):

    def __init__(self,
                 n_embeddings: int,
                 embedding_size: List[int],
                 initializer: TensorInitArgType = NOT_SET,
                 freeze: bool = False,
                 device: Optional[str] = None,
                 ):
        spatial_ndims = self._get_spatial_ndims()
        if len(embedding_size) != spatial_ndims + 1:
            raise ValueError(
                f'`embedding_size` must be a int list with {spatial_ndims + 1} '
                f'elements: got {embedding_size!r}.')
        embedding_size = list(map(int, embedding_size))
        super().__init__(n_embeddings, embedding_size, initializer, freeze, device)

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class Embedding1d(EmbeddingNd):

    def _get_spatial_ndims(self) -> int:
        return 1


class Embedding2d(EmbeddingNd):

    def _get_spatial_ndims(self) -> int:
        return 2


class Embedding3d(EmbeddingNd):

    def _get_spatial_ndims(self) -> int:
        return 3
