from typing import *

import torch
from torch import nn as torch_nn
from torch.jit import ScriptModule
from torch.nn import ModuleList

from . import init
from .core import *
from ...typing_ import *
from ...arg_check import *

__all__ = [
    # constants
    'DEFAULT_GATE_BIAS', 'DEFAULT_WEIGHT_INIT', 'DEFAULT_BIAS_INIT',

    # utils
    'add_parameter', 'get_parameter', 'get_parameters',
    'add_buffer', 'get_buffer', 'get_buffers',
    'set_train_mode',

    # parameter store modules
    'BaseParamStore', 'SimpleParamStore',
    'NormedWeightStore', 'NormedAndScaledWeightStore',
    'get_weight_store', 'get_bias_store',

    # identity layer
    'Identity',

    # base layers and composition layers
    'BaseLayer', 'BaseSingleVariateLayer', 'BaseMultiVariateLayer',
    'BaseSplitLayer', 'BaseMergeLayer',
    'ModuleList', 'Sequential',
    'BaseContextualLayer',

    # linear layers
    'CoreLinear', 'Linear',
    'LinearConv1d', 'LinearConv2d', 'LinearConv3d',
    'LinearConvTranspose1d', 'LinearConvTranspose2d', 'LinearConvTranspose3d',

    # normalizer layers
    'BatchNorm', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d',

    # dropout layers
    'Dropout', 'Dropout1d', 'Dropout2d', 'Dropout3d',
]


# ---- constants ----
DEFAULT_GATE_BIAS: float = 2.0
DEFAULT_WEIGHT_INIT = init.kaming_uniform
DEFAULT_BIAS_INIT = init.zeros


# ---- utils ----
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


def get_parameters(layer: Module, recursive: bool = True
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


def get_buffers(layer: Module, recursive: bool = True
                ) -> Iterator[Tuple[str, Tensor]]:
    return layer.named_buffers(recurse=recursive)


def set_train_mode(layer: Module, training: bool = True):
    layer.train(training)
    return layer


# ---- weight wrapper: a simple weight, or a normed weight ----
class BaseParamStore(Module):
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


class SimpleParamStore(BaseParamStore):
    """A module that carries a direct variable as the parameter."""

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType):
        super().__init__(shape)
        add_parameter(self, 'value', variable(shape, initializer=initializer))

    @jit_method
    def get(self) -> Tensor:
        return self.value

    def set(self, value: TensorOrData) -> None:
        with no_grad():
            assign_data(self.value, value)


@jit
def weight_norm_decompose(weight: Tensor,
                          norm_axis: int,
                          epsilon: float
                          ) -> Tuple[Tensor, Tensor]:
    """
    Decompose `weight` by "weight-norm", i.e., into direction part `v`
    and norm part `v_norm`.

    Args:
        weight: The weight to be decomposed.
        norm_axis: The axis, along with to calculate the weight norm.
        epsilon: Infinitesimal constant to avoid dividing zero.

    Returns:
        A tuple of `(v, v_norm)`.
    """
    v_norm = norm_except_axis(weight, axis=[norm_axis], keepdims=True)
    v = weight / torch.max(v_norm, torch.as_tensor(epsilon, dtype=v_norm.dtype))
    return v, v_norm


class NormedWeightStore(BaseParamStore):
    """A module that carries the weight-normed `weight`, without `g`."""

    __constants__ = BaseParamStore.__constants__ + ('feature_axis', 'epsilon')

    norm_axis: int
    epsilon: float

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 norm_axis: int = 1,
                 epsilon: float = 1e-5):
        super().__init__(shape)
        self.norm_axis = norm_axis
        self.epsilon = epsilon

        weight = variable(shape, initializer=initializer)
        with no_grad():
            v, _ = weight_norm_decompose(weight, norm_axis, epsilon)
        add_parameter(self, 'v', v)

    @jit_method
    def get(self) -> Tensor:
        v, _ = weight_norm_decompose(self.v, self.norm_axis, self.epsilon)
        return v

    def set(self, value: TensorOrData) -> None:
        with no_grad():
            v, _ = weight_norm_decompose(
                as_tensor(value, dtype=get_dtype(self.v)),
                self.norm_axis,
                self.epsilon,
            )
            assign_data(self.v, v)


class NormedAndScaledWeightStore(BaseParamStore):
    """A module that carries the weight-normed `weight`, with `v` and `g`."""

    __constants__ = BaseParamStore.__constants__ + ('feature_axis', 'epsilon')

    norm_axis: int
    epsilon: float

    def __init__(self,
                 shape: List[int],
                 initializer: TensorInitArgType,
                 norm_axis: int = 1,
                 epsilon: float = 1e-5):
        super().__init__(shape)
        self.norm_axis = norm_axis
        self.epsilon = epsilon

        weight = variable(shape, initializer=initializer)
        with no_grad():
            v, g = weight_norm_decompose(weight, norm_axis, epsilon)
        add_parameter(self, 'v', v)
        add_parameter(self, 'g', g)

    @jit_method
    def get(self) -> Tensor:
        v, _ = weight_norm_decompose(self.v, self.norm_axis, self.epsilon)
        return self.g * v

    def set(self, value: TensorOrData) -> None:
        with no_grad():
            v, g = weight_norm_decompose(
                as_tensor(value, dtype=get_dtype(self.v)),
                self.norm_axis,
                self.epsilon,
            )
            assign_data(self.v, v)
            assign_data(self.g, g)


def get_weight_store(shape: List[int],
                     initializer: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                     norm_axis: int = 1,
                     weight_norm: WeightNormArgType = False
                     ) -> BaseParamStore:
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

    Returns:
        The weight object.
    """
    if weight_norm is True or weight_norm == WeightNormMode.FULL:
        return NormedAndScaledWeightStore(shape, initializer, norm_axis)
    elif weight_norm == WeightNormMode.NO_SCALE:
        return NormedWeightStore(shape, initializer, norm_axis)
    elif weight_norm is False or weight_norm == WeightNormMode.NONE:
        return SimpleParamStore(shape, initializer)
    else:
        raise ValueError(f'Invalid value for argument `weight_norm`: '
                         f'{weight_norm!r}.')


def get_bias_store(shape: List[int],
                   initializer: TensorInitArgType = DEFAULT_BIAS_INIT,
                   use_bias: bool = True
                   ) -> Optional[BaseParamStore]:
    """
    Create a module that carries the `bias` parameter.

    Args:
        shape: The shape of the bias.
        initializer: The initializer for the bias.
        use_bias: Whether or not to use the bias?
            If `False`, will return :obj:`None`.

    Returns:
        The bias object, or :obj:`None` if `use_bias` is False.
    """
    if use_bias:
        return SimpleParamStore(shape, initializer)


# ---- identity layer ----
class Identity(Module):

    def forward(self, input: Tensor) -> Tensor:
        return input


# ---- base layers and composition layers ----
class BaseLayer(Module):

    def extra_repr(self) -> str:
        buf = []
        attributes = list(getattr(self, '__constants__', ()))

        for attr in getattr(self, '__annotations__', ()):
            if attr not in attributes:
                attributes.append(attr)

        for attr in attributes:
            attr_val = getattr(self, attr, None)
            if attr_val is None or isinstance(attr_val, Module) or \
                    isinstance(attr_val, Tensor):
                continue
            buf.append(f'{attr}={attr_val!r}')

        return ', '.join(buf)


class BaseSingleVariateLayer(BaseLayer):
    """
    Base class for single-input, single-output layers.

    Sub-classes should override `_call(input: Tensor) -> Tensor` to
    actually implement the module.
    """

    def _call(self, input: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        return self._call(input)


class BaseMultiVariateLayer(BaseLayer):
    """
    Base class for multiple-input, multiple-output layers.
    The inputs and outputs should be given as a list of Tensors.
    """

    def _call(self, inputs: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError()

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        return self._call(inputs)


class BaseSplitLayer(BaseLayer):
    """
    Base class for single-input, multiple-output layers.
    The outputs should be given as a list of Tensors.
    """

    def _call(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> List[Tensor]:
        return self._call(input)


class BaseMergeLayer(BaseLayer):
    """
    Base class for multiple-input, single-output layers.
    The inputs should be given as a list of Tensors.
    """

    def _call(self, inputs: List[Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return self._call(inputs)


class BaseContextualLayer(BaseLayer):
    """
    Base class layers that produces the output according to  the input tensor
    and potentially a contextual tensor.
    """

    def _call(self, input: Tensor, context: Optional[Tensor]) -> Tensor:
        raise NotImplementedError()

    def forward(self, input: Tensor, context: Optional[Tensor] = None) -> Tensor:
        return self._call(input, context)


class Sequential(torch_nn.Sequential):

    def __init__(self, *layers: Union[Module, Sequence[Module]]):
        from tensorkit.layers import flatten_nested_layers
        super().__init__(*flatten_nested_layers(layers))


# ---- linear layers ----
class CoreLinear(BaseLayer):
    """Base class for the core linear layers."""

    __constants__ = (
        # modules
        'weight_store', 'bias_store',

        # attributes
        'in_features', 'out_features', 'in_channels', 'out_channels',
        'kernel_size', 'stride', 'padding', 'output_padding', 'dilation',
        'use_bias',
    )

    weight_store: Module
    bias_store: Optional[Module]

    def __init__(self,
                 weight_shape: List[int],
                 bias_shape: List[int],
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 ):
        weight_store = get_weight_store(
            weight_shape, initializer=weight_init, weight_norm=weight_norm)
        bias_store = get_bias_store(
            bias_shape, initializer=bias_init, use_bias=use_bias)

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

        if data_init is not None:
            data_init.register(self)

    def __repr__(self) -> str:
        attributes = []
        for attr in self.__annotations__:
            val = getattr(self, attr, None)
            if val is not None:
                if attr == 'use_bias':
                    if not val:
                        attributes.append(f'{attr}={val}')
                elif not isinstance(val, (Module, Tensor)):
                    attributes.append(f'{attr}={val!r}')
        return f'{self.__class__.__qualname__}({", ".join(attributes)})'

    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        weight = self.weight_store()
        if self.bias_store is None:
            bias = None
        else:
            bias = self.bias_store()
        return self._call(input, weight, bias)


class Linear(CoreLinear):

    in_features: int
    out_features: int
    use_bias: bool

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_bias: bool = True,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 ):
        in_features = validate_positive_int('in_features', in_features)
        out_features = validate_positive_int('out_features', out_features)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        super().__init__(
            weight_shape=[out_features, in_features],
            bias_shape=[out_features],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
        )

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        output, front_shape = flatten_to_ndims(input, 2)
        output = torch.nn.functional.linear(output, weight, bias)
        output = unflatten_from_ndims(output, front_shape)
        return output


class LinearConvNd(CoreLinear):

    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    use_bias: bool

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
                 ):
        spatial_ndims = self._get_spatial_ndims()
        in_channels = validate_positive_int('in_channels', in_channels)
        out_channels = validate_positive_int('out_channels', out_channels)
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        stride = validate_conv_size('stride', stride, spatial_ndims)
        dilation = validate_conv_size('dilation', dilation, spatial_ndims)
        padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.use_bias = use_bias

        super().__init__(
            weight_shape=[out_channels, in_channels] + kernel_size,
            bias_shape=[out_channels],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class LinearConv1d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        return torch.nn.functional.conv1d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation
        )


class LinearConv2d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        return torch.nn.functional.conv2d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation
        )


class LinearConv3d(LinearConvNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        return torch.nn.functional.conv3d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation
        )


class LinearConvTransposeNd(CoreLinear):

    in_channels: int
    out_channels: int
    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    dilation: List[int]
    output_padding: List[int]
    use_bias: bool

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

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.use_bias = use_bias

        super().__init__(
            weight_shape=[in_channels, out_channels] + kernel_size,
            bias_shape=[out_channels],
            use_bias=use_bias,
            weight_norm=weight_norm,
            weight_init=weight_init,
            bias_init=bias_init,
            data_init=data_init,
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class LinearConvTranspose1d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        return torch.nn.functional.conv_transpose1d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            dilation=self.dilation
        )


class LinearConvTranspose2d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        return torch.nn.functional.conv_transpose2d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            dilation=self.dilation
        )


class LinearConvTranspose3d(LinearConvTransposeNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    @jit_method
    def _call(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
              ) -> Tensor:
        return torch.nn.functional.conv_transpose3d(
            input=input, weight=weight, bias=bias, stride=self.stride,
            padding=self.padding, output_padding=self.output_padding,
            dilation=self.dilation
        )


# ---- normalizer layers ----
class BatchNorm(torch_nn.BatchNorm1d):
    """Batch normalization for dense layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = 1e-5):
        super().__init__(num_features, eps=epsilon, momentum=momentum)

    def _check_input_dim(self, input: Tensor):
        if rank(input) != 2:
            raise ValueError('`BatchNorm` only supports 2d input, '
                             'but the input shape is {}'.format(shape(input)))


class BatchNorm1d(torch_nn.BatchNorm1d):
    """Batch normalization for 1D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = 1e-5):
        super().__init__(num_features, eps=epsilon, momentum=momentum)

    def _check_input_dim(self, input: Tensor):
        if rank(input) != 3:
            raise ValueError('`BatchNorm1d` only supports 3d input, '
                             'but the input shape is {}'.format(shape(input)))


class BatchNorm2d(torch_nn.BatchNorm2d):
    """Batch normalization for 2D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = 1e-5):
        super().__init__(num_features, eps=epsilon, momentum=momentum)

    def _check_input_dim(self, input: Tensor):
        if input.dim() != 4:
            raise ValueError('`BatchNorm2d` only supports 4d input, '
                             'but the input shape is {}'.format(shape(input)))


class BatchNorm3d(torch_nn.BatchNorm3d):
    """Batch normalization for 3D convolutional layers."""

    def __init__(self,
                 num_features: int,
                 momentum: float = 0.1,
                 epsilon: float = 1e-5):
        super().__init__(num_features, eps=epsilon, momentum=momentum)

    def _check_input_dim(self, input: Tensor):
        if rank(input) != 5:
            raise ValueError('`BatchNorm2d` only supports 5d input, '
                             'but the input shape is {}'.format(shape(input)))


# ---- dropout layers ----
Dropout = torch_nn.Dropout


class Dropout1d(Module):
    """Randomly zero out entire channels of the 1d convolution input."""

    __constants__ = ('p', 'keep_prob')

    p: float
    keep_prob: float

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.keep_prob = 1. - p

    def extra_repr(self) -> str:
        return f'p={self.p}'

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() < 2:  # pragma: no cover
            raise ValueError('`input` must be at least 2d, but the '
                             'input shape is {}.'.format(shape(input)))

        output = input
        if self.training:
            noise_shape = output.shape[:-1] + (1,)
            noise = torch.zeros(noise_shape, dtype=output.dtype)
            keep_prob = torch.as_tensor(self.keep_prob, dtype=output.dtype)
            noise = torch.bernoulli(keep_prob.expand(noise_shape), out=noise)
            noise = noise.detach()
            output = output * noise / keep_prob
        return output


Dropout2d = torch_nn.Dropout2d
Dropout3d = torch_nn.Dropout3d
