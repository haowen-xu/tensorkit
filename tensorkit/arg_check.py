from typing import *

__all__ = [
    # general argument validators
    'validate_positive_int',

    # layer argument validators
    'validate_layer', 'validate_layer_factory', 'get_layer_from_layer_or_factory',
    'validate_weight_norm', 'validate_conv_size', 'validate_padding',
    'validate_output_padding',

]


# general argument validators
def validate_positive_int(arg_name: str, arg_value) -> int:
    try:
        ret = int(arg_value)
        if ret > 0:
            return ret
    except ValueError:
        pass
    raise ValueError(f'`{arg_name}` but be a positive int: '
                     f'got {arg_value!r}')


# layer argument validators
def validate_layer(arg_name: str, layer, nullable_: bool = True
                   ) -> Optional['Module']:
    if layer is None:
        if not nullable_:
            raise ValueError(f'`{arg_name}` is required.')
    elif isinstance(layer, Module):
        return layer
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer: '
                        f'got {layer!r}')


def validate_layer_factory(arg_name: str,
                           layer_factory,
                           nullable_: bool = True,
                           args=(),
                           kwargs=None):
    if layer_factory is None:
        if not nullable_:
            raise ValueError(f'`{arg_name}` is required.')
    elif isinstance(layer_factory, type) or callable(layer_factory):
        return layer_factory
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer factory: '
                        f'got {layer_factory!r}.')


def get_layer_from_layer_or_factory(arg_name: str,
                                    layer_or_layer_factory,
                                    args=(),
                                    kwargs=None) -> 'Module':
    if isinstance(layer_or_layer_factory, Module):
        return layer_or_layer_factory
    elif isinstance(layer_or_layer_factory, type) or \
            callable(layer_or_layer_factory):
        return layer_or_layer_factory(*args, **(kwargs or {}))
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer or a '
                        f'layer factory: got {layer_or_layer_factory!r}.')


def validate_weight_norm(weight_norm: 'WeightNormArgType',
                         normalizer: Optional['NormalizerOrNormalizerFactory'],
                         data_init: Optional['DataInitArgType']
                         ) -> 'WeightNormMode':
    if weight_norm is True:
        ret = (WeightNormMode.FULL if normalizer is None
               else WeightNormMode.NO_SCALE)
    elif weight_norm is False:
        ret = WeightNormMode.NONE
    else:
        ret = WeightNormMode(weight_norm)

    if ret != WeightNormMode.NONE and data_init is not None:
        raise ValueError(
            'Weight normalization cannot be used together with '
            'data-dependent initializer.'
        )

    return ret


def validate_conv_size(name: str,
                       value: Union[int, Sequence[int]],
                       spatial_ndims: int
                       ) -> List[int]:
    if not hasattr(value, '__iter__'):
        value = [int(value)] * spatial_ndims
        value_ok = value[0] > 0
    else:
        value = list(map(int, value))
        value_ok = (len(value) == spatial_ndims) and all(v > 0 for v in value)

    if not value_ok:
        raise ValueError(
            f'`{name}` must be either a positive integer, or a sequence of '
            f'positive integers with length `{spatial_ndims}`: got {value}.'
        )
    return value


def validate_padding(padding: 'PaddingArgType',
                     kernel_size: List[int],
                     dilation: List[int],
                     spatial_ndims: int
                     ) -> List[int]:
    if padding == 'none':
        return [0] * spatial_ndims
    elif padding == 'full':
        return [(kernel_size[i] - 1) * dilation[i]
                for i in range(spatial_ndims)]
    elif padding == 'half':
        padding = []
        for i in range(spatial_ndims):
            val = (kernel_size[i] - 1) * dilation[i]
            if val % 2 != 0:
                raise ValueError(
                    f'`(kernel_size - 1) * dilation` is required to be even '
                    f'for `padding` == "half": got `kernel_size` '
                    f'{kernel_size}, and `dilation` {dilation}.'
                )
            padding.append(val // 2)
        return padding
    elif hasattr(padding, '__iter__'):
        if len(padding) != spatial_ndims:
            raise ValueError(
                f'`padding` must be a positive integer, a sequence of '
                f'positive integers with length `{spatial_ndims}`, "none", '
                f'"half" or "full": got {padding}.'
            )
        return list(map(int, padding))
    else:
        return [int(padding)] * spatial_ndims


def validate_output_padding(output_padding: Union[int, Sequence[int]],
                            stride: List[int],
                            dilation: List[int],
                            spatial_ndims: int
                            ) -> List[int]:
    if hasattr(output_padding, '__iter__'):
        if len(output_padding) != spatial_ndims:
            raise ValueError(
                f'`output_padding` must be a non-negative integer, or a '
                f'sequence of non-negative integers with length '
                f'`{spatial_ndims}`: got {output_padding}.'
            )
        output_padding = list(map(int, output_padding))

    else:
        output_padding = [int(output_padding)] * spatial_ndims

    for i in range(spatial_ndims):
        if output_padding[i] < 0 or \
                output_padding[i] >= max(stride[i], dilation[i]):
            raise ValueError(
                f'`output_padding` must be non-negative, and must be smaller '
                f'than either `stride` or `dilation`, but this does not hold '
                f'at spatial axis {i}: '
                f'`output_padding` is {output_padding[i]}, '
                f'`stride` is {stride[i]}, and `dilation` is {dilation[i]}.'
            )

    return output_padding


# import these types for type annotation
from .backend.core import Module
from .backend.init import DataDependentInitializer
from .stochastic import StochasticTensor
from .typing_ import *
