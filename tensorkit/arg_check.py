from typing import *

__all__ = [
    # general argument validators
    'validate_positive_int',

    # layer argument validators
    'validate_layer', 'validate_layer_factory', 'get_layer_from_layer_or_factory',
    'validate_conv_size', 'validate_padding', 'validate_output_padding',
]


# general argument validators
def validate_positive_int(arg_name: str, arg_value) -> int:
    ret = int(arg_value)
    if ret <= 0:
        raise ValueError(f'`{arg_name}` must be a positive int: '
                         f'got {arg_value!r}')
    return ret


# layer argument validators
def validate_layer(arg_name: str, layer) -> 'Module':
    from tensorkit.tensor import is_jit_layer
    if isinstance(layer, Module) or is_jit_layer(layer):
        return layer
    else:
        raise TypeError(f'`{arg_name}` is required to be a layer: got {layer!r}')


def validate_layer_factory(arg_name: str, layer_factory):
    if isinstance(layer_factory, type) or callable(layer_factory):
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
    else:
        if hasattr(padding, '__iter__'):
            ret = list(map(int, padding))
        else:
            ret = [int(padding)] * spatial_ndims

        if len(ret) != spatial_ndims or not all(p > 0 for p in ret):
            raise ValueError(
                f'`padding` must be a positive integer, a sequence of '
                f'positive integers of length `{spatial_ndims}`, "none", '
                f'"half" or "full": got {padding}.'
            )
        return ret


def validate_output_padding(output_padding: Union[int, Sequence[int]],
                            stride: List[int],
                            dilation: List[int],
                            spatial_ndims: int
                            ) -> List[int]:
    if hasattr(output_padding, '__iter__'):
        ret = list(map(int, output_padding))
    else:
        ret = [int(output_padding)] * spatial_ndims

    value_ok = (len(ret) == spatial_ndims)
    if value_ok:
        for i in range(spatial_ndims):
            if ret[i] < 0 or ret[i] >= max(stride[i], dilation[i]):
                value_ok = False
                break

    if not value_ok:
        raise ValueError(
            f'`output_padding` must be a non-negative integer, or a sequence '
            f'of non-negative integers, and must be smaller than either '
            f'`stride` or `dilation`: got `output_padding` {output_padding}, '
            f'`stride` {stride}, and `dilation` {dilation}.'
        )

    return ret


# import these types for type annotation
from .backend.core import Module
from .typing_ import *
