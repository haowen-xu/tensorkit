from typing import *

from .core import jit

__all__ = [
    'split_channel_spatial_shape', 'unsplit_channel_spatial_shape',
    'calculate_deconv_output_padding',
    'calculate_conv_output_size', 'calculate_deconv_output_size',
]


@jit
def _check_conv_args(input_size: List[int],
                     padding: List[Tuple[int, int]],
                     arg_values: List[List[int]],
                     arg_names: List[str]) -> int:
    spatial_ndims = len(input_size)
    if spatial_ndims not in (1, 2, 3):
        raise ValueError(
            '`input_size` is not a 1d, 2d or 3d convolutional input size: '
            'got input size {}.'.format(input_size)
        )

    if len(padding) != spatial_ndims:
        raise ValueError(
            '`padding` is not for {}d convolution: got `padding` {}.'.
            format(spatial_ndims, padding)
        )

    for i in range(len(arg_values)):
        arg_val = arg_values[i]
        if len(arg_val) != spatial_ndims:
            arg_name = arg_names[i]
            raise ValueError(
                '`{}` is not for {}d convolution: got `{}` {}.'.
                format(arg_name, spatial_ndims, arg_name, arg_val)
            )

    return spatial_ndims


@jit
def split_channel_spatial_shape(shape: List[int]) -> Tuple[int, List[int]]:
    if len(shape) not in (2, 3, 4):
        raise ValueError('Invalid `shape`: {}'.format(shape))
    return shape[0], shape[1:]


@jit
def unsplit_channel_spatial_shape(channels: int, size: List[int]) -> List[int]:
    if len(size) not in (1, 2, 3):
        raise ValueError('Invalid `size`: {}'.format(size))
    return [channels] + size


@jit
def calculate_deconv_output_padding(input_size: List[int],
                                    output_size: List[int],
                                    kernel_size: List[int],
                                    stride: List[int],
                                    padding: List[Tuple[int, int]],
                                    dilation: List[int]):
    """
    Calculate the `output_padding` for deconvolution (conv_transpose).

    Args:
        input_size: The input size (shape) of the spatial dimensions.
        output_size: The output size (shape) of the spatial dimensions.
        kernel_size: The kernel size.
        stride: The stride.
        padding: The padding.
        dilation: The dilation.

    Returns:
        The output padding, can be used to construct a deconvolution
        (conv transpose) layer.

    Raises:
        ValueError: If any argument is invalid, or no output padding
            can satisfy the specified arguments.
    """
    spatial_ndims = _check_conv_args(
        input_size, padding,
        [output_size, kernel_size, stride, dilation],
        ['output_size', 'kernel_size', 'stride', 'dilation'],
    )

    ret: List[int] = []
    for i in range(spatial_ndims):
        op = output_size[i] - (
            (input_size[i] - 1) * stride[i] -
            (padding[i][0] + padding[i][1]) +
            (kernel_size[i] - 1) * dilation[i] + 1
        )
        if op < 0 or (op >= stride[i] and op >= dilation[i]):
            raise ValueError(
                'No `output_padding` can satisfy the deconvolution task: '
                'input_size == {}, output_size == {}, '
                'kernel_size == {}, stride == {}, '
                'padding == {}, dilation == {}.'.format(
                    input_size, output_size, kernel_size, stride, padding,
                    dilation
                )
            )
        ret.append(op)

    return ret


@jit
def calculate_conv_output_size(input_size: List[int],
                               kernel_size: List[int],
                               stride: List[int],
                               padding: List[Tuple[int, int]],
                               dilation: List[int]) -> List[int]:
    """
    Calculate the convolution output size for specified arguments.

    Args:
        input_size: The input size (shape) of the spatial dimensions.
        kernel_size: The kernel size.
        stride: The stride.
        padding: The padding.
        dilation: The dilation.

    Returns:
        The output size.
    """
    spatial_ndims = _check_conv_args(
        input_size, padding,
        [input_size, kernel_size, stride, dilation],
        ['input_size', 'kernel_size', 'stride', 'dilation'],
    )

    ret: List[int] = []
    for i in range(spatial_ndims):
        ret.append(
            1 + (input_size[i] + padding[i][0] + padding[i][1] -
                 (kernel_size[i] - 1) * dilation[i] - 1) // stride[i]
        )

    return ret


@jit
def calculate_deconv_output_size(input_size: List[int],
                                 kernel_size: List[int],
                                 stride: List[int],
                                 padding: List[Tuple[int, int]],
                                 output_padding: List[int],
                                 dilation: List[int]) -> List[int]:
    """
    Calculate the deconvolution output size for specified arguments.

    Args:
        input_size: The input size (shape) of the spatial dimensions.
        kernel_size: The kernel size.
        stride: The stride.
        padding: The padding.
        output_padding: The output padding.
        dilation: The dilation.

    Returns:
        The output size.
    """
    spatial_ndims = _check_conv_args(
        input_size, padding,
        [input_size, kernel_size, stride, output_padding, dilation],
        ['input_size', 'kernel_size', 'stride', 'output_padding', 'dilation'],
    )

    ret: List[int] = []
    for i in range(spatial_ndims):
        ret.append(
            output_padding[i] +
            (input_size[i] - 1) * stride[i] -
            (padding[i][0] + padding[i][1]) +
            (kernel_size[i] - 1) * dilation[i] +
            1
        )

    return ret
