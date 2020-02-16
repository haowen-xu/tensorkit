import unittest
from itertools import product

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.ops import *


class UtilsTestCase(unittest.TestCase):

    def test_split_channel_spatial_shape(self):
        for spatial_ndims in (1, 2, 3):
            conv_shape = make_conv_shape([], 6, [7, 8, 9][:spatial_ndims])
            self.assertEqual(
                T.utils.split_channel_spatial_shape(conv_shape),
                (6, [7, 8, 9][:spatial_ndims])
            )
        with pytest.raises(Exception, match='Invalid `shape`'):
            _ = T.utils.split_channel_spatial_shape([])

    def test_unsplit_channel_spatial_shape(self):
        for spatial_ndims in (1, 2, 3):
            conv_shape = make_conv_shape([], 6, [7, 8, 9][:spatial_ndims])
            self.assertEqual(
                T.utils.unsplit_channel_spatial_shape(6, [7, 8, 9][:spatial_ndims]),
                conv_shape
            )
        with pytest.raises(Exception, match='Invalid `size`'):
            _ = T.utils.unsplit_channel_spatial_shape(1, [])

    def test_conv_deconv_output_shape_and_args(self):
        for input_size, kernel_size, stride, padding, dilation in product(
                    ([8, 9, 10], [16, 21, 32], [30, 31, 32]),
                    ([1] * 3, [2] * 3, [3] * 3, [1, 2, 3]),
                    ([1] * 3, [2] * 3, [3] * 3, [1, 2, 3]),
                    ([(0, 0)] * 3, [(1, 1)] * 3, [(2, 2)] * 3, [(3, 3)] * 3,
                     [(1, 2), (2, 3), (3, 4)]),
                    ([1] * 3, [2] * 3, [3] * 3, [1, 2, 3]),
                ):
            args = (input_size, kernel_size, stride, padding, dilation)

            # calculate_conv_output_size
            output_size = [get_conv_output_size(*a) for a in zip(*args)]
            self.assertEqual(
                T.utils.calculate_conv_output_size(
                    input_size=input_size, kernel_size=kernel_size,
                    stride=stride, padding=padding, dilation=dilation,
                ),
                output_size
            )
            layer1 = tk.layers.LinearConv3d(
                1, 1, kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation,
            )
            x = T.zeros(make_conv_shape([1], 1, input_size))
            y = layer1(x)
            self.assertEqual(
                T.utils.split_channel_spatial_shape(T.shape(y)[1:])[1],
                output_size,
            )

            # calculate_deconv_output_padding
            output_padding = T.utils.calculate_deconv_output_padding(
                input_size=output_size, output_size=input_size,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation,
            )
            layer2 = tk.layers.LinearConvTranspose3d(
                1, 1, kernel_size=kernel_size, stride=stride, padding=padding,
                output_padding=output_padding, dilation=dilation,
            )
            z = layer2(y)
            self.assertEqual(
                T.utils.split_channel_spatial_shape(T.shape(z)[1:])[1],
                input_size,
            )

            # calculate_deconv_output_size
            self.assertEqual(
                T.utils.calculate_deconv_output_size(
                    input_size=output_size, kernel_size=kernel_size,
                    stride=stride, padding=padding,
                    output_padding=output_padding, dilation=dilation,
                ),
                input_size
            )

        # test error
        kwargs = dict(kernel_size=[1], stride=[1], dilation=[1], padding=[(0, 0)])
        for input_size in ([], [1, 2, 3, 4]):
            with pytest.raises(Exception,
                               match='`input_size` is not a 1d, 2d or 3d '
                                     'convolutional input size'):
                _ = T.utils.calculate_conv_output_size(input_size, **kwargs)
            with pytest.raises(Exception,
                               match='`input_size` is not a 1d, 2d or 3d '
                                     'convolutional input size'):
                _ = T.utils.calculate_deconv_output_size(input_size, output_padding=[0], **kwargs)

        for arg_name in ('kernel_size', 'stride', 'dilation', 'padding'):
            kwargs2 = dict(kwargs)
            if arg_name == 'padding':
                kwargs2[arg_name] = [(0, 0)] * 2
            else:
                kwargs2[arg_name] = [1, 1]
            with pytest.raises(Exception, match='`.*` is not for .*d convolution'):
                _ = T.utils.calculate_conv_output_size([11], **kwargs2)
            with pytest.raises(Exception, match='`.*` is not for .*d convolution'):
                _ = T.utils.calculate_deconv_output_size([11], output_padding=[0], **kwargs2)

        with pytest.raises(Exception, match='`.*` is not for .*d convolution'):
            _ = T.utils.calculate_deconv_output_size([11], output_padding=[0, 0], **kwargs)

        with pytest.raises(Exception,
                           match='No `output_padding` can satisfy the '
                                 'deconvolution task'):
            _ = T.utils.calculate_deconv_output_padding([2], [1], [1], [1], [(0, 0)], [1])
