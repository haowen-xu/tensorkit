import unittest
from itertools import product

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.arg_check import *
from tests.helper import *
from tests.ops import *


class UtilsTestCase(TestCase):

    def test_flatten_nested_layers(self):
        layers = [tk.layers.Linear(5, 5) for _ in range(5)]
        layers2 = tk.layers.flatten_nested_layers([
            layers[0], layers[1:2], [layers[2], [layers[3], layers[4]]]
        ])
        self.assertListEqual(layers2, layers)

        with pytest.raises(TypeError,
                           match='`nested_layers` is not a nested list '
                                 'of layers.'):
            _ = tk.layers.flatten_nested_layers([1])

        with pytest.raises(TypeError,
                           match='`nested_layers` is not a nested list '
                                 'of layers.'):
            _ = tk.layers.flatten_nested_layers({'a': layers[0]})

        with pytest.raises(TypeError,
                           match='`nested_layers` is not a nested list '
                                 'of layers.'):
            _ = tk.layers.flatten_nested_layers('')

    def test_get_activation_class(self):
        x = T.random.randn([2, 3, 4])

        for origin_name, factory, args, kwargs, expected in [
                    ('Linear', None, None, None, None),
                    ('ReLU', tk.layers.ReLU, (), {}, T.nn.relu(x)),
                    ('Leaky_ReLU', tk.layers.LeakyReLU, (), {}, T.nn.leaky_relu(x)),
                    ('Leaky_ReLU', tk.layers.LeakyReLU, (0.2,), {}, T.nn.leaky_relu(x, 0.2)),
                    ('Leaky_ReLU', tk.layers.LeakyReLU, (), {'negative_slope': 0.2}, T.nn.leaky_relu(x, 0.2)),
                    ('Sigmoid', tk.layers.Sigmoid, (), {}, T.nn.sigmoid(x)),
                    ('Tanh', tk.layers.Tanh, (), {}, T.tanh(x)),
                    ('HardTanh', tk.layers.HardTanh, (), {}, T.clip(x, -1., 1.)),
                    ('HardTanh', tk.layers.HardTanh, (-2., 3.), {}, T.clip(x, -2., 3.)),
                    ('HardTanh', tk.layers.HardTanh, (), {'min_val': -2., 'max_val': 3.}, T.clip(x, -2., 3.)),
                    ('Log_Softmax', tk.layers.LogSoftmax, (), {}, T.nn.log_softmax(x)),
                ]:
            name_candidates = (None,) if origin_name is None else (
                origin_name,
                origin_name.lower(),
                origin_name.replace('_', ''),
                origin_name.replace('_', '').lower()
            )
            for name in name_candidates:
                err_msg = f'{name}, {factory}, {args}, {kwargs}, {expected}'
                self.assertIs(tk.layers.get_activation_class(name), factory)
                if factory is not None:
                    assert_allclose(factory(*args, **kwargs)(x), expected, err_msg=err_msg)

        # unsupported activation
        with pytest.raises(ValueError, match='Unsupported activation: invalid'):
            _ = tk.layers.get_activation_class('invalid')

    def test_deconv_output_padding(self):
        def f(input_size, output_size, kernel_size, stride, padding, dilation):
            output_padding = tk.layers.get_deconv_output_padding(
                input_size=input_size, output_size=output_size,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation,
            )
            spatial_ndims = len(input_size)
            kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
            stride = validate_conv_size('stride', stride, spatial_ndims)
            dilation = validate_conv_size('dilation', dilation, spatial_ndims)
            padding = validate_padding(padding, kernel_size, dilation, spatial_ndims)

            layer_cls = getattr(tk.layers, f'LinearConvTranspose{spatial_ndims}d')
            layer = layer_cls(
                in_channels=1, out_channels=1, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
                output_padding=output_padding,
            )
            x = T.random.randn(make_conv_shape([1], 1, input_size))
            y = layer(x)
            y_shape = T.shape(y)
            true_output_size = [y_shape[a] for a in get_spatial_axis(spatial_ndims)]

            self.assertEqual(true_output_size, output_size)

        def g(output_size, kernel_size, stride, padding, dilation):
            # use conv to generate the `input_size`
            spatial_ndims = len(output_size)
            layer_cls = getattr(tk.layers, f'LinearConv{spatial_ndims}d')
            layer = layer_cls(
                in_channels=1, out_channels=1, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation,
            )
            x = T.random.randn(make_conv_shape([1], 1, output_size))
            y = layer(x)
            y_shape = T.shape(y)
            input_size = [y_shape[a] for a in get_spatial_axis(spatial_ndims)]

            # do check
            f(input_size, output_size, kernel_size, stride, padding, dilation)

        g([1], 1, 1, 0, 1)
        g([2], 2, 1, 0, 1)

        for output_size, kernel_size, stride, padding, dilation in product(
                    ([8, 9, 10], [16, 21, 32], [30, 31, 32]),
                    (1, 2, 3, [1, 2, 3]),
                    (1, 2, 3, [1, 2, 3]),
                    ('none', 'half', 'full', 0, 1, [1, 2, 3]),
                    (1, 2, [1, 2, 3]),
                ):
            try:
                _ = validate_padding(
                    padding,
                    validate_conv_size('kernel_size', kernel_size, 3),
                    validate_conv_size('dilation', dilation, 3),
                    3,
                )
            except Exception:
                continue
            else:
                g(output_size, kernel_size, stride, padding, dilation)

        # test error
        with pytest.raises(ValueError,
                           match='The length of `input_size` != the length of '
                                 '`output_size`'):
            _ = tk.layers.get_deconv_output_padding([1], [2, 3])

        with pytest.raises(ValueError,
                           match='Only 1d, 2d, or 3d `input_size` and '
                                 '`output_size` is supported'):
            _ = tk.layers.get_deconv_output_padding([], [])

        with pytest.raises(ValueError,
                           match='Only 1d, 2d, or 3d `input_size` and '
                                 '`output_size` is supported'):
            _ = tk.layers.get_deconv_output_padding([1, 1, 1, 1], [1, 1, 1, 1])

        with pytest.raises(ValueError,
                           match='No `output_padding` can satisfy the '
                                 'deconvolution task'):
            _ = tk.layers.get_deconv_output_padding([2], [1])
