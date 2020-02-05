import unittest

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.arg_check import *


class ArgCheckTestCase(unittest.TestCase):

    def test_validate_positive_int(self):
        for v in [1, 2, 3]:
            self.assertEqual(validate_positive_int('v', v), v)

        with pytest.raises(ValueError,
                           match='`v` must be a positive int: '
                                 'got -1'):
            _ = validate_positive_int('v', -1)

    def test_validate_layer(self):
        layer = tk.layers.Linear(5, 3)
        for v in [layer, T.jit_compile(layer)]:
            self.assertIs(validate_layer('v', v), v)

        with pytest.raises(TypeError,
                           match='`v` is required to be a layer: got 123'):
            _ = validate_layer('v', 123)

    def test_validate_layer_factory(self):
        for v in [tk.layers.Linear, (lambda: tk.layers.Linear(5, 3))]:
            self.assertIs(validate_layer_factory('v', v), v)

        with pytest.raises(TypeError,
                           match='`v` is required to be a layer factory: '
                                 'got 123'):
            _ = validate_layer_factory('v', 123)

    def test_get_layer_from_layer_or_factory(self):
        factory = lambda in_features, out_features: \
            tk.layers.Linear(in_features, out_features)
        layer = factory(5, 3)
        for v in [layer, T.jit_compile(layer),
                  tk.layers.Linear, factory]:
            out = get_layer_from_layer_or_factory(
                'v', v, args=(5,), kwargs=dict(out_features=3))
            if isinstance(v, T.Module):
                self.assertIs(out, v)
            else:
                self.assertIsInstance(out, tk.layers.Linear)
                self.assertEqual(out.in_features, 5)
                self.assertEqual(out.out_features, 3)

        with pytest.raises(TypeError,
                           match='`v` is required to be a layer or a layer '
                                 'factory: got 123'):
            _ = get_layer_from_layer_or_factory('v', 123)

    def test_validate_conv_size(self):
        for spatial_ndims in (1, 2, 3):
            self.assertEqual(
                validate_conv_size('v', 2, spatial_ndims),
                [2] * spatial_ndims
            )
            self.assertEqual(
                validate_conv_size('v', [1, 2, 3][:spatial_ndims], spatial_ndims),
                [1, 2, 3][:spatial_ndims]
            )
            self.assertEqual(
                validate_conv_size('v', (1, 2, 3)[:spatial_ndims], spatial_ndims),
                [1, 2, 3][:spatial_ndims]
            )

        with pytest.raises(ValueError,
                           match=r'`v` must be either a positive integer, or '
                                 r'a sequence of positive integers with length '
                                 r'`3`: got \[1, 2\]'):
            _ = validate_conv_size('v', [1, 2], 3),

        with pytest.raises(ValueError,
                           match=r'`v` must be either a positive integer, or '
                                 r'a sequence of positive integers with length '
                                 r'`3`: got \[1, 2, 0\]'):
            _ = validate_conv_size('v', [1, 2, 0], 3)

    def test_validate_padding(self):
        for spatial_ndims in (1, 2, 3):
            self.assertEqual(
                validate_padding(
                    'none',
                    kernel_size=[5, 6, 7][:spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [0] * spatial_ndims
            )
            self.assertEqual(
                validate_padding(
                    'full',
                    kernel_size=[5, 6, 7][:spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [4, 10, 18][:spatial_ndims]
            )
            self.assertEqual(
                validate_padding(
                    'half',
                    kernel_size=[5, 6, 7][:spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [2, 5, 9][:spatial_ndims]
            )
            self.assertEqual(
                validate_padding(
                    4,
                    kernel_size=[5, 6, 7][:spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [4] * spatial_ndims
            )
            self.assertEqual(
                validate_padding(
                    [3, 4, 5][:spatial_ndims],
                    kernel_size=[5, 6, 7][:spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [3, 4, 5][:spatial_ndims]
            )
            self.assertEqual(
                validate_padding(
                    (3, 4, 5)[:spatial_ndims],
                    kernel_size=[5, 6, 7][:spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [3, 4, 5][:spatial_ndims]
            )

        with pytest.raises(ValueError,
                           match=r'`\(kernel_size - 1\) \* dilation` is '
                                 r'required to be even for `padding` == "half":'
                                 r' got `kernel_size` \[4, 5, 6\], and '
                                 r'`dilation` \[1, 2, 3\]'):
            _ = validate_padding('half', [4, 5, 6], [1, 2, 3], 3)

        msg_prefix = (
            r'`padding` must be a positive integer, a '
            r'sequence of positive integers of length '
            r'`3`, "none", "half" or "full": got '
        )

        with pytest.raises(ValueError, match=msg_prefix + r'0'):
            _ = validate_padding(0, [1] * 3, [1] * 3, 3)

        with pytest.raises(ValueError, match=msg_prefix + r'\[1, 2\]'):
            _ = validate_padding([1, 2], [1] * 3, [1] * 3, 3)

        with pytest.raises(ValueError, match=msg_prefix + r'\[1, 2, -1\]'):
            _ = validate_padding([1, 2, -1], [1] * 3, [1] * 3, 3)

    def test_validate_output_padding(self):
        for spatial_ndims in (1, 2, 3):
            self.assertEqual(
                validate_output_padding(
                    0,
                    stride=[1, 2, 3][: spatial_ndims],
                    dilation=[1, 2, 3][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [0] * spatial_ndims
            )
            self.assertEqual(
                validate_output_padding(
                    [1, 2, 3][:spatial_ndims],
                    stride=[4, 5, 6][: spatial_ndims],
                    dilation=[3, 4, 5][:spatial_ndims],
                    spatial_ndims=spatial_ndims,
                ),
                [1, 2, 3][:spatial_ndims],
            )

        err_msg = (
            r'`output_padding` must be a non-negative integer, or a sequence '
            r'of non-negative integers, and must be smaller than either '
            r'`stride` or `dilation`'
        )

        with pytest.raises(ValueError, match=err_msg):
            _ = validate_output_padding(-1, [4] * 3, [4] * 3, 3)

        with pytest.raises(ValueError, match=err_msg):
            _ = validate_output_padding([1, 2], [4] * 3, [4] * 3, 3)

        with pytest.raises(ValueError, match=err_msg):
            _ = validate_output_padding([1, 2, -1], [4] * 3, [4] * 3, 3)
