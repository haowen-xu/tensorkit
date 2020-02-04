import unittest

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *
from tests.ops import make_conv_shape


class FlattenToNDimsTestCase(unittest.TestCase):

    def test_FlattenToNDims(self):
        x = T.random.randn(make_conv_shape([3, 4], 6, [5]))

        internal = tk.layers.LinearConv1d(6, 7, kernel_size=1)
        layer = T.jit_compile(tk.layers.FlattenToNDims(internal, 3))

        xx, front_shape = T.flatten_to_ndims(x, 3)
        assert_equal(layer(x), T.unflatten_from_ndims(internal(xx), front_shape))

        with pytest.raises(Exception, match=r'`rank\(input\)` is too low'):
            _ = layer(T.random.randn([1, 1]))


class ConstantPadTestCase(unittest.TestCase):

    def test_ConstantPad(self):
        for value_arg in [{}, {'value': 123.0}]:
            value = value_arg.get('value', 0.0)
            layer = tk.layers.ConstantPad([1, (2, 3), [0, 5]], **value_arg)
            self.assertEqual(
                repr(layer),
                f'ConstantPad(padding=[(1, 1), (2, 3), (0, 5)], value={value})'
            )
            layer = T.jit_compile(layer)

            x = T.random.randn([3, 4, 5])
            assert_equal(layer(x), T.pad(x, [(1, 1), (2, 3), (0, 5)], value=value))

            x = T.random.randn([2, 3, 4, 5])
            assert_equal(layer(x), T.pad(x, [(1, 1), (2, 3), (0, 5)], value=value))

        with pytest.raises(ValueError,
                           match=r'`padding` must be a sequence of int or '
                                 r'tuple of \(int, int\)'):
            _ = tk.layers.ConstantPad([(1, 2, 3)])

    def test_ConstantPadNd(self):
        for value_arg in [{}, {'value': 123.0}]:
            value = value_arg.get('value', 0.0)

            for spatial_ndims in (1, 2, 3):
                layer_factory = getattr(tk.layers, f'ConstantPad{spatial_ndims}d')
                x = T.random.randn(make_conv_shape([3], 4, [5, 6, 7][:spatial_ndims]))

                def spatial_pad(input, spatial_padding):
                    def fn(v):
                        if isinstance(v, int):
                            v = (v, v)
                        return tuple(v)
                    assert(len(spatial_padding) == spatial_ndims)
                    spatial_padding = list(map(fn, spatial_padding))
                    if T.IS_CHANNEL_LAST:
                        spatial_padding = list(spatial_padding) + [(0, 0)]
                    else:
                        spatial_padding = list(spatial_padding)
                    return T.pad(input, spatial_padding, value)

                # single padding argument
                for pad_arg in [0, 1, (1, 2), (0, 1)]:
                    padding = [(pad_arg, pad_arg) if isinstance(pad_arg, int)
                               else pad_arg] * spatial_ndims
                    layer = layer_factory(pad_arg, **value_arg)
                    self.assertEqual(
                        repr(layer),
                        f'ConstantPad{spatial_ndims}d(padding={padding}, value={value})'
                    )
                    layer = T.jit_compile(layer)
                    assert_equal(
                        layer(x),
                        spatial_pad(x, [pad_arg] * spatial_ndims)
                    )

                # multiple padding argument
                pad_arg = [(0, 2), 3, 1][: spatial_ndims]
                padding = [(0, 2), (3, 3), (1, 1)][: spatial_ndims]
                layer = layer_factory(*pad_arg, **value_arg)
                self.assertEqual(
                    repr(layer),
                    f'ConstantPad{spatial_ndims}d(padding={padding}, value={value})'
                )
                layer = T.jit_compile(layer)
                assert_equal(layer(x), spatial_pad(x, padding))

                # error padding argument
                if spatial_ndims == 1:
                    with pytest.raises(ValueError,
                                       match=f'`ConstantPad{spatial_ndims}d` requires '
                                             f'1 spatial padding'):
                        _ = layer_factory(0, 1, 2, 3)
                else:
                    with pytest.raises(ValueError,
                                       match=f'`ConstantPad{spatial_ndims}d` requires '
                                             f'1 or {spatial_ndims} spatial paddings'):
                        _ = layer_factory(0, 1, 2, 3)


class ChannelSwapTestCase(unittest.TestCase):

    def test_channel_last_to_first(self):
        for spatial_ndims in (1, 2, 3):
            layer_factory = getattr(
                tk.layers, f'ChannelLastToFirst{spatial_ndims}d')
            layer = layer_factory()
            self.assertEqual(repr(layer), f'ChannelLastToFirst{spatial_ndims}d()')

            fn = getattr(T.nn, f'channel_last_to_first{spatial_ndims}d')
            x = T.random.randn([3, 4, 5, 6, 7][:spatial_ndims + 2])

            layer = T.jit_compile(layer)
            assert_equal(layer(x), fn(x))

    def test_channel_first_to_last(self):
        for spatial_ndims in (1, 2, 3):
            layer_factory = getattr(
                tk.layers, f'ChannelFirstToLast{spatial_ndims}d')
            layer = layer_factory()
            self.assertEqual(repr(layer), f'ChannelFirstToLast{spatial_ndims}d()')

            fn = getattr(T.nn, f'channel_first_to_last{spatial_ndims}d')
            x = T.random.randn([3, 4, 5, 6, 7][:spatial_ndims + 2])

            layer = T.jit_compile(layer)
            assert_equal(layer(x), fn(x))
