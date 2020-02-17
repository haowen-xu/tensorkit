import unittest
from itertools import product

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *
from tests.ops import *


def check_resblock(ctx,
                   spatial_ndims,
                   resblock_cls,
                   linear_cls,
                   normalizer_cls,
                   dropout_cls,
                   output_padding_arg=None):
    output_padding_arg = output_padding_arg or {}
    x = T.random.randn(make_conv_shape([3], 5, [26, 25, 24][: spatial_ndims]))

    # the simplest form: no shortcut, no activation, no normalizer, no dropout
    # with all other arguments keep to default
    layer = resblock_cls(in_channels=5, out_channels=5, kernel_size=1)
    ctx.assertIsInstance(layer.shortcut, tk.layers.Identity)
    ctx.assertIsInstance(layer.pre_conv0, tk.layers.Identity)
    ctx.assertIsInstance(layer.merge_context0, tk.layers.IgnoreContext)
    ctx.assertIsInstance(layer.conv0, linear_cls)
    ctx.assertIsInstance(layer.conv0.weight_store, tk.layers.SimpleParamStore)
    ctx.assertIsNotNone(layer.conv0.bias_store)
    ctx.assertEqual(layer.conv0.kernel_size, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv0.stride, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv0.padding, [(0, 0)] * spatial_ndims)
    ctx.assertEqual(layer.conv0.dilation, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv0.out_channels, 5)
    ctx.assertIsInstance(layer.pre_conv1, tk.layers.Identity)
    ctx.assertIsInstance(layer.merge_context1, tk.layers.IgnoreContext)
    ctx.assertIsInstance(layer.conv1, linear_cls)
    ctx.assertIsInstance(layer.conv1.weight_store, tk.layers.SimpleParamStore)
    ctx.assertIsNotNone(layer.conv1.bias_store)
    ctx.assertEqual(layer.conv1.kernel_size, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv1.stride, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv1.padding, [(0, 0)] * spatial_ndims)
    ctx.assertEqual(layer.conv1.dilation, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv1.out_channels, 5)
    ctx.assertIsInstance(layer.post_conv1, tk.layers.Identity)

    # force `use_bias` = False
    layer = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                         use_bias=False)
    ctx.assertFalse(layer.conv0.use_bias)
    ctx.assertFalse(layer.conv1.use_bias)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x),
        x + layer.conv1(layer.conv0(x)),
        rtol=1e-4, atol=1e-6,
    )

    # force using shortcut even if not necessary
    layer = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                         use_shortcut=True)
    ctx.assertIsInstance(layer.shortcut, linear_cls)
    ctx.assertIsInstance(layer.shortcut.weight_store, tk.layers.SimpleParamStore)
    ctx.assertFalse(layer.shortcut.use_bias)
    ctx.assertEqual(layer.shortcut.kernel_size, [1] * spatial_ndims)
    ctx.assertEqual(layer.shortcut.stride, [1] * spatial_ndims)
    ctx.assertEqual(layer.shortcut.padding, [(0, 0)] * spatial_ndims)
    ctx.assertEqual(layer.shortcut.dilation, [1] * spatial_ndims)
    ctx.assertIsNotNone(layer.conv0.bias_store)
    ctx.assertIsNotNone(layer.conv1.bias_store)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x),
        layer.shortcut(x) + layer.conv1(layer.conv0(x)),
        rtol=1e-4, atol=1e-6,
    )

    # force `use_bias = True` for `use_shortcut=True`
    layer = resblock_cls(
        in_channels=5, out_channels=5, kernel_size=1,
        use_shortcut=True, use_bias=True)
    ctx.assertIsNotNone(layer.shortcut.bias_store)
    ctx.assertIsNotNone(layer.conv0.bias_store)
    ctx.assertIsNotNone(layer.conv1.bias_store)

    # test conv parameters & resize_at_exit = False
    kernel_size = [5, 3, 2][: spatial_ndims]
    stride = [3, 2, 1][: spatial_ndims]
    padding = [(2, 3), (3, 4), (4, 5)][: spatial_ndims]
    dilation = [1, 3, 4][: spatial_ndims]
    half_padding = [
        ((k - 1) * d // 2, (k - 1) * d - (k - 1) * d // 2)
        for k, d in zip(kernel_size, dilation)
    ]
    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        **output_padding_arg
    )
    ctx.assertIsInstance(layer.shortcut, linear_cls)
    ctx.assertFalse(layer.shortcut.use_bias)
    ctx.assertEqual(layer.shortcut.kernel_size, kernel_size)
    ctx.assertEqual(layer.shortcut.stride, stride)
    ctx.assertEqual(layer.shortcut.padding, padding)
    ctx.assertEqual(layer.shortcut.dilation, dilation)
    ctx.assertEqual(layer.shortcut.out_channels, 4)
    ctx.assertIsInstance(layer.conv0, linear_cls)
    ctx.assertIsNotNone(layer.conv0.bias_store)
    ctx.assertEqual(layer.conv0.kernel_size, kernel_size)
    ctx.assertEqual(layer.conv0.stride, stride)
    ctx.assertEqual(layer.conv0.padding, padding)
    ctx.assertEqual(layer.conv0.dilation, dilation)
    ctx.assertEqual(layer.conv0.out_channels, 4)
    ctx.assertIsInstance(layer.conv1, linear_cls)
    ctx.assertIsNotNone(layer.conv1.bias_store)
    ctx.assertEqual(layer.conv1.kernel_size, kernel_size)
    ctx.assertEqual(layer.conv1.stride, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv1.padding, half_padding)
    ctx.assertEqual(layer.conv1.dilation, dilation)
    ctx.assertEqual(layer.conv1.out_channels, 4)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(layer(x), layer.shortcut(x) + layer.conv1(layer.conv0(x)))

    # test resize_at_exit = True
    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        resize_at_exit=True, **output_padding_arg
    )
    ctx.assertEqual(layer.shortcut.stride, stride)
    ctx.assertEqual(layer.shortcut.padding, padding)
    ctx.assertEqual(layer.shortcut.out_channels, 4)
    ctx.assertEqual(layer.conv0.stride, [1] * spatial_ndims)
    ctx.assertEqual(layer.conv0.padding, half_padding)
    ctx.assertEqual(layer.conv0.out_channels, 5)
    ctx.assertEqual(layer.conv1.stride, stride)
    ctx.assertEqual(layer.conv1.padding, padding)
    ctx.assertEqual(layer.conv1.out_channels, 4)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x),
        layer.shortcut(x) + layer.conv1(layer.conv0(x)),
        rtol=1e-4, atol=1e-6,
    )

    # test normalizer and activation
    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        normalizer=normalizer_cls, activation=tk.layers.LeakyReLU,
        **output_padding_arg,
    )
    tk.layers.set_train_mode(layer, True)
    _ = layer(x)  # initialize the normalizers
    tk.layers.set_train_mode(layer, False)
    ctx.assertFalse(layer.conv0.use_bias)
    ctx.assertIsInstance(layer.pre_conv0, tk.layers.Sequential)
    ctx.assertIsInstance(layer.pre_conv0[0], normalizer_cls)
    ctx.assertIsInstance(layer.pre_conv0[1], tk.layers.LeakyReLU)
    ctx.assertEqual(len(layer.pre_conv0), 2)
    ctx.assertIsInstance(layer.pre_conv1, tk.layers.Sequential)
    ctx.assertIsInstance(layer.pre_conv1[0], normalizer_cls)
    ctx.assertIsInstance(layer.pre_conv1[1], tk.layers.LeakyReLU)
    ctx.assertEqual(len(layer.pre_conv1), 2)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x),
        (layer.shortcut(x) +
         layer.conv1(layer.pre_conv1(layer.conv0(layer.pre_conv0(x))))),
        rtol=1e-4, atol=1e-6,
    )

    # test dropout
    for dropout in [0.3, dropout_cls, dropout_cls(0.3)]:
        layer = resblock_cls(
            in_channels=5, out_channels=4, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            normalizer=normalizer_cls, dropout=dropout,
            **output_padding_arg,
        )
        ctx.assertIsNotNone(layer.conv0.bias_store)
        ctx.assertIsInstance(layer.pre_conv0, normalizer_cls)
        ctx.assertIsInstance(layer.pre_conv1, tk.layers.Sequential)
        if isinstance(dropout, float):
            ctx.assertIsInstance(layer.pre_conv1[0], tk.layers.Dropout)
            ctx.assertEqual(layer.pre_conv1[0].p, dropout)
        elif isinstance(dropout, type):
            ctx.assertIsInstance(layer.pre_conv1[0], dropout_cls)
        else:
            ctx.assertIs(layer.pre_conv1[0], dropout)
        ctx.assertIsInstance(layer.pre_conv1[1], normalizer_cls)
        ctx.assertEqual(len(layer.pre_conv1), 2)
        tk.layers.set_train_mode(layer, True)
        _ = layer(x)
        tk.layers.set_train_mode(layer, False)

        layer = tk.layers.jit_compile(layer)
        assert_allclose(
            layer(x),
            (layer.shortcut(x) +
             layer.conv1(layer.pre_conv1(layer.conv0(layer.pre_conv0(x))))),
            rtol=1e-4, atol=1e-6,
        )

    # test gated
    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        gated=True, gate_bias=1.5, **output_padding_arg,
    )
    ctx.assertIsNotNone(layer.shortcut.bias_store)
    ctx.assertIsNotNone(layer.conv1.bias_store)
    ctx.assertEqual(layer.conv1.out_channels, 8)
    ctx.assertIsInstance(layer.post_conv1, tk.layers.Gated)
    ctx.assertEqual(layer.post_conv1.gate_bias, 1.5)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x),
        (layer.shortcut(x) + layer.post_conv1(
            layer.conv1(layer.pre_conv1(layer.conv0(layer.pre_conv0(x)))))),
        rtol=1e-4, atol=1e-6,
    )

    # test context
    merge_context0 = tk.layers.AddContext()
    merge_context1 = tk.layers.MultiplyContext()
    layer = resblock_cls(
        in_channels=5, out_channels=5, kernel_size=kernel_size,
        stride=stride, padding='half', dilation=dilation,
        merge_context0=merge_context0, merge_context1=merge_context1,
        resize_at_exit=True, **output_padding_arg,
    )
    ctx.assertIs(layer.merge_context0, merge_context0)
    ctx.assertIs(layer.merge_context1, merge_context1)
    ctx_shape = make_conv_shape([3], 5, [1] * spatial_ndims)
    context = [T.random.randn(ctx_shape), T.random.randn(ctx_shape)]

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x, context),
        (layer.shortcut(x) +
         context[0] * context[1] * layer.conv1(
                    context[0] + context[1] + layer.conv0(x))),
        rtol=1e-4, atol=1e-6,
    )

    # test initialized shortcut, conv0 and conv1
    shortcut = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                            stride=1, padding='half', dilation=1)
    conv0 = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                         stride=1, padding='half', dilation=1)
    conv1 = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                         stride=1, padding='half', dilation=1)
    layer = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                         stride=1, padding='half', dilation=1,
                         conv0=conv0, conv1=conv1, shortcut=shortcut)
    ctx.assertIs(layer.shortcut, shortcut)
    ctx.assertIs(layer.conv0, conv0)
    ctx.assertIs(layer.conv1, conv1)

    layer = tk.layers.jit_compile(layer)
    assert_allclose(
        layer(x),
        layer.shortcut(x) + layer.conv1(layer.conv0(x)),
        rtol=1e-4, atol=1e-6,
    )

    # test `weight_norm`
    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        weight_norm='full', **output_padding_arg
    )
    ctx.assertIsInstance(layer.shortcut.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv0.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv1.weight_store, tk.layers.NormedAndScaledWeightStore)

    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        weight_norm='no_scale', **output_padding_arg
    )
    ctx.assertIsInstance(layer.shortcut.weight_store, tk.layers.NormedWeightStore)
    ctx.assertIsInstance(layer.conv0.weight_store, tk.layers.NormedWeightStore)
    ctx.assertIsInstance(layer.conv1.weight_store, tk.layers.NormedWeightStore)

    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        weight_norm=True, **output_padding_arg
    )
    ctx.assertIsInstance(layer.shortcut.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv0.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv1.weight_store, tk.layers.NormedAndScaledWeightStore)

    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        weight_norm=True, normalizer=normalizer_cls, **output_padding_arg
    )
    ctx.assertIsInstance(layer.shortcut.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv0.weight_store, tk.layers.NormedWeightStore)
    ctx.assertIsInstance(layer.conv1.weight_store, tk.layers.NormedAndScaledWeightStore)

    layer = resblock_cls(
        in_channels=5, out_channels=4, kernel_size=kernel_size,
        stride=stride, padding=padding, dilation=dilation,
        weight_norm=True, normalizer=normalizer_cls, dropout=0.5,
        **output_padding_arg
    )
    ctx.assertIsInstance(layer.shortcut.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv0.weight_store, tk.layers.NormedAndScaledWeightStore)
    ctx.assertIsInstance(layer.conv1.weight_store, tk.layers.NormedAndScaledWeightStore)


class ResBlockTestCase(TestCase):

    def test_resblock(self):
        for spatial_ndims in (1, 2, 3):
            resblock_cls = getattr(tk.layers, f'ResBlock{spatial_ndims}d')
            check_resblock(
                ctx=self,
                spatial_ndims=spatial_ndims,
                resblock_cls=resblock_cls,
                linear_cls=getattr(tk.layers, f'LinearConv{spatial_ndims}d'),
                normalizer_cls=getattr(tk.layers, f'BatchNorm{spatial_ndims}d'),
                dropout_cls=getattr(tk.layers, f'Dropout{spatial_ndims}d'),
            )
            with pytest.raises(ValueError,
                               match='The `output_padding` argument is not allowed'):
                _ = resblock_cls(in_channels=5, out_channels=5, kernel_size=1,
                                 output_padding=1)

    def test_resblock_transpose(self):
        for spatial_ndims, output_padding in product((1, 2, 3), (0, 1)):
            check_resblock(
                ctx=self,
                spatial_ndims=spatial_ndims,
                resblock_cls=getattr(tk.layers, f'ResBlockTranspose{spatial_ndims}d'),
                linear_cls=getattr(tk.layers, f'LinearConvTranspose{spatial_ndims}d'),
                normalizer_cls=getattr(tk.layers, f'BatchNorm{spatial_ndims}d'),
                dropout_cls=getattr(tk.layers, f'Dropout{spatial_ndims}d'),
                output_padding_arg={'output_padding': output_padding},
            )
