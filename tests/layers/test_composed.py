import unittest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *
from tests.ops import *


def check_composed_layer(ctx, input, layer_cls, linear_cls, normalizer_cls,
                         in_features, out_features, **kwargs):
    # test pure
    for use_bias in [None, True, False]:
        layer = layer_cls(
            in_features, out_features, use_bias=use_bias,
            bias_init=tk.init.uniform, **kwargs
        )
        expected_use_bias = True if use_bias is None else use_bias
        linear = linear_cls(
            in_features, out_features,
            weight_init=layer[0].weight_store.get(),
            bias_init=(layer[0].bias_store.get() if expected_use_bias
                       else None),
            use_bias=expected_use_bias,
            **kwargs
        )
        ctx.assertIn(layer_cls.__qualname__, repr(layer))
        ctx.assertIsInstance(layer[0], linear_cls)
        ctx.assertEqual(layer[0].use_bias, expected_use_bias)
        assert_allclose(
            T.jit_compile(layer)(input),
            linear(input)
        )

    # test normalizer
    for use_bias in [None, True, False]:
        for normalizer_arg in [normalizer_cls, normalizer_cls(out_features)]:
            layer = layer_cls(
                in_features, out_features, normalizer=normalizer_arg,
                use_bias=use_bias, bias_init=tk.init.uniform, **kwargs
            )
            expected_use_bias = False if use_bias is None else use_bias
            linear = linear_cls(
                in_features, out_features,
                weight_init=layer[0].weight_store.get(),
                bias_init=(layer[0].bias_store.get() if expected_use_bias
                           else None),
                use_bias=expected_use_bias,
                **kwargs
            )
            normalizer = normalizer_cls(out_features)
            ctx.assertIsInstance(layer[0], linear_cls)
            ctx.assertEqual(layer[0].use_bias, expected_use_bias)
            ctx.assertIsInstance(layer[1], normalizer_cls)
            assert_allclose(
                T.jit_compile(layer)(input),
                normalizer(linear(input)),
            )

    # test activation
    activation_cls = tk.layers.Tanh
    for activation_arg in [activation_cls, activation_cls()]:
        layer = layer_cls(
            in_features, out_features, activation=activation_arg,
            bias_init=tk.init.uniform, **kwargs
        )
        linear = linear_cls(
            in_features, out_features,
            weight_init=layer[0].weight_store.get(),
            bias_init=layer[0].bias_store.get(),
            **kwargs
        )
        ctx.assertIsInstance(layer[0], linear_cls)
        ctx.assertIsInstance(layer[1], tk.layers.Tanh)
        assert_allclose(
            T.jit_compile(layer)(input),
            activation_cls()(linear(input)),
        )

    # test gate
    layer = layer_cls(
        in_features, out_features,
        bias_init=tk.init.uniform, gated=True, **kwargs
    )
    linear = linear_cls(
        in_features, out_features * 2.,
        weight_init=layer[0].weight_store.get(),
        bias_init=layer[0].bias_store.get(),
        **kwargs
    )
    ctx.assertIsInstance(layer[0], linear_cls)
    out = linear(input)
    assert_allclose(
        T.jit_compile(layer)(input),
        T.nn.sigmoid(out[:, out_features:] + 2.0) * out[:, :out_features],
    )

    # test gate + activation
    activation = tk.layers.LeakyReLU()
    layer = layer_cls(
        in_features, out_features, activation=activation,
        bias_init=tk.init.uniform, gated=True, **kwargs
    )
    linear = linear_cls(
        in_features, out_features * 2.,
        weight_init=layer[0].weight_store.get(),
        bias_init=layer[0].bias_store.get(),
        **kwargs
    )
    ctx.assertIsInstance(layer[0], linear_cls)
    out = linear(input)
    assert_allclose(
        T.jit_compile(layer)(input),
        (T.nn.sigmoid(out[:, out_features:] + 2.0) *
         activation(out[:, :out_features])),
    )

    # test normalizer + gate + activation
    normalizer = normalizer_cls(out_features * 2)
    activation = tk.layers.LeakyReLU()
    layer = layer_cls(
        in_features, out_features, activation=activation, normalizer=normalizer,
        gated=True, **kwargs
    )
    ctx.assertFalse(layer[0].use_bias)
    linear = linear_cls(
        in_features, out_features * 2.,
        weight_init=layer[0].weight_store.get(),
        use_bias=False,
        **kwargs
    )
    ctx.assertIsInstance(layer[0], linear_cls)
    out = normalizer(linear(input))
    assert_allclose(
        T.jit_compile(layer)(input),
        (T.nn.sigmoid(out[:, out_features:] + 2.0) *
         activation(out[:, :out_features])),
    )


class ComposedTestCase(unittest.TestCase):

    def test_dense(self):
        check_composed_layer(
            self,
            T.random.randn([5, 4]),
            tk.layers.Dense,
            tk.layers.Linear,
            tk.layers.BatchNorm,
            4, 3
        )

    def test_conv_nd(self):
        for spatial_ndims in (1, 2, 3):
            check_composed_layer(
                self,
                T.random.randn(make_conv_shape(
                    [5], 4, [16, 15, 14][:spatial_ndims]
                )),
                getattr(tk.layers, f'Conv{spatial_ndims}d'),
                getattr(tk.layers, f'LinearConv{spatial_ndims}d'),
                getattr(tk.layers, f'BatchNorm{spatial_ndims}d'),
                4, 3,
                kernel_size=3, stride=2, dilation=2, padding='half'
            )

    def test_conv_transpose_nd(self):
        for spatial_ndims in (1, 2, 3):
            for output_padding in (0, 1):
                check_composed_layer(
                    self,
                    T.random.randn(make_conv_shape(
                        [5], 4, [16, 15, 14][:spatial_ndims]
                    )),
                    getattr(tk.layers, f'ConvTranspose{spatial_ndims}d'),
                    getattr(tk.layers, f'LinearConvTranspose{spatial_ndims}d'),
                    getattr(tk.layers, f'BatchNorm{spatial_ndims}d'),
                    4, 3,
                    kernel_size=3, stride=2, dilation=2, padding='half',
                    output_padding=output_padding,
                )
