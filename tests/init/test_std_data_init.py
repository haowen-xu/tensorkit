import unittest
from itertools import product

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *
from tests.ops import *


class StdDataInitTestCase(TestCase):

    def test_repr(self):
        data_init = tk.init.StdDataInit()
        self.assertTrue(repr(data_init).startswith(
            f'{tk.init.StdDataInit.__qualname__}('))
        self.assertIn(f'epsilon=', repr(data_init))

    def test_StdDataInit_for_Linear(self):
        in_features = 7
        out_features = 9
        data_init = tk.init.StdDataInit()

        for use_bias in [True, False]:
            for x in [T.random.randn([11, in_features]),
                      T.random.randn([13, 11, in_features])]:
                def check_x(layer):
                    y = layer(x)
                    y_mean, y_var = T.calculate_mean_and_var(
                        y, axis=T.int_range(-T.rank(x), -1))
                    if use_bias:
                        assert_allclose(y_mean, T.zeros_like(y_mean), atol=1e-6, rtol=1e-4)
                    assert_allclose(y_var, T.ones_like(y_var), atol=1e-6, rtol=1e-4)

                # construct the layer
                layer = tk.layers.Linear(
                    in_features, out_features, data_init=data_init,
                    use_bias=use_bias
                )

                # test initialize via data
                check_x(layer)

                # test new data will not cause it re-initialized
                _ = layer(T.random.randn(T.shape(x)))
                check_x(layer)

        # test error
        if T.is_module_jit_enabled():
            with pytest.raises(TypeError,
                               match='JIT compiled layer is not supported'):
                layer = tk.layers.jit_compile(tk.layers.Linear(5, 3))
                tk.init.StdDataInit()(layer, [T.random.randn([3, 5])])

        with pytest.raises(TypeError, match='`layer` is not a core linear layer'):
            layer = tk.layers.Dense(5, 3)
            tk.init.StdDataInit()(layer, [T.random.randn([3, 5])])

        with pytest.raises(ValueError, match='`inputs` must have exactly one input tensor'):
            layer = tk.layers.Linear(5, 3)
            tk.init.StdDataInit()(layer, [])

        with pytest.raises(ValueError, match='`inputs` must have exactly one input tensor'):
            layer = tk.layers.Linear(5, 3)
            tk.init.StdDataInit()(layer, [T.random.randn([3, 5]), T.random.randn([3, 5])])

    def test_StdDataInit_for_Conv(self):
        in_channels = 7
        out_channels = 9
        data_init = tk.init.StdDataInit()

        for spatial_ndims in (1, 2, 3):
            for transpose, use_bias, kernel_size, stride, padding, dilation in product(
                        (False, True),
                        (True, False),
                        (1, [3, 2, 1][:spatial_ndims]),
                        (1, [2, 3, 1][: spatial_ndims]),
                        (0, 'full'),
                        (1, [1, 3, 2][: spatial_ndims]),
                    ):
                if transpose:
                    cls_name = f'LinearConvTranspose{spatial_ndims}d'
                else:
                    cls_name = f'LinearConv{spatial_ndims}d'
                cls = getattr(tk.layers, cls_name)

                # prepare for the test
                x = T.random.randn(make_conv_shape(
                    [11], in_channels, [16, 15, 14][: spatial_ndims]))

                def check_x(layer):
                    y = layer(x)
                    y_mean, y_var = T.calculate_mean_and_var(
                        y, axis=[-T.rank(x)] + get_spatial_axis(spatial_ndims))
                    if use_bias:
                        assert_allclose(y_mean, T.zeros_like(y_mean), atol=1e-6, rtol=1e-4)
                    assert_allclose(y_var, T.ones_like(y_var), atol=1e-6, rtol=1e-4)

                # construct the layer
                layer = cls(
                    in_channels, out_channels, data_init=data_init,
                    use_bias=use_bias, kernel_size=kernel_size, stride=stride,
                    padding=padding,
                )

                # test initialize via data
                check_x(layer)

                # test new data will not cause it re-initialized
                _ = layer(T.random.randn(T.shape(x)))
                check_x(layer)
