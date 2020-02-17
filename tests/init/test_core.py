import math
import unittest
from functools import partial
from itertools import product
from typing import *
from unittest.mock import Mock

import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class UtilitiesTestCase(TestCase):

    def test_calculate_fan_in_and_fan_out(self):
        for layer, fan_in_and_out in [
                    (tk.layers.Linear(5, 3), (5, 3)),
                    (tk.layers.LinearConv1d(5, 3, 2), (10, 6)),
                    (tk.layers.LinearConv2d(5, 3, (2, 3)), (30, 18)),
                    (tk.layers.LinearConv3d(5, 3, (2, 3, 4)), (120, 72)),
                    (tk.layers.LinearConvTranspose1d(5, 3, 2), (6, 10)),
                    (tk.layers.LinearConvTranspose2d(5, 3, (2, 3)), (18, 30)),
                    (tk.layers.LinearConvTranspose3d(5, 3, (2, 3, 4)), (72, 120)),
                ]:
            self.assertEqual(
                tk.init.calculate_fan_in_and_fan_out(layer.weight_store.get()),
                fan_in_and_out
            )

        with pytest.raises(Exception, match=r'`rank\(tensor\)` < 2'):
            _ = tk.init.calculate_fan_in_and_fan_out(T.random.randn([3]))

    def test_get_activation_gain(self):
        def leaky_relu_gain(negative_slope=T.nn.LEAKY_RELU_DEFAULT_SLOPE):
            return math.sqrt(2.0 / (1. + negative_slope ** 2))

        for origin_name, args, kwargs, gain in [
                    (None, (), {}, 1.0),
                    ('Linear', (), {}, 1.0),
                    ('Sigmoid', (), {}, 1.0),
                    ('Tanh', (), {}, 5.0 / 3),
                    ('ReLU', (), {}, math.sqrt(2)),
                    ('Leaky_ReLU', (), {}, leaky_relu_gain()),
                    ('Leaky_ReLU', (0.2,), {}, leaky_relu_gain(0.2)),
                    ('Leaky_ReLU', (), {'negative_slope': 0.2}, leaky_relu_gain(0.2))
                ]:
            name_candidates = (None,) if origin_name is None else (
                origin_name,
                origin_name.lower(),
                origin_name.replace('_', ''),
                origin_name.replace('_', '').lower()
            )
            for name in name_candidates:
                err_msg = f'{name}, {args}, {kwargs}, {gain}'

                # by class
                self.assertAlmostEqual(
                    tk.init.get_activation_gain(
                        tk.layers.get_activation_class(name),
                        *args, **kwargs
                    ),
                    gain,
                    msg=err_msg
                )

                # by instance
                factory = tk.layers.get_activation_class(name)
                if factory is not None:
                    self.assertAlmostEqual(
                        tk.init.get_activation_gain(factory(*args, **kwargs)),
                        gain,
                        msg=err_msg
                    )

    def test_apply_initializer(self):
        for dtype in float_dtypes:
            weight = T.variable([5, 3], dtype=dtype)
            fan_in_and_fan_out = tk.init.calculate_fan_in_and_fan_out(weight)
            initializer = Mock()

            # test by value
            tk.init.apply_initializer(weight, 123)
            assert_equal(weight, T.full_like(weight, 123))
            tk.init.apply_initializer(weight, 124.)
            assert_equal(weight, T.full_like(weight, 124.))
            tk.init.apply_initializer(weight, np.array(125.))
            assert_equal(weight, T.full_like(weight, 125.))

            value = np.random.randn(*T.shape(weight)).astype(dtype)
            tk.init.apply_initializer(weight, value)
            assert_equal(weight, value)

            # test by initializer
            initializer.reset_mock()
            tk.init.apply_initializer(weight, initializer)
            self.assertEqual(
                initializer.call_args,
                ((weight,), {
                    'gain': 1.0,
                    'mode': 'fan_in',
                    'fan_in_and_fan_out': fan_in_and_fan_out,
                })
            )

            # test fan_in_and_fan_out
            initializer.reset_mock()
            tk.init.apply_initializer(
                weight, initializer, fan_in_and_fan_out=(2, 3))
            self.assertEqual(
                initializer.call_args,
                ((weight,), {
                    'gain': 1.0,
                    'mode': 'fan_in',
                    'fan_in_and_fan_out': (2, 3),
                })
            )

            initializer.reset_mock()
            tk.init.apply_initializer(weight, initializer, mode='fan_out')
            self.assertEqual(
                initializer.call_args,
                ((weight,), {
                    'gain': 1.0,
                    'mode': 'fan_out',
                    'fan_in_and_fan_out': fan_in_and_fan_out,
                })
            )

            # test gain
            initializer.reset_mock()
            tk.init.apply_initializer(weight, initializer, gain=1.5)
            self.assertEqual(
                initializer.call_args,
                ((weight,), {
                    'gain': 1.5,
                    'mode': 'fan_in',
                    'fan_in_and_fan_out': fan_in_and_fan_out,
                })
            )

            for activation in ['LeakyReLU', tk.layers.ReLU, tk.layers.Tanh()]:
                initializer.reset_mock()
                init_gain = tk.init.get_activation_gain(activation)
                tk.init.apply_initializer(
                    weight, initializer, activation=activation)
                self.assertEqual(
                    initializer.call_args,
                    ((weight,), {
                        'gain': init_gain,
                        'mode': 'fan_in',
                        'fan_in_and_fan_out': fan_in_and_fan_out,
                    })
                )

            # unsupported initializer
            with pytest.raises(TypeError, match='Unsupported initializer'):
                tk.init.apply_initializer(weight, object())


class TensorInitiailizersTestCase(TestCase):

    def test_zeros(self):
        for dtype in float_dtypes:
            weight = T.variable([2, 3, 4], dtype=dtype, initializer=1.)
            assert_equal(weight, T.full_like(weight, 1.))
            tk.init.apply_initializer(weight, tk.init.zeros)
            assert_equal(weight, T.full_like(weight, 0.))

    def test_ones(self):
        for dtype in float_dtypes:
            weight = T.variable([2, 3, 4], dtype=dtype, initializer=0.)
            assert_equal(weight, T.full_like(weight, 0.))
            tk.init.apply_initializer(weight, tk.init.ones)
            assert_equal(weight, T.full_like(weight, 1.))

    def test_fill(self):
        for dtype in float_dtypes:
            weight = T.variable([2, 3, 4], dtype=dtype, initializer=0.)
            assert_equal(weight, T.full_like(weight, 0.))
            tk.init.apply_initializer(
                weight, partial(tk.init.fill, fill_value=123.))
            assert_equal(weight, T.full_like(weight, 123.))

    def test_uniform(self):
        for dtype in float_dtypes:
            weight = T.variable([n_samples // 50, 50], dtype=dtype,
                                initializer=0.)
            assert_equal(weight, T.full_like(weight, 0.))

            # uniform with default args
            tk.init.apply_initializer(weight, tk.init.uniform)
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight)) - 0.5),
                5.0 / np.sqrt(12.) / np.sqrt(n_samples)
            )

            # uniform with customized args
            tk.init.apply_initializer(
                weight, partial(tk.init.uniform, low=-4., high=3.))
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight)) - (-0.5)),
                5.0 * 7.0 / np.sqrt(12.) / np.sqrt(n_samples)
            )

    def test_normal(self):
        for dtype in float_dtypes:
            weight = T.variable([n_samples // 50, 50], dtype=dtype,
                                initializer=0.)
            assert_equal(weight, T.full_like(weight, 0.))

            # uniform with default args
            tk.init.apply_initializer(weight, tk.init.normal)
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight))),
                5.0 / np.sqrt(n_samples)
            )

            # uniform with customized args
            tk.init.apply_initializer(
                weight, partial(tk.init.normal, mean=1., std=3.))
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight)) - 1.),
                5.0 * 3. / np.sqrt(n_samples)
            )

    def test_xavier_initializer(self):
        for dtype, initializer, mode in product(
                    float_dtypes,
                    (tk.init.xavier_normal, tk.init.xavier_uniform),
                    (None, 'fan_in', 'fan_out'),
                ):
            weight = T.variable([n_samples // 50, 50], dtype=dtype,
                                initializer=0.)
            assert_equal(weight, T.full_like(weight, 0.))
            mode_arg = {'mode': mode} if mode is not None else {}

            # xavier
            fan_in, fan_out = tk.init.calculate_fan_in_and_fan_out(weight)
            xavier_std = np.sqrt(2.0 / float(fan_in + fan_out))
            tk.init.apply_initializer(weight, initializer, **mode_arg)
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight))),
                5.0 / xavier_std / np.sqrt(n_samples)
            )

            # xavier with custom gain and fan_in/fan_out
            fan_in, fan_out = 23, 17
            init_gain = 1.5
            xavier_std = init_gain * np.sqrt(2.0 / float(fan_in + fan_out))
            tk.init.apply_initializer(
                weight, initializer,
                fan_in_and_fan_out=(fan_in, fan_out),
                gain=init_gain,
                **mode_arg
            )
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight))),
                5.0 / xavier_std / np.sqrt(n_samples)
            )

    def test_kaming_initializer(self):
        for dtype, initializer, mode in product(
                    float_dtypes,
                    (tk.init.kaming_normal, tk.init.kaming_uniform),
                    (None, 'fan_in', 'fan_out'),
                ):
            weight = T.variable([n_samples // 50, 50], dtype=dtype,
                                initializer=0.)
            assert_equal(weight, T.full_like(weight, 0.))
            mode_arg = {'mode': mode} if mode is not None else {}

            # kaming
            fan_in, fan_out = tk.init.calculate_fan_in_and_fan_out(weight)
            if mode == 'fan_out':
                kaming_std = np.sqrt(1.0 / np.sqrt(fan_out))
            else:
                kaming_std = np.sqrt(1.0 / np.sqrt(fan_in))
            tk.init.apply_initializer(weight, initializer, **mode_arg)
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight))),
                5.0 / kaming_std / np.sqrt(n_samples)
            )

            # kaming with custom gain and fan_in/fan_out
            fan_in, fan_out = 23, 17
            init_gain = 1.5
            if mode == 'fan_out':
                kaming_std = init_gain * np.sqrt(1.0 / np.sqrt(fan_out))
            else:
                kaming_std = init_gain * np.sqrt(1.0 / np.sqrt(fan_in))
            tk.init.apply_initializer(
                weight, initializer,
                fan_in_and_fan_out=(fan_in, fan_out),
                gain=init_gain,
                **mode_arg
            )
            self.assertLessEqual(
                np.abs(T.to_numpy(T.reduce_mean(weight))),
                5.0 / kaming_std / np.sqrt(n_samples)
            )

            # test error
            with pytest.raises(ValueError,
                               match='`mode` must be either "fan_in" or "fan_out"'):
                weight = T.variable([n_samples // 50, 50], dtype=dtype,
                                    initializer=0.)
                tk.init.apply_initializer(weight, initializer, mode='invalid')


class _MyDataDependentInitializer(tk.init.DataDependentInitializer):

    watcher: List[Tuple[T.Module, List[T.Tensor]]]

    def __init__(self, watcher):
        self.watcher = watcher

    def _init(self, layer: T.Module, inputs: List[T.Tensor]) -> None:
        _ = layer(inputs[0])
        self.watcher.append((layer, inputs))


class DataDependentInitializerTestCase(TestCase):

    def test_data_dependent_initializer(self):
        data_init = _MyDataDependentInitializer([])

        # construct with the initializer
        data_init.watcher.clear()
        layer = tk.layers.Sequential(
            tk.layers.Linear(5, 3, data_init=data_init))
        self.assertEqual(data_init.watcher, [])
        x = T.random.randn([2, 5])
        y = T.random.randn([2, 5])
        _ = layer(x)
        _ = layer(y)
        self.assertListEqual(data_init.watcher, [(layer[0], [x])])

        # set_initialized(False) to re-enable the initializer
        data_init.watcher.clear()
        tk.init.set_initialized(layer, False)
        x = T.random.randn([2, 5])
        y = T.random.randn([2, 5])
        _ = layer(x)
        _ = layer(y)
        self.assertListEqual(data_init.watcher, [(layer[0], [x])])

        # set_initialize(True) to disable newly constructed data-init
        data_init.watcher.clear()
        layer = tk.layers.Sequential(
            tk.layers.Linear(5, 3, data_init=data_init))
        tk.init.set_initialized(layer, True)
        x = T.random.randn([2, 5])
        _ = layer(x)
        self.assertListEqual(data_init.watcher, [])

        # remove the data-dependent initializers
        data_init.watcher.clear()
        layer = tk.layers.Sequential(
            tk.layers.Linear(5, 3, data_init=data_init))
        tk.init.remove_data_dependent_initializers(layer)
        tk.init.set_initialized(layer, False)
        x = T.random.randn([2, 5])
        _ = layer(x)
        self.assertListEqual(data_init.watcher, [])

        # also `set_initialized` will affect layers with `set_initialized()`
        # method, e.g., `ActNorm`
        x = T.random.randn([2, 3, 5])
        layer = tk.layers.jit_compile(tk.layers.ActNorm(5))
        self.assertFalse(layer.flow.initialized)

        tk.init.set_initialized(layer)
        self.assertTrue(layer.flow.initialized)
        assert_allclose(layer(x), x, rtol=1e-4, atol=1e-6)

        tk.init.set_initialized(layer, False)
        self.assertFalse(layer.flow.initialized)
        y = layer(x)
        y_mean, y_var = T.calculate_mean_and_var(y, axis=[0, 1])
        assert_allclose(y_mean, T.zeros_like(y_mean), rtol=1e-4, atol=1e-6)
        assert_allclose(y_var, T.ones_like(y_var), rtol=1e-4, atol=1e-6)

        self.assertTrue(layer.flow.initialized)
        assert_allclose(layer(x), y, rtol=1e-4, atol=1e-6)

