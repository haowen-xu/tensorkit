import unittest
from itertools import product
from typing import *
from unittest.mock import Mock

import mltk
import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T, WeightNormMode, init, PaddingMode
from tensorkit.tensor import Tensor, Module
from tensorkit.layers import *
from tests.helper import *
from tests.ops import *


class _MyWrapper(BaseLayer):

    __constants__ = ('wrapped',)

    wrapped: Module

    def __init__(self, wrapped: Module):
        super().__init__()
        self.wrapped = wrapped

    def forward(self, input: Tensor) -> Tensor:
        return self.wrapped(input)


class _MyWrapper2(tk.layers.BaseLayer):

    layers: ModuleList

    def __init__(self, layers: Iterable[Module]):
        super().__init__()
        self.layers2 = list(layers)
        self.layers = tk.layers.ModuleList(self.layers2)

    @T.jit_ignore
    def my_check(self) -> bool:
        ret = True
        for l in self.layers:
            if not tk.layers.is_jit_layer(l):
                ret = False
                break
        return ret

    def forward(self, input: Tensor) -> Tensor:
        for l in self.layers:
            input = l(input)
        return input


class _MyGetTraining(BaseLayer):

    def forward(self) -> bool:
        return self.training


class UtilsAndConstantsTestCase(TestCase):

    def test_constants(self):
        self.assertEqual(tk.layers.DEFAULT_GATE_BIAS, 2.0)
        self.assertEqual(tk.layers.DEFAULT_WEIGHT_INIT, tk.init.kaming_uniform)
        self.assertEqual(tk.layers.DEFAULT_BIAS_INIT, tk.init.zeros)

    def test_param_and_buffer(self):
        layer = BaseLayer()

        # add parameter & buffer
        w_initial = T.random.randn([5, 4])
        c_initial = T.random.randn([5, 3])
        add_parameter(layer, 'w', w_initial)
        add_parameter(layer, 'b', None)
        add_buffer(layer, 'c', c_initial)
        add_buffer(layer, 'd', None)

        # get parameter and buffer
        assert_allclose(get_parameter(layer, 'w'), w_initial)
        self.assertIsNone(get_parameter(layer, 'b'))
        assert_allclose(get_buffer(layer, 'c'), c_initial)
        self.assertIsNone(get_buffer(layer, 'd'))

        # assignment
        w_value = np.random.randn(5, 4)
        with T.no_grad():
            T.assign_data(get_parameter(layer, 'w'), w_value)
        assert_allclose(get_buffer(layer, 'w'), w_value)

        # get parameters and buffers
        add_parameter(layer, 'w2', T.as_tensor(w_initial, force_copy=True))
        add_buffer(layer, 'c2', T.as_tensor(c_initial, force_copy=True))

        w = get_parameter(layer, 'w')
        w2 = get_parameter(layer, 'w2')
        c = get_buffer(layer, 'c')
        c2 = get_buffer(layer, 'c2')

        self.assertListEqual(list(iter_parameters(layer)), [w, w2])
        self.assertListEqual(get_parameters(layer), [w, w2])
        self.assertDictEqual(dict(iter_named_parameters(layer)), {'w': w, 'w2': w2})
        self.assertListEqual(list(iter_buffers(layer)), [c, c2])
        self.assertListEqual(get_buffers(layer), [c, c2])
        self.assertDictEqual(dict(iter_named_buffers(layer)), {'c': c, 'c2': c2})

        seq = _MyWrapper(layer)
        self.assertListEqual(list(iter_parameters(seq)), [w, w2])
        self.assertListEqual(list(iter_parameters(seq, recursive=False)), [])
        self.assertListEqual(get_parameters(seq, recursive=False), [])
        self.assertDictEqual(dict(iter_named_parameters(seq)), {'wrapped.w': w, 'wrapped.w2': w2})
        self.assertDictEqual(dict(iter_named_parameters(seq, recursive=False)), {})
        self.assertListEqual(list(iter_buffers(seq)), [c, c2])
        self.assertListEqual(list(iter_buffers(seq, recursive=False)), [])
        self.assertListEqual(get_buffers(seq, recursive=False), [])
        self.assertDictEqual(dict(iter_named_buffers(seq)), {'wrapped.c': c, 'wrapped.c2': c2})
        self.assertDictEqual(dict(iter_named_buffers(seq, recursive=False)), {})

    def test_layer_to_device(self):
        for device in [None, T.CPU_DEVICE]:
            layer = ResBlock2d(3, 4, kernel_size=2, device=device)
            for param in tk.layers.iter_parameters(layer):
                self.assertEqual(T.get_device(param), device or T.current_device())

            for device2 in [None, T.CPU_DEVICE]:
                layer2 = tk.layers.layer_to_device(layer, device=device2)
                for param in tk.layers.iter_parameters(layer2):
                    self.assertEqual(T.get_device(param), device2 or T.current_device())

    def test_set_train_mode(self):
        layers = [tk.layers.jit_compile(_MyGetTraining())
                  for _ in range(3)]
        layer = layers[0]

        # set_train_mode
        self.assertIs(tk.layers.set_train_mode(layer, True), layer)
        self.assertEqual(layer(), True)
        self.assertIs(tk.layers.set_train_mode(layer, False), layer)
        self.assertEqual(layer(), False)

        # set_eval_mode
        tk.layers.set_train_mode(layer, True)
        self.assertEqual(layer(), True)
        self.assertIs(tk.layers.set_eval_mode(layer), layer)
        self.assertEqual(layer(), False)

        # scoped_eval_mode
        for l in layers:
            tk.layers.set_train_mode(l, True)
            self.assertEqual(l(), True)
        with tk.layers.scoped_eval_mode(layers[0], layers[1:]):
            for l in layers:
                self.assertEqual(l(), False)
        for l in layers:
            self.assertEqual(l(), True)

    def test_SimpleParamStore(self):
        initial_value = np.random.randn(2, 3, 4)
        store = SimpleParamStore([2, 3, 4], initializer=initial_value)
        self.assertEqual(repr(store), 'SimpleParamStore(shape=[2, 3, 4])')
        assert_allclose(store.get(), initial_value, rtol=1e-4)
        assert_allclose(store(), initial_value, rtol=1e-4)

        new_value = np.random.randn(2, 3, 4)
        store.set(T.as_tensor(new_value))
        assert_allclose(store.get(), new_value, rtol=1e-4)
        assert_allclose(store(), new_value, rtol=1e-4)

    def test_NormedWeight(self):
        initial_value = np.random.randn(2, 3, 4)
        new_value = np.random.randn(2, 3, 4)

        for axis in [-3, -2, -1, 0, 1, 2]:
            store = NormedWeightStore(
                [2, 3, 4], initializer=initial_value, axis=axis)
            self.assertEqual(repr(store), 'NormedWeightStore(shape=[2, 3, 4])')
            expected_value = T.as_tensor(initial_value) / T.norm_except_axis(
                T.as_tensor(initial_value), axis=axis, keepdims=True)
            assert_allclose(store.get(), expected_value, rtol=1e-4)
            assert_allclose(store(), expected_value, rtol=1e-4)
            assert_allclose(store.v, expected_value, rtol=1e-4)

            store.set(T.as_tensor(new_value))
            expected_value = T.as_tensor(new_value) / T.norm_except_axis(
                T.as_tensor(new_value), axis=axis, keepdims=True)
            assert_allclose(store.get(), expected_value, rtol=1e-4)
            assert_allclose(store(), expected_value, rtol=1e-4)
            assert_allclose(store.v, expected_value, rtol=1e-4)

        for axis in (-4, 3):
            with pytest.raises(ValueError, match='`axis` out of range.'):
                _ = NormedWeightStore(
                    [2, 3, 4], initializer=initial_value, axis=axis)

    def test_NormedAndScaledWeightStore(self):
        initial_value = np.random.randn(2, 3, 4)
        new_value = np.random.randn(2, 3, 4)

        for axis in [-3, -2, -1, 0, 1, 2]:
            store = NormedAndScaledWeightStore(
                [2, 3, 4], initializer=initial_value, axis=axis)
            self.assertEqual(
                repr(store), 'NormedAndScaledWeightStore(shape=[2, 3, 4])')
            assert_allclose(store.get(), initial_value, rtol=1e-4)
            assert_allclose(store(), initial_value, rtol=1e-4)
            assert_allclose(
                store.g,
                T.norm_except_axis(T.as_tensor(initial_value), axis=axis,
                                   keepdims=True),
                rtol=1e-4
            )
            assert_allclose(store.v, T.as_tensor(initial_value) / store.g, rtol=1e-4)

            store.set(T.as_tensor(new_value))
            assert_allclose(store.get(), new_value, rtol=1e-4)
            assert_allclose(store(), new_value, rtol=1e-4)
            assert_allclose(
                store.g,
                T.norm_except_axis(T.as_tensor(new_value), axis=axis,
                                   keepdims=True),
                rtol=1e-4
            )
            assert_allclose(store.v, T.as_tensor(new_value) / store.g, rtol=1e-4)

        for axis in (-4, 3):
            with pytest.raises(ValueError, match='`axis` out of range.'):
                _ = NormedAndScaledWeightStore(
                    [2, 3, 4], initializer=initial_value, axis=axis)

    def test_get_weight_store(self):
        for wn in (True, WeightNormMode.FULL, 'full'):
            store = get_weight_store([2, 3, 4], weight_norm=wn)
            self.assertIsInstance(store, NormedAndScaledWeightStore)
            self.assertEqual(store.shape, [2, 3, 4])
        for wn in (WeightNormMode.NO_SCALE, 'no_scale'):
            store = get_weight_store([2, 3, 4], weight_norm=wn)
            self.assertIsInstance(store, NormedWeightStore)
            self.assertEqual(store.shape, [2, 3, 4])
        for wn in (False, WeightNormMode.NONE, 'none'):
            store = get_weight_store([2, 3, 4], weight_norm=wn)
            self.assertIsInstance(store, SimpleParamStore)
            self.assertEqual(store.shape, [2, 3, 4])
        with pytest.raises(ValueError,
                           match='Invalid value for argument `weight_norm`'):
            _ = get_weight_store([2, 3, 4], weight_norm='invalid')

    def test_get_bias_store(self):
        store = get_bias_store([2, 3, 4], use_bias=True)
        self.assertIsInstance(store, SimpleParamStore)
        self.assertEqual(store.shape, [2, 3, 4])

        store = get_bias_store([2, 3, 4], use_bias=False)
        self.assertIsNone(store)


class IdentityTestCase(TestCase):

    def test_identity(self):
        layer = tk.layers.jit_compile(Identity())
        x = T.random.randn([2, 3, 4])
        assert_equal(x, layer(x))


class _MySingleVariateLayer(BaseLayer):

    bias: float

    def __init__(self, bias: float = 0.):
        super().__init__()
        self.bias = bias

    @T.jit_method
    def set_bias(self, bias: float) -> None:
        self.bias = bias

    @T.jit_ignore
    def _add_numpy_array(self, x: Tensor) -> Tensor:
        return x + T.from_numpy(np.arange(x.shape[-1]),
                                dtype=T.get_dtype(x))

    def forward(self, x: Tensor) -> Tensor:
        return self._add_numpy_array(x * 11. + self.bias)


class _MyMultiVariateLayer(BaseLayer):

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        ret: List[Tensor] = []
        for i in range(len(inputs) - 1):
            ret.append(inputs[i] + inputs[i + 1])
        return ret


class _MySplitLayer(BaseLayer):

    def forward(self, input: Tensor) -> List[Tensor]:
        return [input, input + 1, input + 2]


class _MyMergeLayer(BaseLayer):

    def forward(self, inputs: List[Tensor]) -> Tensor:
        return T.add_n(inputs)


class _AutoRepr(BaseLayer):

    __constants__ = ('b',)

    internal: Module
    weight: Tensor
    a: str
    b: float


class BaseLayersTestCase(TestCase):

    def test_single_variate_layer(self):
        layer = tk.layers.jit_compile(_MySingleVariateLayer())
        x = T.random.randn([2, 3, 4])
        np_offset = T.from_numpy(np.array([0., 1., 2., 3.]))
        assert_allclose(layer(x), x * 11. + np_offset, rtol=1e-4, atol=1e-6)
        layer.set_bias(7.)
        assert_allclose(layer(x), x * 11. + 7. + np_offset, rtol=1e-4, atol=1e-6)

    def test_multi_variate_layer(self):
        layer = tk.layers.jit_compile(_MyMultiVariateLayer())
        x = T.random.randn([2, 3, 4])
        y = T.random.randn([2, 3, 4])
        z = T.random.randn([2, 3, 4])
        a, b = layer([x, y, z])
        assert_allclose(a, x + y, rtol=1e-4, atol=1e-6)
        assert_allclose(b, y + z, rtol=1e-4, atol=1e-6)

    def test_split_layer(self):
        layer = tk.layers.jit_compile(_MySplitLayer())
        x = T.random.randn([2, 3, 4])
        a, b, c = layer(x)
        assert_allclose(a, x, rtol=1e-4, atol=1e-6)
        assert_allclose(b, x + 1, rtol=1e-4, atol=1e-6)
        assert_allclose(c, x + 2, rtol=1e-4, atol=1e-6)

    def test_merge_layer(self):
        layer = tk.layers.jit_compile(_MyMergeLayer())
        x = T.random.randn([2, 3, 4])
        y = T.random.randn([2, 3, 4])
        z = T.random.randn([2, 3, 4])
        out = layer([x, y, z])
        assert_allclose(out, x + y + z, rtol=1e-4, atol=1e-6)

    def test_auto_repr(self):
        layer = _AutoRepr()
        layer.internal = Linear(5, 3)
        layer.weight = T.random.randn([3, 4])
        layer.a = 'hello'
        layer.b = 2.5
        self.assertTrue(repr(layer).startswith('_AutoRepr('))
        self.assertTrue(repr(layer).endswith(')'))
        self.assertIn('b=2.5, a=\'hello\'', repr(layer))
        self.assertNotIn('internal=', repr(layer))
        self.assertNotIn('weight=', repr(layer))


class ModuleListJitCompileTestCase(unittest.TestCase):

    def test_jit_compiled(self):
        if T.is_module_jit_enabled():
            layers = [tk.layers.Linear(3, 5), tk.layers.Linear(5, 4)]
            wrapper = _MyWrapper2(layers)
            wrapper = tk.layers.jit_compile(wrapper)
            x = T.random.randn([1, 3])
            assert_allclose(wrapper(x), layers[1](layers[0](x)))
            self.assertTrue(wrapper.my_check())


class SequentialTestCase(TestCase):

    def test_sequential(self):
        x = T.random.randn([4, 5])
        layers = [Linear(5, 5) for _ in range(5)]

        s = Sequential(layers[0], layers[1:2], [layers[2], [layers[3], layers[4]]])
        self.assertEqual(list(s), layers)
        y = tk.layers.jit_compile(s)(x)

        y2 = x
        for layer in layers:
            y2 = layer(y2)

        assert_allclose(y2, y, rtol=1e-4, atol=1e-6)


def check_core_linear(ctx, input, layer_factory, layer_name, numpy_fn):
    # test with bias
    layer = layer_factory(use_bias=True)
    ctx.assertIn(layer_name, repr(layer))
    ctx.assertIsInstance(layer.weight_store, SimpleParamStore)
    weight = T.to_numpy(layer.weight_store())
    bias = T.to_numpy(layer.bias_store())
    assert_allclose(tk.layers.jit_compile(layer)(T.as_tensor(input, dtype=T.float_x())),
                    numpy_fn(input, weight=weight, bias=bias),
                    rtol=1e-4, atol=1e-6)
    ctx.assertNotIn('use_bias=', repr(layer))

    # test without bias
    layer = layer_factory(use_bias=False)
    ctx.assertIsInstance(layer.weight_store, SimpleParamStore)
    weight = T.to_numpy(layer.weight_store())
    assert_allclose(tk.layers.jit_compile(layer)(T.as_tensor(input, dtype=T.float_x())),
                    numpy_fn(input, weight=weight, bias=None),
                    rtol=1e-4, atol=1e-6)
    ctx.assertIn('use_bias=False', repr(layer))

    # test `weight_norm`
    for wn in [True, WeightNormMode.FULL, 'full']:
        layer = layer_factory(use_bias=False, weight_norm=wn)
        ctx.assertIsInstance(layer.weight_store, NormedAndScaledWeightStore,
                             msg=f'weight_norm={wn}')
        weight = T.to_numpy(layer.weight_store())
        assert_allclose(tk.layers.jit_compile(layer)(T.as_tensor(input, dtype=T.float_x())),
                        numpy_fn(input, weight=weight, bias=None),
                        rtol=1e-4, atol=1e-6)

    for wn in [WeightNormMode.NO_SCALE, 'no_scale']:
        layer = layer_factory(use_bias=False, weight_norm=wn)
        ctx.assertIsInstance(layer.weight_store, NormedWeightStore,
                             msg=f'weight_norm={wn}')
        weight = T.to_numpy(layer.weight_store())
        assert_allclose(tk.layers.jit_compile(layer)(T.as_tensor(input, dtype=T.float_x())),
                        numpy_fn(input, weight=weight, bias=None),
                        rtol=1e-4, atol=1e-6)

    for wn in [False, WeightNormMode.NONE, 'none']:
        layer = layer_factory(use_bias=False, weight_norm=wn)
        ctx.assertIsInstance(layer.weight_store, SimpleParamStore,
                             msg=f'weight_norm={wn}')

    # test `data_init`
    class _MyDataDependentInitializer(init.DataDependentInitializer):
        register = Mock()

    data_init = _MyDataDependentInitializer()
    layer = layer_factory(data_init=data_init)
    ctx.assertEqual(data_init.register.call_args, ((layer,), {}))

    with pytest.raises(TypeError,
                       match=f'Unsupported data dependent initializer: '
                             f'\'hello\''):
        _ = layer_factory(data_init='hello')

    with pytest.raises(TypeError,
                       match=f'Unsupported data dependent initializer: '
                             f'\'hello\''):
        _ = layer_factory(data_init=lambda: 'hello')


class CoreLinearTestCase(TestCase):

    def test_linear(self):
        layer = Linear(5, 3)
        self.assertEqual(
            repr(layer),
            'Linear(in_features=5, out_features=3)'
        )

        check_core_linear(
            self,
            np.random.randn(5, 3).astype(np.float32),
            (lambda **kwargs: Linear(3, 4, **kwargs)),
            'Linear',
            dense,
        )
        check_core_linear(
            self,
            np.random.randn(10, 5, 3).astype(np.float32),
            (lambda **kwargs: Linear(3, 4, **kwargs)),
            'Linear',
            dense
        )

    @slow_test
    def test_conv_nd(self):
        def do_check(spatial_ndims, kernel_size, stride,
                     dilation, padding):
            cls_name = f'LinearConv{spatial_ndims}d'
            layer_factory = getattr(tk.layers, cls_name)
            check_core_linear(
                self,
                np.random.randn(
                    *make_conv_shape(
                        [2], 3, [14, 13, 12][: spatial_ndims]
                    )
                ).astype(np.float32),
                (lambda **kwargs: layer_factory(
                    in_channels=3, out_channels=4,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, dilation=dilation, **kwargs
                )),
                cls_name,
                (lambda input, weight, bias: conv_nd(
                    input, kernel=weight, bias=bias, stride=stride,
                    padding=padding, dilation=dilation,
                )),
            )

        for spatial_ndims in (1, 2):
            for kernel_size, stride, padding, dilation in product(
                        (1, (3, 2, 1)[: spatial_ndims]),
                        (1, (3, 2, 1)[: spatial_ndims]),
                        (0, 1, ((4, 3), 3, (2, 1))[: spatial_ndims],
                         PaddingMode.FULL, PaddingMode.HALF, PaddingMode.NONE),
                        (1, (3, 2, 1)[: spatial_ndims]),
                    ):
                do_check(spatial_ndims, kernel_size, stride, dilation, padding)

        # 3d is too slow, just do one particular test
        do_check(3, (3, 2, 1), (3, 2, 1), (3, 2, 1), PaddingMode.HALF)

    @slow_test
    def test_conv_transpose_nd(self):
        def is_valid_output_padding(spatial_ndims, output_padding, stride, dilation):
            if not hasattr(output_padding, '__iter__'):
                output_padding = [output_padding] * spatial_ndims
            if not hasattr(stride, '__iter__'):
                stride = [stride] * spatial_ndims
            if not hasattr(dilation, '__iter__'):
                dilation = [dilation] * spatial_ndims
            for op, s, d in zip(output_padding, stride, dilation):
                if op >= s and op >= d:
                    return False
            return True

        def do_check(spatial_ndims, kernel_size, stride,
                     dilation, padding, output_padding):
            cls_name = f'LinearConvTranspose{spatial_ndims}d'
            layer_factory = getattr(tk.layers, cls_name)
            fn = lambda: check_core_linear(
                self,
                np.random.randn(
                    *make_conv_shape(
                        [2], 3, [9, 8, 7][: spatial_ndims]
                    )
                ).astype(np.float32),
                (lambda **kwargs: layer_factory(
                    in_channels=3, out_channels=4,
                    kernel_size=kernel_size, stride=stride,
                    padding=padding, output_padding=output_padding,
                    dilation=dilation, **kwargs
                )),
                cls_name,
                (lambda input, weight, bias: conv_transpose_nd(
                    input, kernel=weight, bias=bias, stride=stride,
                    padding=padding, output_padding=output_padding,
                    dilation=dilation,
                )),
            )

            if not is_valid_output_padding(
                    spatial_ndims, output_padding, stride, dilation):
                with pytest.raises(Exception, match='`output_padding`'):
                    fn()
            else:
                fn()

        for spatial_ndims in (1, 2):
            for kernel_size, stride, padding, output_padding, dilation in product(
                        (1, (3, 2, 1)[: spatial_ndims]),
                        (1, (3, 2, 1)[: spatial_ndims]),
                        (0, 1, ((4, 3), 3, (2, 1))[: spatial_ndims],
                         PaddingMode.FULL, PaddingMode.HALF, PaddingMode.NONE),
                        (0, 1, (3, 2, 1)[: spatial_ndims]),
                        (1, (3, 2, 1)[: spatial_ndims]),
                    ):
                do_check(spatial_ndims, kernel_size, stride, dilation, padding,
                         output_padding)

        # 3d is too slow, just do one particular test
        do_check(3, (3, 2, 1), (3, 2, 1), (3, 2, 1), PaddingMode.HALF, 0)


class BatchNormTestCase(TestCase):

    def test_batch_norm(self):
        eps = T.EPSILON
        for spatial_ndims in (0, 1, 2, 3):
            cls = getattr(tk.layers, ('BatchNorm' if not spatial_ndims
                                      else f'BatchNorm{spatial_ndims}d'))
            layer = cls(5, momentum=0.1, epsilon=eps)
            self.assertIn('BatchNorm', repr(layer))
            self.assertTrue(tk.layers.is_batch_norm(layer))
            layer = tk.layers.jit_compile(layer)

            # layer output
            x = T.random.randn(make_conv_shape(
                [3], 5, [6, 7, 8][:spatial_ndims]
            ))

            set_train_mode(layer)
            _ = layer(x)
            set_train_mode(layer, False)
            y = layer(x)
            set_train_mode(layer, True)
            set_eval_mode(layer)
            y2 = layer(x)

            # manually compute the expected output
            if T.backend_name == 'PyTorch':
                dst_shape = [-1] + [1] * spatial_ndims
                weight = T.reshape(layer.weight, dst_shape)
                bias = T.reshape(layer.bias, dst_shape)
                running_mean = T.reshape(layer.running_mean, dst_shape)
                running_var = T.reshape(layer.running_var, dst_shape)
                expected = (((x - running_mean) / T.sqrt(running_var + eps)) *
                            weight + bias)
            else:
                raise RuntimeError()

            # check output
            assert_allclose(y, expected, rtol=1e-4, atol=1e-6)
            assert_allclose(y2, expected, rtol=1e-4, atol=1e-6)

            # check invalid dimensions
            with pytest.raises(Exception, match='only supports .d input'):
                _ = layer(
                    T.random.randn(make_conv_shape(
                        [], 5, [6, 7, 8][:spatial_ndims]
                    ))
                )

    def test_is_batch_norm(self):
        layers = [
            tk.layers.Dense(3, 4),
            tk.layers.BatchNorm(4),
            tk.layers.jit_compile(tk.layers.BatchNorm(4))
        ]

        self.assertFalse(tk.layers.is_batch_norm(layers[0]))
        self.assertTrue(tk.layers.is_batch_norm(layers[1]))
        if not T.is_module_jit_enabled():
            self.assertTrue(tk.layers.is_batch_norm(layers[2]))

        self.assertFalse(tk.layers.has_batch_norm(layers[0]))
        self.assertTrue(tk.layers.has_batch_norm(layers[1]))
        if not T.is_module_jit_enabled():
            self.assertTrue(tk.layers.has_batch_norm(layers[2]))

        seq = tk.layers.Sequential(layers)
        self.assertFalse(tk.layers.is_batch_norm(seq))
        self.assertTrue(tk.layers.has_batch_norm(seq))
        self.assertFalse(tk.layers.has_batch_norm(seq, recursive=False))

    def test_batch_norm_reset(self):
        T.random.seed(1234)
        inputs = T.split(T.random.randn([21, 4]), sections=[7] * 3, axis=0)

        layer = tk.layers.Sequential([
            tk.layers.BatchNorm(4),
            tk.layers.BatchNorm(4),
        ])
        outputs = []

        for i in range(3):
            if i == 1:
                tk.layers.batch_norm_reset(layer)
            tk.layers.set_train_mode(layer)
            for x in inputs[:-1]:
                _ = layer(x)
            tk.layers.set_eval_mode(layer)
            outputs.append(layer(inputs[-1]))

        assert_allclose(outputs[0], outputs[1], rtol=1e-4, atol=1e-6)
        assert_not_allclose(outputs[1], outputs[2], rtol=1e-4, atol=1e-6)

    def test_batch_norm_full_init(self):
        T.random.seed(1234)
        orig_input = T.random.randn([35, 4])
        inputs = T.split(orig_input, sections=[7] * 5, axis=0)
        mean, var = T.calculate_mean_and_var(orig_input, axis=[0])

        layer = tk.layers.Sequential([
            tk.layers.BatchNorm(4, momentum=0.125),
        ])

        def step_fn(input):
            _ = layer(input)

        def data_generator():
            for input in inputs:
                yield (input,)

        for i in range(4):
            if i == 2:
                loop = mltk.TrainLoop()
            elif i == 3:
                loop = mltk.TestLoop()
            else:
                loop = None

            fn = lambda: tk.layers.batch_norm_full_init(
                layer, data_generator(), step_fn, loop=loop)
            tk.layers.set_train_mode(layer)

            if loop is not None:
                with loop:
                    fn()
            else:
                fn()
            tk.layers.set_eval_mode(layer)

            if T.backend_name == 'PyTorch':
                self.assertEqual(layer[0].momentum, 0.125)
                assert_allclose(layer[0].running_mean, mean, atol=1e-4, rtol=1e-6)
            else:
                raise NotImplementedError()


class DropoutTestCase(TestCase):

    def test_dropout(self):
        n_samples = 10000
        for spatial_ndims in (0, 1, 2, 3):
            cls = getattr(tk.layers, ('Dropout' if not spatial_ndims
                                      else f'Dropout{spatial_ndims}d'))
            layer = cls(p=0.3)
            self.assertIn('p=0.3', repr(layer))
            self.assertIn('Dropout', repr(layer))
            layer = tk.layers.jit_compile(layer)

            x = 1. + T.random.rand(
                make_conv_shape([1], n_samples, [2, 2, 2][:spatial_ndims])
            )

            # ---- train mode ----
            set_train_mode(layer, True)
            y = layer(x)

            # check: channels should be all zero or no zero
            spatial_axis = tuple(get_spatial_axis(spatial_ndims))

            all_zero = np.all(T.to_numpy(y) == 0, axis=spatial_axis, keepdims=True)
            no_zero = np.all(T.to_numpy(y) != 0, axis=spatial_axis, keepdims=True)
            self.assertTrue(np.all(np.logical_or(all_zero, no_zero)))

            # check: the probability of not being zero
            self.assertLessEqual(
                np.abs(np.mean(all_zero) - 0.3),
                5.0 / np.sqrt(n_samples) * 0.3 * 0.7  # 5-sigma
            )

            # check: the value
            assert_allclose(y, (T.to_numpy(x) * no_zero) / 0.7,
                            rtol=1e-4, atol=1e-6)

            # ---- eval mode ----
            set_train_mode(layer, False)
            y = layer(x)
            self.assertTrue(np.all(T.to_numpy(y) != 0))
            assert_allclose(y, x, rtol=1e-4, atol=1e-6)

            set_train_mode(layer, True)
            set_eval_mode(layer)
            y = layer(x)
            self.assertTrue(np.all(T.to_numpy(y) != 0))
            assert_allclose(y, x, rtol=1e-4, atol=1e-6)


class EmbeddingTestCase(TestCase):

    def test_embedding(self):
        n_channels = 3
        n_embeddings = n_samples

        for spatial_ndims in (0, 1, 2, 3):
            w_shape = make_conv_shape([], n_channels, [4, 5, 6][:spatial_ndims])
            w_size = int(np.prod(w_shape))

            layer = getattr(tk.layers, (f'Embedding{spatial_ndims}d'
                                        if spatial_ndims > 0 else 'Embedding'))(
                n_embeddings,
                w_shape
            )
            weight = layer.weight

            # check the weight
            self.assertEqual(T.shape(weight), [n_embeddings] + w_shape)
            reduce_axis = list(range(len(w_shape) + 1))
            reduce_axis.pop(-1 if T.IS_CHANNEL_LAST else 1)
            w_mean = np.average(T.to_numpy(weight), axis=tuple(reduce_axis))
            np.testing.assert_array_less(
                w_mean,
                3. / np.sqrt(n_samples * w_size / n_channels)
            )

            # check the output
            layer = jit_compile(layer)
            weight_array = T.to_numpy(weight)

            for in_shape in ([7], [7, 8]):
                indices = T.random.randint(0, n_samples, in_shape)
                indices = T.concat([indices, indices[:3]], axis=0)

                # check the output
                output = layer(indices)
                assert_allclose(output, T.embedding(weight, indices))

                # check the grad
                if spatial_ndims in (0, 1):
                    out_sum = T.reduce_sum(output ** 2)
                    [grad] = T.grad([out_sum], [weight])
                    expected_grad = np.zeros(T.shape(weight))
                    for idx in T.to_numpy(indices).reshape([-1]):
                        expected_grad[idx] += 2. * weight_array[idx]
                    assert_allclose(grad, expected_grad)

            # test the constructor error
            if spatial_ndims > 0:
                with pytest.raises(ValueError,
                                   match=f'`embedding_size` must be a int list '
                                         f'with {spatial_ndims + 1} elements'):
                    _ = getattr(tk.layers, f'Embedding{spatial_ndims}d')(
                        n_embeddings,
                        w_shape[:-1]
                    )

        # test no grad
        layer = Embedding(n_embeddings, n_channels, freeze=True)
        weight = layer.weight
        self.assertEqual(T.shape(weight), [n_embeddings, n_channels])

        layer = jit_compile(layer)
        indices = T.random.randint(0, n_samples, [7, 8])
        output = layer(indices)
        assert_allclose(output, T.embedding(weight, indices))

        out_sum = T.reduce_sum(output ** 2)
        try:
            [grad] = T.grad([out_sum], [weight])
        except Exception:
            pass
        else:
            self.assertTrue(T.is_null_grad(weight, grad))

        # test errors
        with pytest.raises(ValueError,
                           match='`embedding_size` must not be empty'):
            _ = Embedding(n_embeddings, [])
