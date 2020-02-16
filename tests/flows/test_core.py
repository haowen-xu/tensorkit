import math
import unittest
from typing import Optional, Tuple

import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.tensor import Tensor, reshape_tail, as_tensor_backend, zeros_like, shape
from tensorkit.tensor.random import randn
from tensorkit.flows import *
from tests.helper import *
from tests.ops import *


class _MyFlow(Flow):

    def __init__(self):
        super().__init__(x_event_ndims=1,
                         y_event_ndims=2,
                         explicitly_invertible=True)

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = reshape_tail(0.5 * (input - 1.), 2, [-1])
        else:
            output = reshape_tail(input * 2. + 1., 1, [-1, 1])

        output_log_det = input_log_det
        if compute_log_det:
            log_2 = as_tensor_backend(math.log(2.), dtype=output.dtype)
            if output_log_det is None:
                if inverse:
                    output_log_det = -log_2 * input.shape[-2]
                else:
                    output_log_det = log_2 * input.shape[-1]
            else:
                if inverse:
                    output_log_det = output_log_det - log_2 * input.shape[-2]
                else:
                    output_log_det = output_log_det + log_2 * input.shape[-1]
        return output, output_log_det


class _MyBadFlow(Flow):

    def __init__(self):
        super().__init__(x_event_ndims=1,
                         y_event_ndims=1,
                         explicitly_invertible=True)

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        output = input
        output_log_det = input_log_det
        if compute_log_det:
            if output_log_det is None:
                output_log_det = zeros_like(output)
            else:
                output_log_det = input_log_det
        return output, output_log_det


class BaseFlowTestCase(unittest.TestCase):

    def test_constructor(self):
        flow = Flow(x_event_ndims=1,
                    y_event_ndims=2,
                    explicitly_invertible=True)
        self.assertEqual(flow.get_x_event_ndims(), 1)
        self.assertEqual(flow.get_y_event_ndims(), 2)
        self.assertEqual(flow.is_explicitly_invertible(), True)

        flow = Flow(x_event_ndims=3,
                    y_event_ndims=1,
                    explicitly_invertible=False)
        self.assertEqual(flow.get_x_event_ndims(), 3)
        self.assertEqual(flow.get_y_event_ndims(), 1)
        self.assertEqual(flow.is_explicitly_invertible(), False)

    def test_invert(self):
        flow = _MyFlow()
        inv_flow = flow.invert()
        self.assertIsInstance(inv_flow, InverseFlow)

    def test_call(self):
        flow = T.jit_compile(_MyFlow())
        self.assertEqual(flow.get_x_event_ndims(), 1)
        self.assertEqual(flow.get_y_event_ndims(), 2)
        self.assertEqual(flow.is_explicitly_invertible(), True)

        # test call
        x = T.random.randn([2, 3, 4])
        expected_y = T.reshape(x * 2. + 1., [2, 3, 4, 1])
        expected_log_det = T.full([2, 3], math.log(2.) * 4)
        input_log_det = T.random.randn([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            input_log_det)

        # test input shape error
        with pytest.raises(Exception,
                           match='`input` is required to be at least .*d'):
            _ = flow(T.random.randn([]))
        with pytest.raises(Exception,
                           match='`input` is required to be at least .*d'):
            _ = flow(T.random.randn([3]), inverse=True)

        # test input_log_det shape error
        with pytest.raises(Exception,
                           match='The shape of `input_log_det` is not expected'):
            _ = flow(x, T.random.randn([2, 4]))
        with pytest.raises(Exception,
                           match='The shape of `input_log_det` is not expected'):
            _ = flow(expected_y, T.random.randn([2, 4]), inverse=True)

        # test output_log_det shape error
        flow = T.jit_compile(_MyBadFlow())
        with pytest.raises(Exception,
                           match='The shape of `output_log_det` is not expected'):
            _ = flow(x)
        with pytest.raises(Exception,
                           match='The shape of `output_log_det` is not expected'):
            _ = flow(x, inverse=True)


class FeatureMappingFlowTestCase(unittest.TestCase):

    def test_constructor(self):
        flow = FeatureMappingFlow(axis=-1,
                                  event_ndims=2,
                                  explicitly_invertible=True)
        self.assertEqual(flow.get_event_ndims(), 2)
        self.assertEqual(flow.axis, -1)
        flow = T.jit_compile(flow)

        self.assertEqual(flow.get_axis(), -1)
        self.assertEqual(flow.get_x_event_ndims(), 2)
        self.assertEqual(flow.get_y_event_ndims(), 2)
        self.assertEqual(flow.is_explicitly_invertible(), True)

        with pytest.raises(ValueError,
                           match='`event_ndims` must be at least 1'):
            _ = FeatureMappingFlow(axis=-1, event_ndims=0, explicitly_invertible=True)

        with pytest.raises(ValueError,
                           match='`-event_ndims <= axis < 0` does not hold'):
            _ = FeatureMappingFlow(axis=-2, event_ndims=1, explicitly_invertible=True)

        with pytest.raises(ValueError,
                           match='`-event_ndims <= axis < 0` does not hold'):
            _ = FeatureMappingFlow(axis=0, event_ndims=1, explicitly_invertible=True)


class InverseFlowTestCase(unittest.TestCase):

    def test_InverseFlow(self):
        original_flow = T.jit_compile(_MyFlow())
        flow = InverseFlow(original_flow)
        self.assertIs(flow.original_flow, original_flow)
        self.assertIs(flow.invert(), original_flow)

        flow = T.jit_compile(flow)
        self.assertEqual(flow.get_x_event_ndims(), 2)
        self.assertEqual(flow.get_y_event_ndims(), 1)
        self.assertTrue(flow.is_explicitly_invertible())

        x = T.random.randn([2, 3, 4, 1])
        expected_y = T.reshape((x - 1.) * 0.5, [2, 3, 4])
        expected_log_det = -T.full([2, 3], math.log(2.) * 4)
        input_log_det = T.random.randn([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            input_log_det)

        with pytest.raises(TypeError,
                           match='`flow` must be an explicitly invertible flow'):
            _ = InverseFlow(tk.layers.Linear(5, 3))

        base_flow = _MyFlow()
        base_flow.explicitly_invertible = False
        with pytest.raises(TypeError,
                           match='`flow` must be an explicitly invertible flow'):
            _ = InverseFlow(T.jit_compile(base_flow))


class _MyFlow1(Flow):

    def __init__(self):
        super().__init__(x_event_ndims=1, y_event_ndims=1,
                         explicitly_invertible=True)

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = (input - 1.) * 0.5
        else:
            output = input * 2. + 1.

        output_log_det = input_log_det
        if compute_log_det:
            log_2 = T.as_tensor_backend(math.log(2.), dtype=output.dtype)
            if output_log_det is None:
                if inverse:
                    output_log_det = -log_2 * input.shape[-1]
                else:
                    output_log_det = log_2 * input.shape[-1]
            else:
                if inverse:
                    output_log_det = output_log_det - log_2 * input.shape[-1]
                else:
                    output_log_det = output_log_det + log_2 * input.shape[-1]

        return output, output_log_det


class SequentialFlowTestCase(unittest.TestCase):

    def test_constructor(self):
        flows = [T.jit_compile(_MyFlow1()), T.jit_compile(_MyFlow())]
        flow = T.jit_compile(SequentialFlow(flows))
        self.assertEqual(flow.get_x_event_ndims(), 1)
        self.assertEqual(flow.get_y_event_ndims(), 2)
        self.assertTrue(flow.is_explicitly_invertible())

        flow2 = _MyFlow()
        flow2.explicitly_invertible = False
        flows = [T.jit_compile(_MyFlow1()), T.jit_compile(flow2)]
        flow = T.jit_compile(SequentialFlow(flows))
        self.assertFalse(flow.is_explicitly_invertible())

        with pytest.raises(ValueError,
                           match='`flows` must not be empty'):
            _ = SequentialFlow([])

        with pytest.raises(TypeError,
                           match=r'`flows\[0\]` is not a flow'):
            _ = SequentialFlow([tk.layers.Linear(5, 3), _MyFlow()])

        with pytest.raises(TypeError,
                           match=r'`flows\[1\]` is not a flow'):
            _ = SequentialFlow([_MyFlow(), tk.layers.Linear(5, 3)])

        with pytest.raises(ValueError,
                           match=r'`x_event_ndims` of `flows\[1\]` != '
                                 r'`y_event_ndims` of `flows\[0\]`: '
                                 r'1 vs 2'):
            _ = SequentialFlow([_MyFlow(), _MyFlow()])

    def test_call(self):
        # test call and inverse call
        flows = [_MyFlow1(), T.jit_compile(_MyFlow1())]
        flow = T.jit_compile(SequentialFlow(flows))

        x = T.random.randn([2, 3, 4])
        expected_y = (x * 2. + 1.) * 2. + 1.
        expected_log_det = T.full([2, 3], math.log(2.) * 8)
        input_log_det = T.random.randn([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            input_log_det)

        # test no inverse call
        flows = [_MyFlow1()]
        flows[0].explicitly_invertible = False
        flow = T.jit_compile(SequentialFlow(flows))

        with pytest.raises(Exception,
                           match='Not an explicitly invertible flow'):
            _ = flow(x, inverse=True)


def check_invertible_matrix(ctx, m, size):
    matrix, log_det = m(inverse=False, compute_log_det=False)
    ctx.assertIsNone(log_det)

    matrix, log_det = m(inverse=False, compute_log_det=True)
    ctx.assertEqual(T.shape(matrix), [size, size])
    assert_allclose(T.matrix_inverse(T.matrix_inverse(matrix)),
                    matrix, rtol=1e-4, atol=1e-6)
    assert_allclose(T.linalg.slogdet(matrix)[1], log_det,
                    rtol=1e-4, atol=1e-6)

    inv_matrix, inv_log_det = m(inverse=True, compute_log_det=True)
    ctx.assertEqual(T.shape(inv_matrix), [size, size])
    assert_allclose(T.matrix_inverse(inv_matrix),
                    matrix, rtol=1e-4, atol=1e-6)
    assert_allclose(T.matrix_inverse(T.matrix_inverse(inv_matrix)),
                    inv_matrix, rtol=1e-4, atol=1e-6)
    assert_allclose(inv_log_det, -log_det, rtol=1e-4, atol=1e-6)
    assert_allclose(T.linalg.slogdet(inv_matrix)[1], -log_det,
                    rtol=1e-4, atol=1e-6)


class InvertibleMatrixTestCase(unittest.TestCase):

    def test_invertible_matrices(self):
        for cls in (LooseInvertibleMatrix, StrictInvertibleMatrix):
            for n in [1, 3, 5]:
                m = cls(np.random.randn(n, n))
                self.assertEqual(repr(m), f'{cls.__qualname__}(size={n})')
                self.assertEqual(m.size, n)

                m = T.jit_compile(m)

                # check the initial value is an orthogonal matrix
                matrix, _ = m(inverse=False, compute_log_det=False)
                inv_matrix, _ = m(inverse=True, compute_log_det=False)
                assert_allclose(np.eye(n), T.matmul(matrix, inv_matrix),
                                rtol=1e-4, atol=1e-6)
                assert_allclose(np.eye(n), T.matmul(inv_matrix, matrix),
                                rtol=1e-4, atol=1e-6)

                # check the invertibility
                check_invertible_matrix(self, m, n)

                # check the gradient
                matrix, log_det = m(inverse=False, compute_log_det=True)
                params = list(tk.layers.get_parameters(m))
                grads = T.grad(
                    [T.reduce_sum(matrix), T.reduce_sum(log_det)], params)

                # update with gradient, then check the invertibility
                if cls is StrictInvertibleMatrix:
                    for param, grad in zip(params, grads):
                        with T.no_grad():
                            T.assign(param, param + 0.001 * grad)
                    check_invertible_matrix(self, m, n)


def check_invertible_linear(ctx,
                            spatial_ndims: int,
                            invertible_linear_factory,
                            linear_factory,
                            strict: bool,):
    for batch_shape in ([2], [2, 3]):
        num_features = 4
        spatial_shape = [5, 6, 7][:spatial_ndims]
        x = T.random.randn(make_conv_shape(
            batch_shape, num_features, spatial_shape))

        # construct the layer
        flow = invertible_linear_factory(num_features, strict=strict)
        ctx.assertIn(f'num_features={num_features}', repr(flow))
        flow = T.jit_compile(flow)

        # derive the expected answer
        weight, log_det = flow.invertible_matrix(
            inverse=False, compute_log_det=True)
        linear_kwargs = {}
        if spatial_ndims > 0:
            linear_kwargs['kernel_size'] = 1
        linear = linear_factory(
            num_features, num_features,
            weight_init=T.reshape(weight, T.shape(weight) + [1] * spatial_ndims),
            use_bias=False,
            **linear_kwargs
        )
        x_flatten, front_shape = T.flatten_to_ndims(x, spatial_ndims + 2)
        expected_y = T.unflatten_from_ndims(linear(x_flatten), front_shape)
        expected_log_det = T.expand(
            T.reduce_sum(T.expand(log_det, spatial_shape)), batch_shape)

        # check the invertible layer
        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            T.random.randn(batch_shape))


class InvertibleLinearTestCase(unittest.TestCase):

    def test_invertible_dense(self):
        T.random.seed(1234)
        for strict in (True, False):
            check_invertible_linear(
                self,
                spatial_ndims=0,
                invertible_linear_factory=InvertibleDense,
                linear_factory=tk.layers.Linear,
                strict=strict
            )

    def test_invertible_conv_nd(self):
        T.random.seed(1234)
        for spatial_ndims in (1, 2, 3):
            for strict in (True, False):
                check_invertible_linear(
                    self,
                    spatial_ndims=spatial_ndims,
                    invertible_linear_factory=getattr(
                        tk.flows, f'InvertibleConv{spatial_ndims}d'),
                    linear_factory=getattr(
                        tk.layers, f'LinearConv{spatial_ndims}d'),
                    strict=strict
                )


def check_scale(ctx,
                scale: Scale,
                x,
                pre_scale,
                expected_y,
                expected_log_det):
    assert(T.shape(x) == T.shape(expected_log_det))

    # dimension error
    with pytest.raises(Exception,
                       match=r'`rank\(input\) >= event_ndims` does not hold'):
        _ = scale(T.random.randn([1]), T.random.randn([1]), event_ndims=2)

    with pytest.raises(Exception,
                       match=r'`rank\(input\) >= rank\(pre_scale\)` does not hold'):
        _ = scale(T.random.randn([1]), T.random.randn([1, 2]), event_ndims=1)

    with pytest.raises(Exception,
                       match=r'The shape of `input_log_det` is not expected'):
        _ = scale(T.random.randn([2, 3]),
                  T.random.randn([2, 3]),
                  event_ndims=1,
                  input_log_det=T.random.randn([3]))

    with pytest.raises(Exception,
                       match=r'The shape of `input_log_det` is not expected'):
        _ = scale(T.random.randn([2, 3]),
                  T.random.randn([2, 3]),
                  event_ndims=2,
                  input_log_det=T.random.randn([2]))

    # check call
    for event_ndims in range(T.rank(pre_scale), T.rank(x)):
        this_expected_log_det = T.reduce_sum(
            expected_log_det, axis=T.int_range(-event_ndims, 0))
        input_log_det = T.random.randn(T.shape(this_expected_log_det))

        # check no compute log_det
        y, log_det = scale(x, pre_scale, event_ndims=event_ndims,
                           compute_log_det=False)
        assert_allclose(y, expected_y, rtol=1e-4, atol=1e-6)
        ctx.assertIsNone(log_det)

        # check compute log_det
        y, log_det = scale(x, pre_scale, event_ndims=event_ndims)
        assert_allclose(y, expected_y, rtol=1e-4, atol=1e-6)
        assert_allclose(log_det, this_expected_log_det, rtol=1e-4, atol=1e-6)

        # check compute log_det with input_log_det
        y, log_det = scale(
            x, pre_scale, event_ndims=event_ndims, input_log_det=input_log_det)
        assert_allclose(y, expected_y, rtol=1e-4, atol=1e-6)
        assert_allclose(log_det, input_log_det + this_expected_log_det,
                        rtol=1e-4, atol=1e-6)

        # check inverse, no compute log_det
        inv_x, log_det = scale(expected_y, pre_scale, event_ndims=event_ndims,
                               inverse=True, compute_log_det=False)
        assert_allclose(inv_x, x, rtol=1e-4, atol=1e-6)
        ctx.assertIsNone(log_det)

        # check inverse, compute log_det
        inv_x, log_det = scale(expected_y, pre_scale, event_ndims=event_ndims,
                               inverse=True)
        assert_allclose(inv_x, x, rtol=1e-4, atol=1e-6)
        assert_allclose(log_det, -this_expected_log_det, rtol=1e-4, atol=1e-6)

        # check inverse, compute log_det with input_log_det
        inv_x, log_det = scale(expected_y, pre_scale, event_ndims=event_ndims,
                               inverse=True, input_log_det=input_log_det)
        assert_allclose(inv_x, x, rtol=1e-4, atol=1e-6)
        assert_allclose(log_det, input_log_det - this_expected_log_det,
                        rtol=1e-4, atol=1e-6)


class _BadScale1(Scale):

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        scale = pre_scale
        if compute_log_scale:
            log_scale: Optional[Tensor] = randn([2, 3, 4])
        else:
            log_scale: Optional[Tensor] = None
        return scale, log_scale


class _BadScale2(Scale):

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        scale = pre_scale
        if compute_log_scale:
            log_scale: Optional[Tensor] = randn([1] + shape(pre_scale))
        else:
            log_scale: Optional[Tensor] = None
        return scale, log_scale


class ScaleTestCase(unittest.TestCase):

    def test_ExpScale(self):
        T.random.seed(1234)

        x = T.random.randn([2, 3, 4])
        scale = ExpScale()
        scale = T.jit_compile(scale)

        for pre_scale in [T.random.randn([4]),
                          T.random.randn([3, 1]),
                          T.random.randn([2, 1, 1]),
                          T.random.randn([2, 3, 4])]:
            expected_y = x * T.exp(pre_scale)
            expected_log_det = T.broadcast_to(pre_scale, T.shape(x))
            check_scale(self, scale, x, pre_scale, expected_y, expected_log_det)

    def test_SigmoidScale(self):
        T.random.seed(1234)

        x = T.random.randn([2, 3, 4])

        for pre_scale_bias in [None, 0., 1.5]:
            scale = SigmoidScale(**(
                {'pre_scale_bias': pre_scale_bias}
                if pre_scale_bias is not None else {}
            ))
            if pre_scale_bias is None:
                pre_scale_bias = 0.
            self.assertIn(f'pre_scale_bias={pre_scale_bias}', repr(scale))
            scale = T.jit_compile(scale)

            for pre_scale in [T.random.randn([4]),
                              T.random.randn([3, 1]),
                              T.random.randn([2, 1, 1]),
                              T.random.randn([2, 3, 4])]:
                expected_y = x * T.nn.sigmoid(pre_scale + pre_scale_bias)
                expected_log_det = T.broadcast_to(
                    T.nn.log_sigmoid(pre_scale + pre_scale_bias), T.shape(x))
                check_scale(self, scale, x, pre_scale, expected_y, expected_log_det)

    def test_LinearScale(self):
        T.random.seed(1234)

        x = T.random.randn([2, 3, 4])
        scale = LinearScale(epsilon=T.EPSILON)
        self.assertIn('epsilon=', repr(scale))
        scale = T.jit_compile(scale)

        for pre_scale in [T.random.randn([4]),
                          T.random.randn([3, 1]),
                          T.random.randn([2, 1, 1]),
                          T.random.randn([2, 3, 4])]:
            expected_y = x * pre_scale
            expected_log_det = T.broadcast_to(
                T.log(T.abs(pre_scale)), T.shape(x))
            check_scale(self, scale, x, pre_scale, expected_y, expected_log_det)

    def test_bad_output(self):
        T.random.seed(1234)
        x = T.random.randn([2, 3, 1])

        scale = _BadScale1()
        with pytest.raises(Exception,
                           match='The shape of the final 1d of `log_scale` is '
                                 'not expected'):
            _ = scale(x, x, event_ndims=1)

        scale = _BadScale2()
        with pytest.raises(Exception, match='shape'):
            _ = scale(x, x, event_ndims=0)
        with pytest.raises(Exception,
                           match='The shape of the computed `output_log_det` '
                                 'is not expected'):
            _ = scale(x, x, event_ndims=0, input_log_det=x)
