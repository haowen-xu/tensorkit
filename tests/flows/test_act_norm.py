import unittest
from itertools import product

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.flows import *
from tests.helper import *
from tests.ops import *


def check_act_norm(ctx, spatial_ndims: int, cls):
    num_features = 4
    channel_axis = get_channel_axis(spatial_ndims)

    def do_check(batch_shape, scale_type, initialized, dtype):
        x = T.random.randn(make_conv_shape(
            batch_shape, num_features, [6, 7, 8][: spatial_ndims]), dtype=dtype)

        # check construct
        flow = cls(num_features, scale=scale_type, initialized=initialized,
                   dtype=dtype)
        ctx.assertIn(f'num_features={num_features}', repr(flow))
        ctx.assertIn(f'axis={-(spatial_ndims + 1)}', repr(flow))
        ctx.assertIn(f'scale_type={scale_type!r}', repr(flow))
        flow = tk.layers.jit_compile(flow)

        # check initialize
        if not initialized:
            # must initialize with sufficient data
            with pytest.raises(Exception,
                               match='at least .* dimensions'):
                _ = flow(T.random.randn(
                    make_conv_shape([], num_features, [6, 7, 8][: spatial_ndims]),
                    dtype=dtype
                ))

            # must initialize with inverse = Fale
            with pytest.raises(Exception,
                               match='`ActNorm` must be initialized with '
                                     '`inverse = False`'):
                _ = flow(x, inverse=True)

            # do initialize
            y, _ = flow(x, compute_log_det=False)
            y_mean, y_var = T.calculate_mean_and_var(
                y,
                axis=[a for a in range(-T.rank(y), 0) if a != channel_axis]
            )
            assert_allclose(y_mean, T.zeros([num_features]), rtol=1e-4, atol=1e-6)
            assert_allclose(y_var, T.ones([num_features]), rtol=1e-4, atol=1e-6)
        else:
            y, _ = flow(x, compute_log_det=False)
            assert_allclose(y, x, rtol=1e-4, atol=1e-6)

        # prepare for the expected result
        scale_obj = ExpScale() if scale_type == 'exp' else LinearScale()

        if T.IS_CHANNEL_LAST:
            aligned_shape = [num_features]
        else:
            aligned_shape = [num_features] + [1] * spatial_ndims
        bias = T.reshape(flow.bias, aligned_shape)
        pre_scale = T.reshape(flow.pre_scale, aligned_shape)

        expected_y, expected_log_det = scale_obj(
            x + bias, pre_scale, event_ndims=(spatial_ndims + 1), compute_log_det=True)

        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            T.random.randn(T.shape(expected_log_det)))

    for batch_shape in ([11], [11, 12]):
        do_check(batch_shape, 'exp', False, T.float32)

    for scale_type in ('exp', 'linear'):
        do_check([11], scale_type, False, T.float32)

    for initialized in (True, False):
        do_check([11], 'exp', initialized, T.float32)

    for dtype in float_dtypes:
        do_check([11], 'exp', False, dtype)


class ActNormTestCase(unittest.TestCase):

    @slow_test
    def test_ActNorm(self):
        T.random.seed(1234)
        check_act_norm(self, 0, ActNorm)

    @slow_test
    def test_ActNormNd(self):
        T.random.seed(1234)
        for spatial_ndims in (1, 2, 3):
            check_act_norm(
                self,
                spatial_ndims,
                getattr(tk.flows, f'ActNorm{spatial_ndims}d')
            )
