import unittest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.flows import *
from tests.helper import *
from tests.ops import *


def check_shuffling_flow(ctx,
                         spatial_ndims: int,
                         cls):
    num_features = 5

    for batch_shape in ([2], [2, 3]):
        shape = make_conv_shape(
            batch_shape, num_features, [6, 7, 8][: spatial_ndims])

        # test constructor
        flow = cls(num_features)
        ctx.assertIn(f'num_features={num_features}', repr(flow))
        permutation = tk.layers.get_parameter(flow, 'permutation')
        inv_permutation = tk.layers.get_parameter(flow, 'inv_permutation')
        assert_equal(T.argsort(permutation), inv_permutation)
        assert_equal(T.argsort(inv_permutation), permutation)
        flow = tk.layers.jit_compile(flow)

        # prepare for the answer
        x = T.random.randn(shape)
        channel_axis = get_channel_axis(spatial_ndims)
        expected_y = T.index_select(x, permutation, axis=channel_axis)
        assert_equal(
            T.index_select(expected_y, inv_permutation, axis=channel_axis),
            x,
        )
        expected_log_det = T.zeros(batch_shape)

        # check the flow
        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            T.random.randn(batch_shape))


class RearrangementTestCase(TestCase):

    def test_FeatureShuffleFlow(self):
        check_shuffling_flow(self, 0, FeatureShufflingFlow)

    def test_FeatureShuffleFlowNd(self):
        for spatial_ndims in (1, 2, 3):
            check_shuffling_flow(
                self,
                spatial_ndims,
                getattr(tk.flows, f'FeatureShufflingFlow{spatial_ndims}d'),
            )
