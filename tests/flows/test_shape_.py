import unittest
from itertools import product

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.flows import *
from tests.helper import *
from tests.ops import *


class ReshapeFlowTestCase(unittest.TestCase):

    def test_ReshapeFlow(self):
        flow = ReshapeFlow([4, -1], [-1])
        self.assertEqual(flow.x_event_shape, [4, -1])
        self.assertEqual(flow.y_event_shape, [-1])
        self.assertEqual(flow.get_x_event_ndims(), 2)
        self.assertEqual(flow.get_y_event_ndims(), 1)
        self.assertIn('x_event_shape=[4, -1]', repr(flow))
        self.assertIn('y_event_shape=[-1]', repr(flow))
        flow = tk.layers.jit_compile(flow)

        x = T.random.randn([2, 3, 4, 5])
        expected_y = T.reshape_tail(x, 2, [-1])
        expected_log_det = T.zeros([2, 3])

        flow_standard_check(self, flow, x, expected_y, expected_log_det,
                            T.random.randn([2, 3]))

        with pytest.raises(ValueError,
                           match='Too many `-1` specified in `x_event_shape`'):
            _ = ReshapeFlow([-1, -1], [-1])

        with pytest.raises(ValueError,
                           match='All elements of `x_event_shape` must be '
                                 'positive integers or `-1`'):
            _ = ReshapeFlow([-1, -2], [-1])

        with pytest.raises(ValueError,
                           match='Too many `-1` specified in `y_event_shape`'):
            _ = ReshapeFlow([-1], [-1, -1])

        with pytest.raises(ValueError,
                           match='All elements of `y_event_shape` must be '
                                 'positive integers or `-1`'):
            _ = ReshapeFlow([-1], [-1, -2])


class SpaceDepthTransformFlowTestCase(unittest.TestCase):

    def test_space_depth_transform(self):
        T.random.seed(1234)

        for spatial_ndims, batch_shape, block_size in product(
                    (1, 2, 3),
                    ([2], [2, 3]),
                    (1, 2, 4),
                ):
            # prepare for the data
            n_channels = 5
            x = T.random.randn(make_conv_shape(
                batch_shape, n_channels, [4, 8, 12][:spatial_ndims]))
            y = getattr(T.nn, f'space_to_depth{spatial_ndims}d')(x, block_size)
            log_det = T.zeros(batch_shape)
            input_log_det = T.random.randn(batch_shape)

            # construct the classes
            cls = getattr(tk.flows, f'SpaceToDepth{spatial_ndims}d')
            inv_cls = getattr(tk.flows, f'DepthToSpace{spatial_ndims}d')

            flow = cls(block_size)
            self.assertEqual(flow.block_size, block_size)
            inv_flow = inv_cls(block_size)
            self.assertEqual(inv_flow.block_size, block_size)

            self.assertIsInstance(flow.invert(), inv_cls)
            self.assertEqual(flow.invert().block_size, block_size)
            self.assertIsInstance(inv_flow.invert(), cls)
            self.assertEqual(inv_flow.invert().block_size, block_size)

            # check call
            flow_standard_check(self, flow, x, y, log_det, input_log_det)
            flow_standard_check(self, inv_flow, y, x, log_det, input_log_det)

            # test error
            with pytest.raises(ValueError,
                               match='`block_size` must be at least 1'):
                _ = cls(0)

            with pytest.raises(ValueError,
                               match='`block_size` must be at least 1'):
                _ = inv_cls(0)
