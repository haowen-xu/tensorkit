import unittest
from typing import Optional, Tuple

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.tensor import Tensor
from tests.helper import *
from tests.ops import make_conv_shape


class _MyFlow(tk.flows.Flow):

    @T.jit_method
    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            raise RuntimeError('Not invertible.')
        output = input * 2.
        if compute_log_det or input_log_det is not None:
            raise RuntimeError('Should not compute log-det.')
        return output, input_log_det


class FlowLayerTestCase(unittest.TestCase):

    def test_FlowLayer(self):
        flow = T.jit_compile(_MyFlow(
            x_event_ndims=0, y_event_ndims=0, explicitly_invertible=True))
        layer = T.jit_compile(tk.layers.FlowLayer(flow))

        x = T.random.randn([3, 4, 5])
        assert_allclose(layer(x), x * 2.)

        with pytest.raises(TypeError, match='`flow` must be a flow'):
            _ = tk.layers.FlowLayer(object())


class ActNormLayerTestCase(unittest.TestCase):

    def test_ActNorm(self):
        layer = tk.layers.ActNorm(5)
        flow = layer.flow
        self.assertIsInstance(flow, tk.flows.ActNorm)
        self.assertEqual(flow.num_features, 5)

        # initialize the actnorm
        _ = layer(T.random.randn([3, 4, 5]))

        # check call
        layer = T.jit_compile(layer)
        x = T.random.randn([3, 4, 5])
        assert_allclose(layer(x), flow(x)[0])

    def test_ActNormNd(self):
        for spatial_ndims in (1, 2, 3):
            layer = getattr(tk.layers, f'ActNorm{spatial_ndims}d')(5)
            flow = layer.flow
            self.assertIsInstance(
                flow,
                getattr(tk.flows, f'ActNorm{spatial_ndims}d')
            )
            self.assertEqual(flow.num_features, 5)

            # initialize the actnorm
            shape = make_conv_shape([2], 5, [6, 7, 8][:spatial_ndims])
            _ = layer(T.random.randn(shape))

            # check call
            layer = T.jit_compile(layer)
            x = T.random.randn(shape)
            assert_allclose(layer(x), flow(x)[0])
