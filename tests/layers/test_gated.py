import unittest

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class GatedTestCase(unittest.TestCase):

    def test_Gated(self):
        gated = tk.layers.Gated(feature_axis=-2, num_features=3,
                                gate_bias=1.5)
        self.assertIn(
            'feature_axis=-2, num_features=3, gate_bias=1.5',
            repr(gated)
        )
        gated = T.jit_compile(gated)

        x = T.random.randn([6, 5])
        assert_allclose(gated(x), x[:3, ...] * T.nn.sigmoid(x[3:, ...] + 1.5))

        x = T.random.randn([3, 6, 5])
        assert_allclose(gated(x), x[:, :3, ...] * T.nn.sigmoid(x[:, 3:, ...] + 1.5))

        with pytest.raises(Exception,
                           match='shape of the pre-gated output is invalid'):
            _ = gated(T.random.randn([7, 3]))

    def test_GatedWithActivation(self):
        gated = tk.layers.GatedWithActivation(
            feature_axis=-2, num_features=3, gate_bias=1.5,
            activation=tk.layers.LeakyReLU(),
        )
        self.assertIn(
            'feature_axis=-2, num_features=3, gate_bias=1.5',
            repr(gated)
        )
        gated = T.jit_compile(gated)

        x = T.random.randn([6, 5])
        assert_allclose(
            gated(x),
            T.nn.leaky_relu(x[:3, ...]) * T.nn.sigmoid(x[3:, ...] + 1.5)
        )

        x = T.random.randn([3, 6, 5])
        assert_allclose(
            gated(x),
            T.nn.leaky_relu(x[:, :3, ...]) * T.nn.sigmoid(x[:, 3:, ...] + 1.5)
        )

        with pytest.raises(Exception,
                           match='shape of the pre-gated output is invalid'):
            _ = gated(T.random.randn([7, 3]))
