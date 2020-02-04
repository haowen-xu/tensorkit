import functools
import unittest

import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit.variational import *
from tests.helper import assert_allclose


def prepare_test_payload(reparameterized):
    np.random.seed(1234)
    x = T.as_tensor(np.random.normal(size=[7, 13]))  # input
    y = T.requires_grad(T.as_tensor(np.random.normal(size=[13])))  # param
    if reparameterized:
        z = y * x  # sample
    else:
        z = T.stop_grad(y) * x
    f = T.exp(y * z)
    log_f = y * z
    log_q = (x ** 2 - 1) * (y ** 3)
    return x, y, z, f, log_f, log_q


class SGVBEstimatorTestCase(unittest.TestCase):

    def test_sgvb(self):
        assert_allclose_ = functools.partial(assert_allclose, rtol=1e-5, atol=1e-6)

        # default
        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        cost = sgvb_estimator(f)
        assert_allclose_(-cost, sgvb_estimator(f, negative=True))
        cost_shape = T.shape(cost)
        assert_allclose_(
            T.grad([T.reduce_sum(cost)], [y])[0],
            T.reduce_sum(2 * x * y * f, axis=[0])
        )

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        cost_r = sgvb_estimator(f, axis=[0])
        assert_allclose_(-cost_r, sgvb_estimator(f, axis=[0], negative=True))
        self.assertListEqual(cost_shape[1:], T.shape(cost_r))
        assert_allclose_(
            T.grad([T.reduce_sum(cost_r)], [y])[0],
            T.reduce_sum(2 * x * y * f, axis=[0]) / 7
        )

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        cost_rk = sgvb_estimator(f, axis=[0], keepdims=True)
        assert_allclose_(
            -cost_rk,
            sgvb_estimator(f, axis=[0], keepdims=True, negative=True))
        self.assertListEqual([1] + cost_shape[1:], T.shape(cost_rk))
        assert_allclose_(
            T.grad([T.reduce_sum(cost_rk)], [y])[0],
            T.reduce_sum(2 * x * y * f, axis=[0]) / 7
        )


class IWAEEstimatorTestCase(unittest.TestCase):

    def test_error(self):
        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        with pytest.raises(Exception,
                           match='`iwae_estimator` requires to take multiple '
                                 'samples'):
            _ = iwae_estimator(log_f, axis=None)
        with pytest.raises(Exception,
                           match='`iwae_estimator` requires to take multiple '
                                 'samples'):
            _ = iwae_estimator(log_f, axis=[])

    def test_iwae(self):
        assert_allclose_ = functools.partial(assert_allclose, rtol=1e-5, atol=1e-6)

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        wk_hat = f / T.reduce_sum(f, axis=[0], keepdims=True)
        cost = iwae_estimator(log_f, axis=[0])
        assert_allclose_(-cost, iwae_estimator(log_f, axis=[0], negative=True))
        cost_shape = T.shape(cost)
        assert_allclose_(
            T.grad([T.reduce_sum(cost)], [y])[0],
            T.reduce_sum(wk_hat * (2 * x * y), axis=[0])
        )

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        wk_hat = f / T.reduce_sum(f, axis=[0], keepdims=True)
        cost_k = iwae_estimator(log_f, axis=[0], keepdims=True)
        assert_allclose_(
            -cost_k,
            T.to_numpy(iwae_estimator(log_f, axis=[0], keepdims=True,
                                      negative=True)))
        self.assertListEqual([1] + cost_shape, T.shape(cost_k))
        assert_allclose_(
            T.grad([T.reduce_sum(cost_k)], [y])[0],
            T.reduce_sum(wk_hat * (2 * x * y), axis=[0])
        )

