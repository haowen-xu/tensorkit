import functools
import unittest

import numpy as np
import pytest

from tensorkit import backend as Z
from tensorkit import *


def prepare_test_payload(reparameterized):
    np.random.seed(1234)
    x = Z.from_numpy(np.random.normal(size=[7, 13]))  # input
    y = Z.requires_grad(Z.from_numpy(np.random.normal(size=[13])))  # param
    if reparameterized:
        z = y * x  # sample
    else:
        z = Z.stop_grad(y) * x
    f = Z.exp(y * z)
    log_f = y * z
    log_q = (x ** 2 - 1) * (y ** 3)
    return x, y, z, f, log_f, log_q


class SGVBEstimatorTestCase(unittest.TestCase):

    def test_sgvb(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        # default
        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        cost = sgvb_estimator(f)
        np.testing.assert_allclose(
            Z.to_numpy(-cost),
            Z.to_numpy(sgvb_estimator(f, negative=True)))
        cost_shape = Z.shape(cost)
        assert_allclose(
            Z.to_numpy(Z.grad([Z.reduce_sum(cost)], [y])[0]),
            Z.to_numpy(Z.reduce_sum(2 * x * y * f, axes=[0]))
        )

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        cost_r = sgvb_estimator(f, axes=[0])
        np.testing.assert_allclose(
            Z.to_numpy(-cost_r),
            Z.to_numpy(sgvb_estimator(f, axes=[0], negative=True)))
        self.assertListEqual(cost_shape[1:], Z.shape(cost_r))
        assert_allclose(
            Z.to_numpy(Z.grad([Z.reduce_sum(cost_r)], [y])[0]),
            Z.to_numpy(Z.reduce_sum(2 * x * y * f, axes=[0]) / 7)
        )

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        cost_rk = sgvb_estimator(f, axes=[0], keepdims=True)
        np.testing.assert_allclose(
            Z.to_numpy(-cost_rk),
            Z.to_numpy(sgvb_estimator(f, axes=[0], keepdims=True,
                                      negative=True)))
        self.assertListEqual([1] + cost_shape[1:], Z.shape(cost_rk))
        assert_allclose(
            Z.to_numpy(Z.grad([Z.reduce_sum(cost_rk)], [y])[0]),
            Z.to_numpy(Z.reduce_sum(2 * x * y * f, axes=[0]) / 7)
        )


class IWAEEstimatorTestCase(unittest.TestCase):

    def test_error(self):
        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        with pytest.raises(Exception,
                           match='`iwae_estimator` requires to take multiple '
                                 'samples'):
            _ = iwae_estimator(log_f, axes=None)
        with pytest.raises(Exception,
                           match='`iwae_estimator` requires to take multiple '
                                 'samples'):
            _ = iwae_estimator(log_f, axes=[])

    def test_iwae(self):
        assert_allclose = functools.partial(
            np.testing.assert_allclose, rtol=1e-5, atol=1e-6)

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        wk_hat = f / Z.reduce_sum(f, axes=[0], keepdims=True)
        cost = iwae_estimator(log_f, axes=[0])
        np.testing.assert_allclose(
            Z.to_numpy(-cost),
            Z.to_numpy(iwae_estimator(log_f, axes=[0], negative=True)))
        cost_shape = Z.shape(cost)
        assert_allclose(
            Z.to_numpy(Z.grad([Z.reduce_sum(cost)], [y])[0]),
            Z.to_numpy(Z.reduce_sum(wk_hat * (2 * x * y), axes=[0]))
        )

        x, y, z, f, log_f, log_q = prepare_test_payload(reparameterized=True)
        wk_hat = f / Z.reduce_sum(f, axes=[0], keepdims=True)
        cost_k = iwae_estimator(log_f, axes=[0], keepdims=True)
        np.testing.assert_allclose(
            Z.to_numpy(-cost_k),
            Z.to_numpy(iwae_estimator(log_f, axes=[0], keepdims=True,
                                      negative=True)))
        self.assertListEqual([1] + cost_shape, Z.shape(cost_k))
        assert_allclose(
            Z.to_numpy(Z.grad([Z.reduce_sum(cost_k)], [y])[0]),
            Z.to_numpy(Z.reduce_sum(wk_hat * (2 * x * y), axes=[0]))
        )

