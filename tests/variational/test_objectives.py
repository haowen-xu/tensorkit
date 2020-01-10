import unittest

import numpy as np
import pytest

from tensorkit import backend as Z
from tensorkit import *


def prepare_test_payload():
    np.random.seed(1234)
    log_p = Z.from_numpy(np.random.normal(size=[13]))
    log_q = Z.from_numpy(np.random.normal(size=[7, 13]))
    return log_p, log_q


def assert_allclose(a, b):
    np.testing.assert_allclose(a, b, atol=1e-4)


class ELBOObjectiveTestCase(unittest.TestCase):

    def test_elbo(self):
        log_p, log_q = prepare_test_payload()

        obj = elbo_objective(log_p, log_q)
        obj_shape = Z.shape(obj)
        assert_allclose(Z.to_numpy(obj), Z.to_numpy(log_p - log_q))

        obj_r = elbo_objective(log_p, log_q, axes=[0])
        self.assertListEqual(obj_shape[1:], Z.shape(obj_r))
        assert_allclose(
            Z.to_numpy(obj_r),
            Z.to_numpy(Z.reduce_mean(log_p - log_q, axes=[0]))
        )

        obj_rk = elbo_objective(log_p, log_q, axes=[0], keepdims=True)
        self.assertListEqual([1] + obj_shape[1:], Z.shape(obj_rk))
        assert_allclose(
            Z.to_numpy(obj_rk),
            Z.to_numpy(Z.reduce_mean(log_p - log_q, axes=[0], keepdims=True))
        )


class MonteCarloObjectiveTestCase(unittest.TestCase):

    def test_error(self):
        log_p, log_q = prepare_test_payload()
        with pytest.raises(Exception,
                           match='`monte_carlo_objective` requires to take '
                                 'multiple samples'):
            _ = monte_carlo_objective(log_p, log_q, axes=None)
        with pytest.raises(Exception,
                           match='`monte_carlo_objective` requires to take '
                                 'multiple samples'):
            _ = monte_carlo_objective(log_p, log_q, axes=[])

    def test_monto_carlo_objective(self):
        log_p, log_q = prepare_test_payload()

        obj = monte_carlo_objective(log_p, log_q, axes=[0])
        obj_shape = Z.shape(obj)
        assert_allclose(
            Z.to_numpy(obj),
            Z.to_numpy(Z.log_mean_exp(log_p - log_q, axes=[0]))
        )

        obj_k = monte_carlo_objective(log_p, log_q, axes=[0], keepdims=True)
        self.assertListEqual([1] + obj_shape, Z.shape(obj_k))
        assert_allclose(
            Z.to_numpy(obj_k),
            Z.to_numpy(Z.log_mean_exp(log_p - log_q, axes=[0], keepdims=True))
        )
