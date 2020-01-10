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


class ImportanceSamplingLogLikelihoodTestCase(unittest.TestCase):

    def test_error(self):
        log_p, log_q = prepare_test_payload()
        with pytest.raises(Exception,
                           match='`importance_sampling_log_likelihood` requires '
                                 'to take[^@]*multiple samples'):
            _ = importance_sampling_log_likelihood(log_p, log_q, axes=None)
        with pytest.raises(Exception,
                           match='`importance_sampling_log_likelihood` requires '
                                 'to take[^@]*multiple samples'):
            _ = importance_sampling_log_likelihood(log_p, log_q, axes=[])

    def test_monto_carlo_objective(self):
        log_p, log_q = prepare_test_payload()

        ll = importance_sampling_log_likelihood(log_p, log_q, axes=[0])
        ll_shape = Z.shape(ll)
        assert_allclose(
            Z.to_numpy(ll),
            Z.to_numpy(Z.log_mean_exp(log_p - log_q, axes=[0]))
        )

        ll_k = importance_sampling_log_likelihood(
            log_p, log_q, axes=[0], keepdims=True)
        self.assertListEqual([1] + ll_shape, Z.shape(ll_k))
        assert_allclose(
            Z.to_numpy(ll_k),
            Z.to_numpy(Z.log_mean_exp(log_p - log_q, axes=[0], keepdims=True))
        )
