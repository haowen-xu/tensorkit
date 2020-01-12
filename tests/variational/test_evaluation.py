import unittest

import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *


def prepare_test_payload():
    np.random.seed(1234)
    log_p = T.as_tensor(np.random.normal(size=[13]))
    log_q = T.as_tensor(np.random.normal(size=[7, 13]))
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
        ll_shape = T.shape(ll)
        assert_allclose(
            T.to_numpy(ll),
            T.to_numpy(T.log_mean_exp(log_p - log_q, axes=[0]))
        )

        ll_k = importance_sampling_log_likelihood(
            log_p, log_q, axes=[0], keepdims=True)
        self.assertListEqual([1] + ll_shape, T.shape(ll_k))
        assert_allclose(
            T.to_numpy(ll_k),
            T.to_numpy(T.log_mean_exp(log_p - log_q, axes=[0], keepdims=True))
        )
