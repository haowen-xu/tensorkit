import unittest
from functools import partial

import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit.variational import *
from tests.helper import *


def prepare_test_payload():
    log_p = T.as_tensor(np.random.normal(size=[13]))
    log_q = T.as_tensor(np.random.normal(size=[7, 13]))
    return log_p, log_q


assert_allclose_ = partial(assert_allclose, atol=1e-4)


class ImportanceSamplingLogLikelihoodTestCase(TestCase):

    def test_error(self):
        log_p, log_q = prepare_test_payload()
        with pytest.raises(Exception,
                           match='`importance_sampling_log_likelihood` requires '
                                 'to take[^@]*multiple samples'):
            _ = importance_sampling_log_likelihood(log_p, log_q, axis=None)
        with pytest.raises(Exception,
                           match='`importance_sampling_log_likelihood` requires '
                                 'to take[^@]*multiple samples'):
            _ = importance_sampling_log_likelihood(log_p, log_q, axis=[])

    def test_monto_carlo_objective(self):
        log_p, log_q = prepare_test_payload()

        ll = importance_sampling_log_likelihood(log_p, log_q, axis=[0])
        ll_shape = T.shape(ll)
        assert_allclose_(ll, T.log_mean_exp(log_p - log_q, axis=[0]))
        assert_allclose_(
            T.reduce_mean(ll),
            importance_sampling_log_likelihood(log_p, log_q, axis=[0], reduction='mean')
        )
        assert_allclose_(
            T.reduce_sum(ll),
            importance_sampling_log_likelihood(log_p, log_q, axis=[0], reduction='sum')
        )

        ll_k = importance_sampling_log_likelihood(
            log_p, log_q, axis=[0], keepdims=True)
        self.assertListEqual([1] + ll_shape, T.shape(ll_k))
        assert_allclose_(
            ll_k,
            T.log_mean_exp(log_p - log_q, axis=[0], keepdims=True)
        )
