import unittest

import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *


class VariationalInferenceTestCase(unittest.TestCase):

    def test_construction(self):
        vi = VariationalInference(T.float_scalar(1.),
                                  T.float_scalar(2.))
        self.assertIsNone(vi.axes)
        self.assertIsInstance(vi.training, VariationalTrainingObjectives)
        self.assertIsInstance(vi.lower_bound, VariationalLowerBounds)
        self.assertIsInstance(vi.evaluation, VariationalEvaluation)

        np.testing.assert_equal(T.to_numpy(vi.log_joint), 1.)
        np.testing.assert_equal(T.to_numpy(vi.latent_log_joint), 2.)

    def test_errors(self):
        # test no sampling axis should cause errors
        vi = VariationalInference(
            T.float_scalar(0.), T.float_scalar(0.), axes=None)
        with pytest.raises(
                Exception, match='`monte_carlo_objective` requires to take '
                                 'multiple samples'):
            _ = vi.lower_bound.monte_carlo_objective()
        with pytest.raises(
                Exception, match='`iwae_estimator` requires to take multiple '
                                 'samples'):
            _ = vi.training.iwae()
        with pytest.raises(
                Exception, match='`importance_sampling_log_likelihood` '
                                 'requires to take[^@]*multiple samples'):
            _ = vi.evaluation.importance_sampling_log_likelihood()
        # with pytest.raises(
        #         Exception, match='vimco training objective requires '
        #                          'multi-samples'):
        #     _ = vi.training.vimco()

    def test_elbo(self):
        log_p = T.random.randn(shape=[5, 7])
        log_q = T.random.randn(shape=[1, 3, 5, 7])

        # test without sampling axis
        vi = VariationalInference(log_p, log_q)
        output = vi.lower_bound.elbo()
        answer = elbo_objective(log_p, log_q)
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))

        # test with sampling axis
        vi = VariationalInference(log_p, log_q, axes=[0, 1])
        output = vi.lower_bound.elbo()
        answer = elbo_objective(log_p, log_q, axes=[0, 1])
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))

    def test_importance_weighted_objective(self):
        log_p = T.random.randn(shape=[5, 7])
        log_q = T.random.randn(shape=[1, 3, 5, 7])

        vi = VariationalInference(log_p, log_q, axes=[0, 1])
        output = vi.lower_bound.monte_carlo_objective()
        answer = monte_carlo_objective(log_p, log_q, axes=[0, 1])
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))

    def test_sgvb(self):
        log_p = T.random.randn(shape=[5, 7])
        log_q = T.random.randn(shape=[1, 3, 5, 7])

        # test without sampling axis
        vi = VariationalInference(log_p, log_q)
        output = vi.training.sgvb()
        answer = -sgvb_estimator(log_p - log_q)
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))

        # test with sampling axis
        vi = VariationalInference(log_p, log_q, axes=[0, 1])
        output = vi.training.sgvb()
        answer = -sgvb_estimator(log_p - log_q, axes=[0, 1])
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))

    def test_iwae(self):
        log_p = T.random.randn(shape=[5, 7])
        log_q = T.random.randn(shape=[1, 3, 5, 7])

        vi = VariationalInference(log_p, log_q, axes=[0, 1])
        output = vi.training.iwae()
        answer = -iwae_estimator(log_p - log_q, axes=[0, 1])
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))

    def test_is_loglikelihood(self):
        log_p = T.random.randn(shape=[5, 7])
        log_q = T.random.randn(shape=[1, 3, 5, 7])

        vi = VariationalInference(log_p, log_q, axes=[0, 1])
        output = vi.evaluation.importance_sampling_log_likelihood()
        answer = importance_sampling_log_likelihood(log_p, log_q, axes=[0, 1])
        np.testing.assert_allclose(T.to_numpy(output), T.to_numpy(answer))
