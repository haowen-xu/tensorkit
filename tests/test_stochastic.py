import unittest

import numpy as np

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tests.helper import *


class StochasticTensorTestCase(unittest.TestCase):

    def test_basic_interface(self):
        normal = UnitNormal(shape=[2, 3])
        samples = normal.sample(n_samples=5)
        samples_0 = samples.tensor[0]
        samples_no_grad = T.stop_grad(samples.tensor)
        log_prob = normal.log_prob(samples.tensor, group_ndims=0)
        log_prob_reduce_1 = T.reduce_sum(log_prob, axis=[-1])

        ##
        t = StochasticTensor(
            tensor=samples_no_grad,
            distribution=normal,
            n_samples=5,
            group_ndims=0,
            reparameterized=False,
        )

        self.assertIs(t.tensor, samples_no_grad)
        self.assertIs(t.distribution, normal)
        self.assertEqual(t.n_samples, 5)
        self.assertEqual(t.group_ndims, 0)
        self.assertEqual(t.reparameterized, False)
        self.assertIsNone(t.transform_origin)
        self.assertIsNone(t._cached_log_prob)
        self.assertIsNone(t._cached_prob)

        self.assertEqual(repr(t), f'StochasticTensor({t.tensor!r})')
        self.assertEqual(hash(t), hash(t))
        self.assertEqual(t, t)
        self.assertNotEqual(t, StochasticTensor(
            tensor=samples_0,
            distribution=normal,
            n_samples=5,
            group_ndims=0,
            reparameterized=False,
        ))
        self.assertEqual(t.continuous, True)

        # log_prob()
        this_log_prob = t.log_prob()
        self.assertIs(t._cached_log_prob, this_log_prob)
        self.assertIs(t.log_prob(), t._cached_log_prob)
        self.assertIs(t.log_prob(group_ndims=0), t._cached_log_prob)
        assert_allclose(this_log_prob, log_prob, rtol=1e-4)

        this_log_prob = t.log_prob(group_ndims=1)
        self.assertIsNot(this_log_prob, t._cached_log_prob)
        assert_allclose(this_log_prob, log_prob_reduce_1, rtol=1e-4)

        # prob()
        this_prob = t.prob()
        self.assertIs(t._cached_prob, this_prob)
        self.assertIs(t.prob(), t._cached_prob)
        self.assertIs(t.prob(group_ndims=0), t._cached_prob)
        assert_allclose(this_prob, np.exp(T.to_numpy(log_prob)), rtol=1e-4)

        this_prob = t.prob(group_ndims=1)
        self.assertIsNot(this_prob, t._cached_prob)
        assert_allclose(this_prob, np.exp(T.to_numpy(log_prob_reduce_1)), rtol=1e-4)

        ##
        normal.continuous = False
        t = StochasticTensor(
            tensor=samples_0,
            distribution=normal,
            n_samples=None,
            group_ndims=1,
            reparameterized=True,
            log_prob=log_prob_reduce_1,
            transform_origin=samples,
        )
        self.assertEqual(t.continuous, False)

        self.assertIs(t.tensor, samples_0)
        self.assertIs(t.distribution, normal)
        self.assertEqual(t.n_samples, None)
        self.assertEqual(t.group_ndims, 1)
        self.assertEqual(t.reparameterized, True)
        self.assertIs(t.transform_origin, samples)
        self.assertIs(t._cached_log_prob, log_prob_reduce_1)
        self.assertIsNone(t._cached_prob)
