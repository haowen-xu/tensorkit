import unittest
from itertools import product

import mock
import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tensorkit.distributions.utils import copy_distribution
from tests.helper import *


def sigmoid(x):
    return np.where(x >= 0, 1. / (1 + np.exp(-x)),
                    np.exp(x) / (1 + np.exp(x)))


class BernoulliTestCase(TestCase):

    def test_construct(self):
        logits = np.random.randn(2, 3, 4)
        probs = sigmoid(logits)

        for int_dtype, float_dtype in product(int_dtypes, float_dtypes):
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            probs_t = T.as_tensor(probs, dtype=float_dtype)
            mutual_params = {'logits': logits_t, 'probs': probs_t}

            # construct from logits or probs
            for key, val in mutual_params.items():
                other_key = [k for k in mutual_params if k != key][0]
                bernoulli = Bernoulli(event_ndims=1, dtype=int_dtype,
                                      epsilon=1e-6, **{key: val})
                self.assertEqual(bernoulli.continuous, False)
                self.assertEqual(bernoulli.reparameterized, False)
                self.assertEqual(bernoulli.min_event_ndims, 0)
                self.assertEqual(bernoulli.dtype, int_dtype)
                self.assertEqual(bernoulli.event_ndims, 1)
                self.assertEqual(bernoulli.epsilon, 1e-6)
                self.assertIs(getattr(bernoulli, key), val)
                assert_allclose(
                    getattr(bernoulli, other_key),
                    mutual_params[other_key],
                    rtol=1e-4
                )
                self.assertEqual(bernoulli._mutual_params, {key: val})

            # must specify either logits or probs, but not both
            with pytest.raises(ValueError,
                               match='Either `logits` or `probs` must be '
                                     'specified, but not both.'):
                _ = Bernoulli(logits=logits_t, probs=probs_t, dtype=int_dtype)

            with pytest.raises(ValueError,
                               match='Either `logits` or `probs` must be '
                                     'specified, but not both.'):
                _ = Bernoulli(logits=None, probs=None, dtype=int_dtype)

            # nan test
            for key, val in mutual_params.items():
                with pytest.raises(Exception,
                                   match='Infinity or NaN value encountered'):
                    _ = Bernoulli(
                        validate_tensors=True, dtype=int_dtype,
                        **{key: T.as_tensor(np.nan, dtype=float_dtype)})

    def test_copy(self):
        logits = np.random.randn(2, 3, 4)
        logits_t = T.as_tensor(logits)
        bernoulli = Bernoulli(logits=logits_t, event_ndims=1)

        with mock.patch('tensorkit.distributions.bernoulli.copy_distribution',
                        wraps=copy_distribution) as f_copy:
            bernoulli2 = bernoulli.copy(event_ndims=2)
            self.assertIsInstance(bernoulli2, Bernoulli)
            self.assertIs(bernoulli2.logits, bernoulli.logits)
            self.assertEqual(bernoulli2.event_ndims, 2)
            self.assertEqual(f_copy.call_args, ((), {
                'cls': Bernoulli,
                'base': bernoulli,
                'attrs': ('dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
                'mutual_attrs': (('logits', 'probs'),),
                'compute_deps': {'logits': ('epsilon',)},
                'original_mutual_params': {'logits': bernoulli.logits},
                'overrided_params': {'event_ndims': 2},
            }))

    def test_sample_and_log_prob(self):
        logits = np.random.randn(2, 3, 4)
        logits_t = T.as_tensor(logits)

        for int_dtype in int_dtypes:
            bernoulli = Bernoulli(logits=logits_t, event_ndims=1,
                                  dtype=int_dtype)

            # n_samples is None
            t = bernoulli.sample()
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, bernoulli)
            self.assertEqual(T.get_dtype(t.tensor), int_dtype)
            self.assertEqual(t.n_samples, None)
            self.assertEqual(t.group_ndims, 0)
            self.assertEqual(t.reparameterized, False)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), [2, 3, 4])

            for log_pdf in [t.log_prob(), bernoulli.log_prob(t)]:
                assert_allclose(
                    log_pdf,
                    T.random.bernoulli_log_prob(
                        given=t.tensor, logits=logits_t, group_ndims=1)
                )

            # n_samples == 5
            t = bernoulli.sample(n_samples=5, group_ndims=-1)
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, bernoulli)
            self.assertEqual(T.get_dtype(t.tensor), int_dtype)
            self.assertEqual(t.n_samples, 5)
            self.assertEqual(t.group_ndims, -1)
            self.assertEqual(t.reparameterized, False)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), [5, 2, 3, 4])

            for log_pdf in [t.log_prob(-1), bernoulli.log_prob(t, -1)]:
                assert_allclose(
                    log_pdf,
                    T.random.bernoulli_log_prob(
                        given=t.tensor, logits=logits_t, group_ndims=0)
                )
