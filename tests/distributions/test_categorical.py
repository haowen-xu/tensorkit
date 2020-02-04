import unittest
from itertools import product

import mock
import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tensorkit.distributions.categorical import BaseCategorical
from tensorkit.distributions.utils import copy_distribution
from tests.helper import *


def log_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_diff = x - x_max
    return x_diff - np.log(np.sum(np.exp(x_diff), axis=axis, keepdims=True))


def softmax(x, axis=-1):
    return np.exp(log_softmax(x, axis))


class _MyBaseCategorical(BaseCategorical):
    min_event_ndims = 1

    def __init__(self, **kwargs):
        for mutual_key in ('logits', 'probs'):
            kwargs.setdefault(mutual_key, None)
        super().__init__(**kwargs)


class CategoricalTestCase(unittest.TestCase):

    def test_construct_base(self):
        np.random.seed(1234)
        logits = np.random.randn(2, 3, 4)
        probs = softmax(logits)
        logits = np.log(probs)

        for dtype, float_dtype in product(number_dtypes, float_dtypes):
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            probs_t = T.as_tensor(probs, dtype=float_dtype)
            mutual_params = {'logits': logits_t, 'probs': probs_t}

            # construct from logits or probs
            for key, val in mutual_params.items():
                other_key = [k for k in mutual_params if k != key][0]
                cat = _MyBaseCategorical(
                    event_ndims=1, dtype=dtype,
                    epsilon=1e-6, **{key: val})
                self.assertEqual(cat.continuous, False)
                self.assertEqual(cat.reparameterized, False)
                self.assertEqual(cat.dtype, dtype)
                self.assertEqual(cat.event_ndims, 1)
                self.assertEqual(cat.epsilon, 1e-6)
                self.assertIs(getattr(cat, key), val)
                assert_allclose(
                    getattr(cat, other_key),
                    mutual_params[other_key],
                    rtol=1e-4
                )
                self.assertEqual(cat._mutual_params, {key: val})

            # must specify either logits or probs, but not both
            with pytest.raises(ValueError,
                               match='Either `logits` or `probs` must be '
                                     'specified, but not both.'):
                _ = _MyBaseCategorical(
                    logits=logits_t, probs=probs_t, dtype=dtype,
                    event_ndims=0
                )

            with pytest.raises(ValueError,
                               match='Either `logits` or `probs` must be '
                                     'specified, but not both.'):
                _ = _MyBaseCategorical(
                    logits=None, probs=None, dtype=dtype,
                    event_ndims=0
                )

            # shape test on logits or probs
            for key in mutual_params:
                param_val = T.zeros([], dtype=float_dtype)
                with pytest.raises(ValueError,
                                   match=rf'`{key}` must be at least 1d: '
                                         rf'got shape \[\]'):
                    _ = _MyBaseCategorical(dtype=dtype, event_ndims=0,
                                           **{key: param_val})

            # nan test
            for key, val in mutual_params.items():
                with pytest.raises(Exception,
                                   match='Infinity or NaN value encountered'):
                    _ = _MyBaseCategorical(
                        validate_tensors=True, dtype=dtype, event_ndims=2,
                        **{key: T.as_tensor(np.asarray([[np.nan]]),
                                             dtype=float_dtype)}
                    )

    def test_copy(self):
        np.random.seed(1234)
        logits = np.random.randn(2, 3, 4)
        logits_t = T.as_tensor(logits)
        cat = _MyBaseCategorical(logits=logits_t, probs=None, event_ndims=1,
                                 dtype=T.int32)

        with mock.patch('tensorkit.distributions.categorical.copy_distribution',
                        wraps=copy_distribution) as f_copy:
            cat2 = cat.copy(event_ndims=2)
            self.assertIsInstance(cat2, _MyBaseCategorical)
            self.assertIs(cat2.logits, cat.logits)
            self.assertEqual(cat2.event_ndims, 2)
            self.assertEqual(f_copy.call_args, ((), {
                'cls': _MyBaseCategorical,
                'base': cat,
                'attrs': ('dtype', 'event_ndims', 'validate_tensors', 'epsilon'),
                'mutual_attrs': (('logits', 'probs'),),
                'compute_deps': {'logits': ('epsilon',)},
                'original_mutual_params': {'logits': cat.logits},
                'overrided_params': {'event_ndims': 2},
            }))

    def test_Categorical_and_OneHotCategorical(self):
        np.random.seed(1234)
        logits = np.random.randn(2, 3, 4)

        def do_test(dtype, float_dtype, is_one_hot):
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            if is_one_hot:
                cls, other_cls = OneHotCategorical, Categorical
                sample_shape = [2, 3, 4]
                Z_log_prob_fn = T.random.one_hot_categorical_log_prob
                min_event_ndims = 1
            else:
                cls, other_cls = Categorical, OneHotCategorical
                sample_shape = [2, 3]
                Z_log_prob_fn = T.random.categorical_log_prob
                min_event_ndims = 0

            cat = cls(logits=logits_t,
                      dtype=dtype,
                      event_ndims=1 + int(cls is OneHotCategorical))
            self.assertFalse(cat.continuous)
            self.assertFalse(cat.reparameterized)
            self.assertEqual(cat.min_event_ndims, min_event_ndims)

            # copy()
            cat2 = cat.copy()
            self.assertIsInstance(cat2, cls)
            self.assertIs(cat2.logits, cat.logits)

            # to_indexed
            for dtype2 in number_dtypes:
                cat2 = cat.to_indexed(dtype=dtype2)
                self.assertIsInstance(cat2, Categorical)
                self.assertEqual(cat2.dtype, dtype2)
                if cls == Categorical and dtype2 == dtype:
                    self.assertIs(cat2, cat)
                else:
                    self.assertIs(cat2.logits, cat.logits)
                    self.assertEqual(
                        cat2.event_ndims,
                        cat.event_ndims - int(cls is OneHotCategorical))

            # to_one_hot
            for dtype2 in number_dtypes:
                cat2 = cat.to_one_hot(dtype=dtype2)
                self.assertIsInstance(cat2, OneHotCategorical)
                self.assertEqual(cat2.dtype, dtype2)
                if cls == OneHotCategorical and dtype2 == dtype:
                    self.assertIs(cat2, cat)
                else:
                    self.assertIs(cat2.logits, cat.logits)
                    self.assertEqual(
                        cat2.event_ndims,
                        cat.event_ndims + int(cls is Categorical))

            # sample(n_samples=None)
            t = cat.sample()
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, cat)
            self.assertEqual(T.get_dtype(t.tensor), dtype)
            self.assertEqual(t.n_samples, None)
            self.assertEqual(t.group_ndims, 0)
            self.assertEqual(t.reparameterized, False)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), sample_shape)

            for log_pdf in [t.log_prob(), cat.log_prob(t)]:
                assert_allclose(
                    log_pdf,
                    Z_log_prob_fn(given=t.tensor, logits=logits_t, group_ndims=1)
                )

            # sample(n_samples=5)
            t = cat.sample(n_samples=5, group_ndims=-1)
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, cat)
            self.assertEqual(T.get_dtype(t.tensor), dtype)
            self.assertEqual(t.n_samples, 5)
            self.assertEqual(t.group_ndims, -1)
            self.assertEqual(t.reparameterized, False)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), [5] + sample_shape)

            for log_pdf in [t.log_prob(-1), cat.log_prob(t, -1)]:
                assert_allclose(
                    log_pdf,
                    Z_log_prob_fn(given=t.tensor, logits=logits_t, group_ndims=0)
                )

        for is_one_hot in [True, False]:
            do_test(T.int32, T.float32, is_one_hot)
            do_test(T.int64, T.float64, is_one_hot)
