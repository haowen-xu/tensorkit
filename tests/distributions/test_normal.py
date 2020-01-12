import unittest
from itertools import product
from typing import Optional

import mock
import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions.normal import BaseNormal
from tensorkit.distributions.utils import copy_distribution
from tests.helper import float_dtypes


class UnitNormalTestCase(unittest.TestCase):

    def test_construct(self):
        np.random.seed(1234)

        for shape, event_ndims, dtype in \
                product(([], [2, 3]), range(0, 3), float_dtypes):
            if event_ndims > len(shape):
                continue

            normal = UnitNormal(shape, dtype=dtype, event_ndims=event_ndims)
            self.assertEqual(normal.value_shape, shape)
            self.assertEqual(normal.dtype, dtype)
            self.assertEqual(normal.event_ndims, event_ndims)

            np.testing.assert_equal(T.to_numpy(normal.mean), np.zeros(shape))
            np.testing.assert_equal(T.to_numpy(normal.std), np.ones(shape))
            np.testing.assert_equal(T.to_numpy(normal.logstd), np.zeros(shape))

    def test_copy(self):
        np.random.seed(1234)
        shape = [2, 3]
        normal = UnitNormal(shape=[2, 3], event_ndims=1, dtype=T.float32)

        # read out mean, std and logstd, to ensure these cached attrs are generated
        np.testing.assert_equal(T.to_numpy(normal.mean), np.zeros(shape))
        np.testing.assert_equal(T.to_numpy(normal.std), np.ones(shape))
        np.testing.assert_equal(T.to_numpy(normal.logstd), np.zeros(shape))

        # same dtype and shape, the cached attrs are copied
        normal2 = normal.copy(event_ndims=2)
        self.assertIsInstance(normal2, UnitNormal)
        self.assertEqual(normal2.dtype, T.float32)
        self.assertEqual(normal2.value_shape, [2, 3])
        self.assertEqual(normal2.event_ndims, 2)
        for key in ('mean', 'std', 'logstd'):
            self.assertIs(getattr(normal2, key), getattr(normal, key))

        # shape mismatch, no copy cached attrs
        normal2 = normal.copy(shape=[3])
        self.assertIsInstance(normal2, UnitNormal)
        self.assertEqual(normal2.dtype, T.float32)
        self.assertEqual(normal2.value_shape, [3])
        self.assertEqual(normal2.event_ndims, 1)
        for key in ('mean', 'std', 'logstd'):
            self.assertIsNot(getattr(normal2, key), getattr(normal, key))

        # dtype mismatch, no copy cached attrs
        normal2 = normal.copy(dtype=T.float64)
        self.assertIsInstance(normal2, UnitNormal)
        self.assertEqual(normal2.dtype, T.float64)
        self.assertEqual(normal2.value_shape, [2, 3])
        self.assertEqual(normal2.event_ndims, 1)
        for key in ('mean', 'std', 'logstd'):
            self.assertIsNot(getattr(normal2, key), getattr(normal, key))

    def test_sample_and_log_prob(self):
        np.random.seed(1234)

        for dtype in float_dtypes:
            normal = UnitNormal(shape=[2, 3, 4], event_ndims=1, dtype=dtype)

            # sample(n_samples=None)
            t = normal.sample()
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, normal)
            self.assertEqual(t.n_samples, None)
            self.assertEqual(t.group_ndims, 0)
            self.assertEqual(t.reparameterized, True)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.get_dtype(t.tensor), dtype)
            self.assertEqual(T.shape(t.tensor), [2, 3, 4])

            for log_pdf in [t.log_prob(), normal.log_prob(t)]:
                np.testing.assert_allclose(
                    T.to_numpy(log_pdf),
                    T.to_numpy(
                        T.random.randn_log_pdf(given=t.tensor, group_ndims=1)
                    )
                )

            # sample(n_samples=5)
            t = normal.sample(n_samples=5, group_ndims=-1, reparameterized=False)
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, normal)
            self.assertEqual(t.n_samples, 5)
            self.assertEqual(t.group_ndims, -1)
            self.assertEqual(t.reparameterized, False)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.get_dtype(t.tensor), dtype)
            self.assertEqual(T.shape(t.tensor), [5, 2, 3, 4])

            for log_pdf in [t.log_prob(-1), normal.log_prob(t, -1)]:
                np.testing.assert_allclose(
                    T.to_numpy(log_pdf),
                    T.to_numpy(
                        T.random.randn_log_pdf(given=t.tensor, group_ndims=0)
                    )
                )


class _MyBaseNormal(BaseNormal):

    _extra_args = ('xyz',)

    def __init__(self,
                 mean: T.Tensor,
                 std: Optional[T.Tensor] = None,
                 *,
                 logstd: Optional[T.Tensor] = None,
                 reparameterized: bool = True,
                 event_ndims: int = 0,
                 validate_tensors: Optional[bool] = None,
                 xyz: int = 0):
        super().__init__(
            mean=mean, std=std, logstd=logstd,  reparameterized=reparameterized,
            event_ndims=event_ndims, validate_tensors=validate_tensors
        )
        self.xyz = xyz


class NormalTestCase(unittest.TestCase):

    def test_construct(self):
        np.random.seed(1234)
        mean = np.random.randn(3, 4)
        logstd = np.random.randn(2, 3, 4)
        std = np.exp(logstd)

        for dtype in float_dtypes:
            mean_t = T.as_tensor(mean, dtype=dtype)
            std_t = T.as_tensor(std, dtype=dtype)
            logstd_t = T.as_tensor(logstd, dtype=dtype)
            mutual_params = {'std': std_t, 'logstd': logstd_t}

            # construct from mean & std/logstd
            for key, val in mutual_params.items():
                other_key = [k for k in mutual_params if k != key][0]
                normal = _MyBaseNormal(mean=mean_t, event_ndims=1, **{key: val})
                self.assertEqual(normal.continuous, True)
                self.assertEqual(normal.reparameterized, True)
                self.assertEqual(normal.min_event_ndims, 0)
                self.assertEqual(normal.event_ndims, 1)
                self.assertIs(normal.mean, mean_t)
                self.assertIs(getattr(normal, key), val)
                np.testing.assert_allclose(
                    T.to_numpy(getattr(normal, other_key)),
                    T.to_numpy(mutual_params[other_key]),
                    rtol=1e-4
                )
                self.assertEqual(normal._mutual_params, {key: val})

                # mean and std/logstd must have the same dtype
                for other_dtype in float_dtypes:
                    if other_dtype != dtype:
                        other_val = T.cast(val, other_dtype)
                        with pytest.raises(ValueError,
                                           match=f'The dtype of `mean` does '
                                                 f'not equal the dtype of '
                                                 f'`{key}`: {dtype} vs '
                                                 f'{other_dtype}'):
                            _ = _MyBaseNormal(mean=mean_t, **{key: other_val})

            # must specify either std or logstd, but not both
            with pytest.raises(ValueError,
                               match='Either `std` or `logstd` must be '
                                     'specified, but not both.'):
                _ = _MyBaseNormal(mean=mean_t, std=std_t, logstd=logstd_t)

            with pytest.raises(ValueError,
                               match='Either `std` or `logstd` must be '
                                     'specified, but not both.'):
                _ = _MyBaseNormal(mean=mean_t, std=None, logstd=None)

            # nan test
            with pytest.raises(Exception,
                               match='Infinity or NaN value encountered'):
                _ = _MyBaseNormal(mean=T.as_tensor(np.nan, dtype=dtype),
                                  logstd=logstd_t, validate_tensors=True)

            for key, val in mutual_params.items():
                with pytest.raises(Exception,
                                   match='Infinity or NaN value encountered'):
                    _ = _MyBaseNormal(mean=mean_t, validate_tensors=True,
                                      **{key: T.as_tensor(np.nan, dtype=dtype)})

            normal = _MyBaseNormal(mean=mean_t, std=T.zeros_like(std_t),
                                   validate_tensors=True)
            with pytest.raises(Exception,
                               match='Infinity or NaN value encountered'):
                _ = normal.logstd

    def test_copy(self):
        np.random.seed(1234)
        mean = np.random.randn(3, 4)
        logstd = np.random.randn(2, 3, 4)
        mean_t = T.as_tensor(mean)
        logstd_t = T.as_tensor(logstd)
        normal = _MyBaseNormal(mean=mean_t, logstd=logstd_t, event_ndims=1,
                               xyz=123, reparameterized=False)
        self.assertEqual(normal.xyz, 123)

        with mock.patch('tensorkit.distributions.normal.copy_distribution',
                        wraps=copy_distribution) as f_copy:
            normal2 = normal.copy(event_ndims=2)
            self.assertIsInstance(normal2, _MyBaseNormal)
            self.assertEqual(normal2.xyz, 123)
            self.assertIs(normal2.mean, normal.mean)
            self.assertIs(normal2.logstd, normal.logstd)
            self.assertEqual(normal2.event_ndims, 2)
            self.assertEqual(normal2.reparameterized, False)
            self.assertEqual(f_copy.call_args, ((), {
                'cls': _MyBaseNormal,
                'base': normal,
                'attrs': ('mean', 'reparameterized', 'event_ndims',
                          'validate_tensors', 'xyz'),
                'mutual_attrs': (('std', 'logstd'),),
                'original_mutual_params': {'logstd': normal.logstd},
                'overrided_params': {'event_ndims': 2},
            }))

    def test_Normal(self):
        np.random.seed(1234)
        mean = np.random.randn(3, 4)
        logstd = np.random.randn(2, 3, 4)
        mean_t = T.as_tensor(mean)
        logstd_t = T.as_tensor(logstd)

        normal = Normal(mean=mean_t, logstd=logstd_t, event_ndims=1)

        # copy()
        normal2 = normal.copy()
        self.assertIsInstance(normal2, Normal)
        self.assertIs(normal2.logstd, logstd_t)
        self.assertEqual(normal2.event_ndims, 1)

        # sample(n_samples=None)
        t = normal.sample()
        self.assertIsInstance(t, StochasticTensor)
        self.assertIs(t.distribution, normal)
        self.assertEqual(t.n_samples, None)
        self.assertEqual(t.group_ndims, 0)
        self.assertEqual(t.reparameterized, True)
        self.assertIsInstance(t.tensor, T.Tensor)
        self.assertEqual(T.shape(t.tensor), [2, 3, 4])

        for log_pdf in [t.log_prob(), normal.log_prob(t)]:
            np.testing.assert_allclose(
                T.to_numpy(log_pdf),
                T.to_numpy(
                    T.random.normal_log_pdf(given=t.tensor, mean=mean_t,
                                            logstd=logstd_t, group_ndims=1)
                )
            )

        # sample(n_samples=5)
        t = normal.sample(n_samples=5, group_ndims=-1, reparameterized=False)
        self.assertIsInstance(t, StochasticTensor)
        self.assertIs(t.distribution, normal)
        self.assertEqual(t.n_samples, 5)
        self.assertEqual(t.group_ndims, -1)
        self.assertEqual(t.reparameterized, False)
        self.assertIsInstance(t.tensor, T.Tensor)
        self.assertEqual(T.shape(t.tensor), [5, 2, 3, 4])

        for log_pdf in [t.log_prob(-1), normal.log_prob(t, -1)]:
            np.testing.assert_allclose(
                T.to_numpy(log_pdf),
                T.to_numpy(
                    T.random.normal_log_pdf(given=t.tensor, mean=mean_t,
                                            logstd=logstd_t, group_ndims=0)
                )
            )

    def test_TruncatedNormal(self):
        np.random.seed(1234)
        mean = np.random.randn(3, 4)
        logstd = np.random.randn(2, 3, 4)
        std = np.exp(logstd)

        mean_t = T.as_tensor(mean)
        logstd_t = T.as_tensor(logstd)
        std_t = T.as_tensor(std)

        with pytest.raises(ValueError, match='`low` < `high` does not hold'):
            _ = TruncatedNormal(mean=mean_t, logstd=logstd_t, low=2., high=1.)
        with pytest.raises(ValueError, match='`low` < `high` does not hold'):
            _ = TruncatedNormal(mean=mean_t, logstd=logstd_t, low=2., high=2.)

        for low, high in [(-2., 3.), (-2., None), (None, 3.), (None, None)]:
            normal = TruncatedNormal(
                mean=mean_t, logstd=logstd_t, low=low, high=high,
                event_ndims=1, epsilon=1e-6, log_zero=-1e6,
            )

            # copy()
            normal2 = normal.copy()
            self.assertIsInstance(normal2, TruncatedNormal)
            self.assertIs(normal2.logstd, logstd_t)
            self.assertEqual(normal2.event_ndims, 1)
            self.assertEqual(normal2.low, low)
            self.assertEqual(normal2.high, high)
            self.assertEqual(normal2.event_ndims, 1)
            self.assertEqual(normal2.epsilon, 1e-6)
            self.assertEqual(normal2.log_zero, -1e6)

            # sample(n_samples=None)
            t = normal.sample()
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, normal)
            self.assertEqual(t.n_samples, None)
            self.assertEqual(t.group_ndims, 0)
            self.assertEqual(t.reparameterized, True)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), [2, 3, 4])

            for log_pdf in [t.log_prob(), normal.log_prob(t)]:
                np.testing.assert_allclose(
                    T.to_numpy(log_pdf),
                    T.to_numpy(
                        T.random.truncated_normal_log_pdf(
                            given=t.tensor, mean=mean_t, std=std_t,
                            logstd=logstd_t, group_ndims=1, low=low, high=high,
                        )
                    )
                )

            # sample(n_samples=5)
            t = normal.sample(n_samples=5, group_ndims=-1, reparameterized=False)
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, normal)
            self.assertEqual(t.n_samples, 5)
            self.assertEqual(t.group_ndims, -1)
            self.assertEqual(t.reparameterized, False)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), [5, 2, 3, 4])

            for log_pdf in [t.log_prob(-1), normal.log_prob(t, -1)]:
                np.testing.assert_allclose(
                    T.to_numpy(log_pdf),
                    T.to_numpy(
                        T.random.truncated_normal_log_pdf(
                            given=t.tensor, mean=mean_t, std=std_t,
                            logstd=logstd_t, group_ndims=0, low=low, high=high,
                        )
                    )
                )
