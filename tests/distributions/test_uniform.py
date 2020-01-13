import unittest
from itertools import product

import mock
import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions.utils import copy_distribution
from tests.helper import float_dtypes


class UniformTestCase(unittest.TestCase):

    def test_construct(self):
        np.random.seed(1234)

        for dtype in float_dtypes:
            # specify no args
            uniform = Uniform(dtype=dtype, event_ndims=0)
            self.assertEqual(uniform.value_shape, [])
            self.assertEqual(uniform.low, None)
            self.assertEqual(uniform.high, None)
            self.assertEqual(uniform.event_ndims, 0)
            self.assertEqual(uniform.log_zero, -1e7)

            # specify `low` and `high` float
            uniform = Uniform(low=-1., high=2., dtype=dtype, event_ndims=0)
            self.assertEqual(uniform.value_shape, [])
            self.assertEqual(uniform.low, -1.)
            self.assertEqual(uniform.high, 2.)
            self.assertEqual(uniform.event_ndims, 0)

            # specify `low`, `high` tensors
            low_t = T.full([2, 1], -1., dtype=dtype)
            high_t = T.full([1, 3], 2., dtype=dtype)
            uniform = Uniform(low=low_t, high=high_t, dtype=dtype,
                              event_ndims=2)
            self.assertEqual(uniform.value_shape, [2, 3])
            self.assertEqual(uniform.dtype, dtype)
            self.assertEqual(uniform.event_ndims, 2)
            self.assertIs(uniform.low, low_t)
            self.assertIs(uniform.high, high_t)

            # specify `low` or `high`, one as tensor and one as numpy array
            for low, high in [(low_t, T.to_numpy(high_t)),
                              (T.to_numpy(low_t), high_t)]:
                uniform = Uniform(low=low, high=high,
                                  dtype=T.float32,  # should be ignored
                                  event_ndims=2)
                self.assertEqual(uniform.value_shape, [2, 3])
                self.assertEqual(uniform.dtype, dtype)
                self.assertEqual(T.get_dtype(uniform.low), dtype)
                self.assertEqual(T.get_dtype(uniform.high), dtype)
                self.assertEqual(uniform.event_ndims, 2)
                np.testing.assert_equal(
                    T.to_numpy(uniform.low), T.to_numpy(low_t))
                np.testing.assert_equal(
                    T.to_numpy(uniform.high), T.to_numpy(high_t))

        for event_ndims, dtype, shape in product(range(0, 3), float_dtypes,
                                                 ([], [2, 3])):
            if event_ndims > len(shape):
                continue

            # specify `shape`
            uniform = Uniform(shape=shape, dtype=dtype, event_ndims=event_ndims)
            self.assertEqual(uniform.value_shape, shape)
            self.assertEqual(uniform.dtype, dtype)
            self.assertEqual(uniform.event_ndims, event_ndims)
            self.assertEqual(uniform.low, None)
            self.assertEqual(uniform.high, None)

            # specify `shape` and `low`, `high` floats
            uniform = Uniform(shape=shape, low=-1., high=2., dtype=dtype,
                              event_ndims=event_ndims)
            self.assertEqual(uniform.value_shape, shape)
            self.assertEqual(uniform.dtype, dtype)
            self.assertEqual(uniform.event_ndims, event_ndims)
            self.assertEqual(uniform.low, -1.)
            self.assertEqual(uniform.high, 2.)

            # specify just one of `low`, `high` as float, another as tensor
            uniform = Uniform(
                shape=shape, low=-1., high=T.as_tensor(2., dtype=dtype),
                event_ndims=event_ndims)
            self.assertEqual(uniform.value_shape, shape)
            self.assertEqual(uniform.dtype, dtype)
            self.assertEqual(uniform.event_ndims, event_ndims)
            self.assertEqual(uniform.low, -1.)
            self.assertEqual(uniform.high, 2.)

        for event_ndims, dtype, shape in product(range(0, 3), float_dtypes,
                                                 ([], [2, 3])):
            if event_ndims > len(shape) + 2:
                continue
            # specify `shape` and `low`, `high` tensors
            low_t = T.full([2, 1], -1., dtype=dtype)
            high_t = T.full([1, 3], 2., dtype=dtype)
            uniform = Uniform(shape=shape, low=low_t, high=high_t,
                              dtype=T.float32,  # should be ignored
                              event_ndims=event_ndims)
            self.assertEqual(uniform.value_shape, shape + [2, 3])
            self.assertEqual(uniform.dtype, dtype)
            self.assertEqual(uniform.event_ndims, event_ndims)
            self.assertIs(uniform.low, low_t)
            self.assertIs(uniform.high, high_t)

        with pytest.raises(ValueError,
                           match='`low` and `high` must be both specified, or '
                                 'neither specified'):
            _ = Uniform(low=-1.)

        with pytest.raises(ValueError,
                           match='`high.dtype` != `low.dtype`: float64 vs float32'):
            _ = Uniform(low=T.full([2, 3], -1., dtype=T.float32),
                        high=T.full([2, 3], 2., dtype=T.float64))

        with pytest.raises(ValueError,
                           match='`low` < `high` does not hold: `low` == 2.0, '
                                 '`high` == 1.0'):
            _ = Uniform(low=2., high=1.)

        with pytest.raises(Exception, match='`low` < `high` does not hold'):
            _ = Uniform(low=T.full([2, 3], 2., dtype=T.float32),
                        high=T.full([2, 3], -1., dtype=T.float32),
                        validate_tensors=True)

    def test_copy(self):
        np.random.seed(1234)
        T.random.seed(1234)

        for dtype in float_dtypes:
            low_t = T.full([2, 1], -1., dtype=dtype)
            high_t = T.full([1, 3], 2., dtype=dtype)
            uniform = Uniform(shape=[5, 4], low=low_t, high=high_t,
                              event_ndims=1, log_zero=-1e6,
                              reparameterized=False)
            self.assertIs(uniform.low, low_t)
            self.assertIs(uniform.high, high_t)
            self.assertEqual(uniform.reparameterized, False)
            self.assertEqual(uniform.value_shape, [5, 4, 2, 3])
            self.assertEqual(uniform.dtype, dtype)
            self.assertEqual(uniform.event_ndims, 1)
            self.assertEqual(uniform.log_zero, -1e6)

            with mock.patch('tensorkit.distributions.uniform.copy_distribution',
                            wraps=copy_distribution) as f_copy:
                uniform2 = uniform.copy(event_ndims=2)
                self.assertIsInstance(uniform2, Uniform)
                self.assertIs(uniform2.low, low_t)
                self.assertIs(uniform2.high, high_t)
                self.assertEqual(uniform2.value_shape, [5, 4, 2, 3])
                self.assertEqual(uniform2.dtype, dtype)
                self.assertEqual(uniform2.event_ndims, 2)
                self.assertEqual(uniform2.log_zero, -1e6)
                self.assertEqual(f_copy.call_args, ((), {
                    'cls': Uniform,
                    'base': uniform,
                    'attrs': (('shape', '_shape'), 'low', 'high', 'dtype',
                              'reparameterized', 'event_ndims', 'log_zero',
                              'validate_tensors'),
                    'overrided_params': {'event_ndims': 2},
                }))

    def test_sample_and_log_prob(self):
        np.random.seed(1234)
        T.random.seed(1234)

        array_low = np.random.randn(2, 1)
        array_high = np.exp(np.random.randn(1, 3)) + 1.
        log_zero = -1e6

        def log_prob(x, low, high, group_ndims=0):
            if low is None and high is None:
                low, high = 0., 1.
            log_pdf = -np.log(np.ones_like(x) * (high - low))
            log_pdf = np.where(
                np.logical_and(low <= x, x <= high), log_pdf, log_zero)
            log_pdf = np.sum(log_pdf, axis=tuple(range(-group_ndims, 0)))
            return log_pdf

        for shape, dtype, (low, high), event_ndims in product(
                [None, [], [5, 4]],
                float_dtypes,
                [(None, None), (-1., 2.), (array_low, array_high)],
                range(5)):
            low_rank = len(np.shape(low)) if low is not None else 0
            if event_ndims > len(shape or []) + low_rank:
                continue

            if isinstance(low, np.ndarray):
                low_t = T.as_tensor(low, dtype=dtype)
                high_t = T.as_tensor(high, dtype=dtype)
                uniform = Uniform(shape=shape, low=low_t, high=high_t,
                                  event_ndims=event_ndims, log_zero=log_zero)
                value_shape = (shape or []) + [2, 3]
                self.assertIs(uniform.low, low_t)
                self.assertIs(uniform.high, high_t)
            else:
                uniform = Uniform(shape=shape, low=low, high=high, dtype=dtype,
                                  event_ndims=event_ndims, log_zero=log_zero)
                value_shape = shape or []
                self.assertEqual(uniform.low, low)
                self.assertEqual(uniform.high, high)
            self.assertEqual(uniform.log_zero, log_zero)
            self.assertEqual(uniform.value_shape, value_shape)

            # sample(n_samples=None)
            t = uniform.sample()
            x = T.to_numpy(t.tensor)
            sample_shape = value_shape
            self.assertIsInstance(t, StochasticTensor)
            self.assertIs(t.distribution, uniform)
            self.assertEqual(T.get_dtype(t.tensor), dtype)
            self.assertEqual(t.n_samples, None)
            self.assertEqual(t.group_ndims, 0)
            self.assertEqual(t.reparameterized, True)
            self.assertIsInstance(t.tensor, T.Tensor)
            self.assertEqual(T.shape(t.tensor), sample_shape)

            for log_pdf in [t.log_prob(), uniform.log_prob(t)]:
                self.assertEqual(T.get_dtype(log_pdf), dtype)
                np.testing.assert_allclose(
                    T.to_numpy(log_pdf), log_prob(x, low, high, event_ndims),
                    rtol=1e-4
                )
            # test log-prob on out-of-range values
            np.testing.assert_allclose(
                T.to_numpy(uniform.log_prob(t.tensor * 10.)),
                log_prob(x * 10., low, high, event_ndims),
                rtol=1e-4,
            )

            # sample(n_samples=7)
            if event_ndims >= 1:
                t = uniform.sample(n_samples=7, group_ndims=-1,
                                   reparameterized=False)
                x = T.to_numpy(t.tensor)
                sample_shape = [7] + value_shape
                self.assertIsInstance(t, StochasticTensor)
                self.assertIs(t.distribution, uniform)
                self.assertEqual(T.get_dtype(t.tensor), dtype)
                self.assertEqual(t.n_samples, 7)
                self.assertEqual(t.group_ndims, -1)
                self.assertEqual(t.reparameterized, False)
                self.assertIsInstance(t.tensor, T.Tensor)
                self.assertEqual(T.shape(t.tensor), sample_shape)
                reduce_ndims = event_ndims - 1

                for log_pdf in [t.log_prob(),
                                uniform.log_prob(t, group_ndims=-1)]:
                    self.assertEqual(T.get_dtype(log_pdf), dtype)
                    np.testing.assert_allclose(
                        T.to_numpy(log_pdf),
                        log_prob(x, low, high, reduce_ndims),
                        rtol=1e-4
                    )

        # test reparameterized
        low_t = T.requires_grad(T.as_tensor(array_low))
        high_t = T.requires_grad(T.as_tensor(array_high))
        uniform = Uniform(low=low_t, high=high_t)

        t = uniform.sample()
        self.assertTrue(t.reparameterized)
        u = (T.to_numpy(t.tensor) - array_low) / (array_high - array_low)
        [low_grad, high_grad] = T.grad([T.reduce_sum(t.tensor)], [low_t, high_t])
        np.testing.assert_allclose(
            T.to_numpy(low_grad), np.sum(1. - u, axis=-1, keepdims=True), rtol=1e-4)
        np.testing.assert_allclose(
            T.to_numpy(high_grad), np.sum(u, axis=0, keepdims=True), rtol=1e-4)

        t = uniform.sample(reparameterized=False)
        w_t = T.requires_grad(T.as_tensor(np.random.randn(2, 3)))
        self.assertFalse(t.reparameterized)
        [low_grad, high_grad] = T.grad(
            [T.reduce_sum(w_t * t.tensor)], [low_t, high_t],
            allow_unused=True)
        self.assertTrue(T.is_null_grad(low_t, low_grad))
        self.assertTrue(T.is_null_grad(high_t, high_grad))
