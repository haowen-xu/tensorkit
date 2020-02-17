import math
import unittest
from functools import partial
from itertools import product
from typing import *

import numpy as np
import pytest
from scipy.stats import norm

from tensorkit import tensor as T
from tests.helper import *


def do_check_log_prob(given, batch_ndims, Z_log_prob_fn, np_log_prob):
    # test log_prob
    for group_ndims in range(0, batch_ndims + 1):
        assert_allclose(
            Z_log_prob_fn(given, group_ndims=group_ndims),
            np.sum(np_log_prob, axis=tuple(range(-group_ndims, 0))),
            rtol=1e-2
        )
    with pytest.raises(Exception, match='`group_ndims` is too large'):
        _ = Z_log_prob_fn(given, group_ndims=batch_ndims + 1)


def normal_cdf(x):
    return norm.cdf(x)


class TensorRandomTestCase(unittest.TestCase):

    def test_seed(self):
        T.random.seed(1234)
        x = T.to_numpy(T.random.normal(T.as_tensor(0.), T.as_tensor(1.)))
        y = T.to_numpy(T.random.normal(T.as_tensor(0.), T.as_tensor(1.)))
        self.assertFalse(np.allclose(x, y))

        T.random.seed(1234)
        z = T.to_numpy(T.random.normal(T.as_tensor(0.), T.as_tensor(1.)))
        assert_allclose(x, z)

    def test_rand(self):
        np.random.seed(1234)
        T.random.seed(1234)

        for dtype in float_dtypes:
            # test sample dtype and shape
            t = T.random.rand([n_samples, 2, 3, 4], dtype=dtype)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

            # test sample mean
            x = T.to_numpy(t)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(0.5 - x_mean),
                (3. * np.sqrt(1. / 12) / np.sqrt(n_samples) *
                 np.ones_like(x_mean))
            )

    def test_uniform(self):
        np.random.seed(1234)
        T.random.seed(1234)

        for low, high in [(-1., 2.), (3.5, 7.5)]:
            for dtype in float_dtypes:
                # test sample dtype and shape
                t = T.random.uniform([n_samples, 2, 3, 4], low=low, high=high,
                                     dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

                # test sample mean
                x = T.to_numpy(t)
                x_mean = np.mean(x, axis=0)
                np.testing.assert_array_less(
                    np.abs(x_mean - 0.5 * (high + low)),
                    (3. * np.sqrt((high - low) ** 2 / 12) / np.sqrt(n_samples) *
                     np.ones_like(x_mean))
                )

        with pytest.raises(Exception,
                           match='`low` < `high` does not hold'):
            _ = T.random.uniform([2, 3, 4], low=2., high=1.)

    def test_shuffle_and_random_permutation(self):
        T.random.seed(1234)
        x = np.arange(24).reshape([2, 3, 4])

        # shuffle
        for axis in range(-len(x.shape), len(x.shape)):
            equal_count = 0
            for k in range(100):
                t = T.random.shuffle(T.from_numpy(x), axis=axis)
                if np.all(np.equal(T.to_numpy(t), x)):
                    equal_count += 1
                assert_equal(np.sort(T.to_numpy(t), axis=axis), x)
            self.assertLess(equal_count, 100)

        # random_permutation
        for dtype in int_dtypes:
            for n in [0, 1, 5]:
                x = np.arange(n)
                equal_count = 0
                for k in range(100):
                    t = T.random.random_permutation(n, dtype=dtype)
                    self.assertEqual(T.get_dtype(t), dtype)
                    if np.all(np.equal(T.to_numpy(t), x)):
                        equal_count += 1
                    assert_equal(np.sort(T.to_numpy(t)), x)
                if n > 1:
                    self.assertLess(equal_count, 100)

    def test_randn(self):
        np.random.seed(1234)
        T.random.seed(1234)

        for dtype in float_dtypes:
            # test sample dtype and shape
            t = T.random.randn([n_samples, 2, 3, 4], dtype=dtype)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

            # test sample mean
            x = T.to_numpy(t)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(x_mean),
                3. / np.sqrt(n_samples) * np.ones_like(x_mean)
            )

            # test log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(x.shape),
                Z_log_prob_fn=T.random.randn_log_pdf,
                np_log_prob=np.log(np.exp(-x ** 2 / 2.) / np.sqrt(2 * np.pi)))

    def test_truncated_randn(self):
        np.random.seed(1234)
        T.random.seed(1234)
        log_zero = -1e6

        def log_prob(given, low, high):
            log_pdf = (-given ** 2 / 2. - 0.5 * np.log(2 * np.pi))

            if low is not None or high is not None:
                low_cdf = normal_cdf(low) if low is not None else 0.
                high_cdf = normal_cdf(high) if high is not None else 1.
                log_Z = np.log(high_cdf - low_cdf)
                log_pdf -= log_Z

                # filter out zero ranges
                filters = []
                if low is not None:
                    filters.append(low <= given)
                if high is not None:
                    filters.append(given <= high)
                if len(filters) > 1:
                    filters = [np.logical_and(*filters)]
                log_pdf = np.where(filters[0], log_pdf, log_zero)
            return log_pdf

        for low, high in [(-2., 3.), (-2., None), (None, 3.), (None, None)]:
            for dtype in float_dtypes:
                # test sample dtype and shape
                t = T.random.truncated_randn(
                    [n_samples, 2, 3, 4], low=low, high=high, dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

                # test sample value range
                x = T.to_numpy(t)
                if low is not None:
                    self.assertGreaterEqual(np.min(x), low)
                if high is not None:
                    self.assertLessEqual(np.max(x), high)

                # test log_prob
                do_check_log_prob(
                    given=t,
                    batch_ndims=len(x.shape),
                    Z_log_prob_fn=partial(
                        T.random.truncated_randn_log_pdf,
                        low=low, high=high, log_zero=log_zero,
                    ),
                    np_log_prob=log_prob(x, low, high)
                )
                do_check_log_prob(
                    given=t * 10.,  # where the majority is out of range [low, high]
                    batch_ndims=len(x.shape),
                    Z_log_prob_fn=partial(
                        T.random.truncated_randn_log_pdf,
                        low=low, high=high, log_zero=log_zero,
                    ),
                    np_log_prob=log_prob(x * 10., low, high)
                )

        # test Z almost equal to zero, thus `... - log(Z)` == -log_zero
        assert_equal(
            T.random.truncated_randn_log_pdf(
                T.as_tensor(np.array(10000.0)),
                low=9999.0,
                high=10001.0,
            ),
            1e7
        )

    def test_normal(self):
        np.random.seed(1234)
        T.random.seed(1234)

        mean = np.random.randn(2, 3, 4)
        logstd = np.random.randn(3, 4)
        std = np.exp(logstd)

        def log_prob(given):
            # np.log(np.exp(-(given - mean) ** 2 / (2. * std ** 2)) /
            #        (np.sqrt(2 * np.pi) * std))
            return (
                -(given - mean) ** 2 * (0.5 * np.exp(-2. * logstd)) -
                np.log(np.sqrt(2 * np.pi)) - logstd
            )

        # test n_samples by manual expanding the param shape
        for dtype in float_dtypes:
            # test sample dtype and shape
            mean_t = T.cast(T.expand(T.as_tensor(mean), [n_samples, 2, 3, 4]), dtype)
            std_t = T.cast(T.expand(T.as_tensor(std), [n_samples, 1, 3, 4]), dtype)
            logstd_t = T.cast(T.expand(T.as_tensor(logstd), [n_samples, 1, 3, 4]), dtype)
            t = T.random.normal(mean_t, std_t)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

            # test sample mean
            x = T.to_numpy(t)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(x_mean - mean),
                np.tile(np.expand_dims(3 * std / np.sqrt(n_samples), axis=0),
                        [2, 1, 1])
            )

            # test log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.normal_log_pdf, mean=mean_t, logstd=logstd_t),
                np_log_prob=log_prob(x))

        # test with n_samples
        for dtype in float_dtypes:
            # test sample dtype and shape
            mean_t = T.as_tensor(mean, dtype)
            std_t = T.as_tensor(std, dtype)
            logstd_t = T.as_tensor(logstd, dtype)
            t = T.random.normal(mean_t, std_t, n_samples=n_samples)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

            # test sample mean
            x = T.to_numpy(t)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(x_mean - mean),
                np.tile(np.expand_dims(3 * std / np.sqrt(n_samples), axis=0),
                        [2, 1, 1])
            )

            # test log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.normal_log_pdf, mean=mean_t, logstd=logstd_t),
                np_log_prob=log_prob(x))

        # test no n_samples
        for dtype in float_dtypes:
            mean_t = T.as_tensor(mean, dtype)
            std_t = T.as_tensor(std, dtype)
            logstd_t = T.as_tensor(logstd, dtype)
            t = T.random.normal(mean_t, std_t)
            self.assertEqual(T.get_dtype(t), dtype)

            # test log_prob
            x = T.to_numpy(t)
            do_check_log_prob(
                given=t,
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.normal_log_pdf, mean=mean_t, logstd=logstd_t),
                np_log_prob=log_prob(x))

        # test reparameterized
        w = np.random.randn(2, 3, 4)

        for dtype in float_dtypes:
            w_t = T.requires_grad(T.as_tensor(w))
            mean_t = T.requires_grad(T.as_tensor(mean, dtype))
            std_t = T.requires_grad(T.as_tensor(std, dtype))
            t = w_t * T.random.normal(mean_t, std_t)
            [mean_grad, std_grad] = T.grad(
                [t], [mean_t, std_t], [T.ones_like(t)])
            assert_allclose(mean_grad, w, rtol=1e-4)
            assert_allclose(
                std_grad,
                np.sum(T.to_numpy((t - w_t * mean_t) / std_t), axis=0),
                rtol=1e-4
            )

        # test not reparameterized
        for dtype in float_dtypes:
            w_t = T.requires_grad(T.as_tensor(w))
            mean_t = T.requires_grad(T.as_tensor(mean, dtype))
            std_t = T.requires_grad(T.as_tensor(std, dtype))
            t = w_t * T.random.normal(mean_t, std_t, reparameterized=False)
            [mean_grad, std_grad] = T.grad(
                [t], [mean_t, std_t], [T.ones_like(t)], allow_unused=True)
            self.assertTrue(T.is_null_grad(mean_t, mean_grad))
            self.assertTrue(T.is_null_grad(std_t, std_grad))

        # given has lower rank than params, broadcasted to match param
        for dtype in float_dtypes:
            mean_t = T.as_tensor(mean, dtype)
            logstd_t = T.as_tensor(logstd, dtype)
            for val in (0., 1., -1.):
                assert_allclose(
                    T.random.normal_log_pdf(T.float_scalar(val), mean_t, logstd_t),
                    log_prob(val),
                    rtol=1e-4
                )

        # dtype mismatch
        with pytest.raises(Exception, match='`mean.dtype` != `std.dtype`'):
            _ = T.random.normal(T.as_tensor(mean, T.float32),
                                T.as_tensor(std, T.float64))

        # check numerics
        mean_t = T.as_tensor(mean)
        std_t = T.zeros_like(mean_t)
        logstd_t = T.as_tensor(T.log(std_t))
        t = T.random.normal(mean_t, std_t)
        with pytest.raises(Exception,
                           match='Infinity or NaN value encountered'):
            _ = T.random.normal_log_pdf(
                t, mean_t, logstd_t, validate_tensors=True)

    def test_truncated_normal(self):
        np.random.seed(1234)
        T.random.seed(1234)

        mean = np.random.randn(2, 3, 4)
        logstd = np.random.randn(3, 4)
        std = np.exp(logstd)
        log_zero = -1e6

        def log_prob(given, low, high):
            # np.log(np.exp(-(given - mean) ** 2 / (2. * std ** 2)) /
            #        (np.sqrt(2 * np.pi) * std))
            log_pdf = (
                -(given - mean) ** 2 * (0.5 * np.exp(-2. * logstd)) -
                0.5 * np.log(2 * np.pi) - logstd
            )
            if low is not None or high is not None:
                low_cdf = normal_cdf(low) if low is not None else 0.
                high_cdf = normal_cdf(high) if high is not None else 1.
                log_Z = np.log(high_cdf - low_cdf)
                log_pdf -= log_Z

                # filter out zero ranges
                filters = []
                if low is not None:
                    filters.append(low * std + mean <= given)
                if high is not None:
                    filters.append(given <= high * std + mean)
                if len(filters) > 1:
                    filters = [np.logical_and(*filters)]
                log_pdf = np.where(filters[0], log_pdf, log_zero)
            return log_pdf

        def do_test(low, high, dtype):
            # test(n_samples=n_samples)
            mean_t = T.as_tensor(mean, dtype)
            std_t = T.as_tensor(std, dtype)
            logstd_t = T.as_tensor(logstd, dtype)
            t = T.random.truncated_normal(
                mean_t, std_t, n_samples=n_samples, low=low, high=high)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])

            # test sample value range
            x = T.to_numpy(t)
            if low is not None:
                np.testing.assert_array_less(
                    (low * std + mean - 1e-7) * np.ones_like(x), x)
            if high is not None:
                np.testing.assert_array_less(
                    x, np.ones_like(x) * high * std + mean + 1e-7)

            # test log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.truncated_normal_log_pdf, mean=mean_t,
                    std=std_t, logstd=logstd_t, low=low, high=high,
                    log_zero=log_zero,
                ),
                np_log_prob=log_prob(x, low, high))
            do_check_log_prob(
                given=t * 10.,  # where the majority is out of [low, high] range
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.truncated_normal_log_pdf, mean=mean_t,
                    std=std_t, logstd=logstd_t, low=low, high=high,
                    log_zero=log_zero,
                ),
                np_log_prob=log_prob(x * 10., low, high))

            # test(n_samples=None)
            mean_t = T.as_tensor(mean, dtype)
            std_t = T.as_tensor(std, dtype)
            logstd_t = T.as_tensor(logstd, dtype)
            t = T.random.truncated_normal(mean_t, std_t, low=low, high=high)
            self.assertEqual(T.get_dtype(t), dtype)

            # test sample value range
            x = T.to_numpy(t)
            if low is not None:
                np.testing.assert_array_less(
                    (low * std + mean - 1e-7) * np.ones_like(x), x)
            if high is not None:
                np.testing.assert_array_less(
                    x, np.ones_like(x) * high * std + mean + 1e-7)

            # test log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.truncated_normal_log_pdf, mean=mean_t,
                    std=std_t, logstd=logstd_t, low=low, high=high,
                    log_zero=log_zero,
                ),
                np_log_prob=log_prob(x, low, high))
            do_check_log_prob(
                given=t * 10.,  # where the majority is out of [low, high] range
                batch_ndims=len(x.shape),
                Z_log_prob_fn=partial(
                    T.random.truncated_normal_log_pdf, mean=mean_t,
                    std=std_t, logstd=logstd_t, low=low, high=high,
                    log_zero=log_zero,
                ),
                np_log_prob=log_prob(x * 10., low, high))

            # test reparameterized
            w = np.random.randn(2, 3, 4)

            w_t = T.requires_grad(T.as_tensor(w))
            mean_t = T.requires_grad(T.as_tensor(mean, dtype))
            std_t = T.requires_grad(T.as_tensor(std, dtype))
            t = w_t * T.random.truncated_normal(mean_t, std_t)
            [mean_grad, std_grad] = T.grad(
                [t], [mean_t, std_t], [T.ones_like(t)])
            assert_allclose(mean_grad, w, rtol=1e-4)
            assert_allclose(
                std_grad,
                np.sum(T.to_numpy((t - w_t * mean_t) / std_t), axis=0),
                rtol=1e-4
            )

            # test not reparameterized
            w_t = T.requires_grad(T.as_tensor(w))
            mean_t = T.requires_grad(T.as_tensor(mean, dtype))
            std_t = T.requires_grad(T.as_tensor(std, dtype))
            t = w_t * T.random.truncated_normal(
                mean_t, std_t, reparameterized=False)
            [mean_grad, std_grad] = T.grad(
                [t], [mean_t, std_t], [T.ones_like(t)], allow_unused=True)
            self.assertTrue(T.is_null_grad(mean_t, mean_grad))
            self.assertTrue(T.is_null_grad(std_t, std_grad))

            # given has lower rank than params, broadcasted to match param
            mean_t = T.as_tensor(mean, dtype)
            std_t = T.as_tensor(std, dtype)
            logstd_t = T.as_tensor(logstd, dtype)
            assert_allclose(
                T.random.truncated_normal_log_pdf(
                    T.float_scalar(0.), mean_t, std_t, logstd_t,
                    low=low, high=high, log_zero=log_zero
                ),
                log_prob(0., low=low, high=high),
                rtol=1e-4
            )

            # dtype mismatch
            with pytest.raises(Exception, match='`mean.dtype` != `std.dtype`'):
                _ = T.random.truncated_normal(T.as_tensor(mean, T.float32),
                                              T.as_tensor(std, T.float64),
                                              low=low,
                                              high=high)

            # check numerics
            mean_t = T.as_tensor(mean)
            std_t = T.zeros_like(mean_t)
            logstd_t = T.as_tensor(T.log(std_t))
            t = T.random.normal(mean_t, std_t)
            with pytest.raises(Exception,
                               match='Infinity or NaN value encountered'):
                _ = T.random.truncated_normal_log_pdf(
                    t, mean_t, std_t, logstd_t, validate_tensors=True)

        for low, high in [(-2., 3.), (-2., None), (None, 3.), (None, None)]:
            do_test(low, high, T.float32)

        for dtype in float_dtypes:
            do_test(-2., 3., dtype)

    def test_bernoulli(self):
        def sigmoid(x):
            return np.where(
                x >= 0,
                np.exp(-np.log1p(np.exp(-x))),
                np.exp(x - np.log1p(np.exp(x)))
            )

        def log_sigmoid(x):
            return np.where(
                x >= 0,
                -np.log1p(np.exp(-x)),
                x - np.log1p(np.exp(x))
            )

        def log_prob(given):
            # return np.log(probs ** given * (1 - probs) ** (1 - given))
            # return given * np.log(probs) + (1 - given) * np.log1p(-probs)
            return (
                given * log_sigmoid(logits) +
                (1 - given) * log_sigmoid(-logits)
            )

        np.random.seed(1234)
        T.random.seed(1234)

        logits = np.random.randn(2, 3, 4)
        probs = sigmoid(logits)

        # bernoulli_logits_to_probs and bernoulli_probs_to_logits
        for float_dtype in float_dtypes:
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            probs_t = T.as_tensor(probs, dtype=float_dtype)
            assert_allclose(
                T.random.bernoulli_probs_to_logits(probs_t),
                logits,
                rtol=1e-4
            )
            assert_allclose(
                T.random.bernoulli_logits_to_probs(logits_t),
                probs,
                rtol=1e-4
            )

        t = T.random.bernoulli_probs_to_logits(T.as_tensor(np.array([0., 1.])))
        T.assert_finite(t, 'logits')

        # sample
        def do_test_sample(n_z, sample_shape, float_dtype, dtype):
            probs_t = T.as_tensor(probs, dtype=float_dtype)
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            t = T.random.bernoulli(
                probs=probs_t, n_samples=n_z, dtype=dtype)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), sample_shape + [2, 3, 4])

            # all values must be either 0 or 1
            x = T.to_numpy(t)
            self.assertEqual(set(x.flatten().tolist()), {0, 1})

            # check the log prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(t.shape),
                Z_log_prob_fn=partial(T.random.bernoulli_log_prob,
                                      logits=logits_t),
                np_log_prob=log_prob(x)
            )

        for n_z, sample_shape in [(None, []), (100, [100])]:
            do_test_sample(n_z, sample_shape, T.float32, T.int32)

        for float_dtype in float_dtypes:
            do_test_sample(n_z, sample_shape, float_dtype, T.int32)
            do_test_sample(n_z, sample_shape, float_dtype, T.int64)

        for dtype in number_dtypes:
            do_test_sample(n_z, sample_shape, T.float32, dtype)
            do_test_sample(n_z, sample_shape, T.float64, dtype)

        with pytest.raises(Exception, match='`n_samples` must be at least 1'):
            _ = T.random.bernoulli(probs=T.as_tensor(probs), n_samples=0)

        # given has lower rank than params, broadcasted to match param
        for float_dtype in float_dtypes:
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            for val in (0, 1):
                assert_allclose(
                    T.random.bernoulli_log_prob(T.int_scalar(val), logits_t),
                    log_prob(val),
                    rtol=1e-4
                )

    def test_categorical(self):
        def log_softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_diff = x - x_max
            return x_diff - np.log(
                np.sum(np.exp(x_diff), axis=axis, keepdims=True))

        def softmax(x, axis):
            return np.exp(log_softmax(x, axis))

        def one_hot(x: np.ndarray, n_classes: int):
            I = np.eye(n_classes, dtype=x.dtype)
            return I[x.astype(np.int32)]

        def log_prob(given, probs, n_classes: int, is_one_hot: bool = False):
            if not is_one_hot:
                given = one_hot(given, n_classes)
            # return np.log(np.prod(element_pow(probs, one-hot-given), axis=-1))
            return np.sum(given * np.log(probs), axis=-1)

        np.random.seed(1234)
        T.random.seed(1234)

        n_classes = 5
        logits = np.clip(np.random.randn(2, 3, 4, n_classes) / 10.,
                         a_min=-0.3, a_max=0.3)
        probs = softmax(logits, axis=-1)

        # categorical_logits_to_probs and categorical_probs_to_logits
        for dtype in float_dtypes:
            logits_t = T.as_tensor(logits, dtype=dtype)
            probs_t = T.as_tensor(probs, dtype=dtype)

            assert_allclose(
                T.random.categorical_logits_to_probs(logits_t),
                probs,
                rtol=1e-4
            )
            assert_allclose(
                T.random.categorical_probs_to_logits(probs_t),
                np.log(probs),
                rtol=1e-4
            )

        T.assert_finite(
            T.random.categorical_probs_to_logits(T.as_tensor(np.array([0., 1.]))),
            'logits'
        )

        # sample
        def do_test_sample(is_one_hot: bool,
                           n_z: Optional[int],
                           dtype: Optional[str],
                           float_dtype: str):
            probs_t = T.as_tensor(probs, dtype=float_dtype)
            logits_t = T.as_tensor(logits, dtype=float_dtype)
            value_shape = [n_classes] if is_one_hot else []

            if dtype is not None:
                expected_dtype = dtype
            else:
                expected_dtype = T.int32 if is_one_hot else T.categorical_dtype

            # sample
            sample_shape = [n_z] if n_z is not None else []
            kwargs = {'dtype': dtype} if dtype else {}
            t = (T.random.one_hot_categorical if is_one_hot
                 else T.random.categorical)(probs_t, n_samples=n_z, **kwargs)
            self.assertEqual(T.get_dtype(t), expected_dtype)
            self.assertEqual(T.shape(t), sample_shape + [2, 3, 4] + value_shape)

            # check values
            x = T.to_numpy(t)
            if is_one_hot:
                self.assertEqual(set(x.flatten().tolist()), {0, 1})
            else:
                if n_z is None:
                    self.assertTrue(set(x.flatten().tolist()).
                                    issubset(set(range(n_classes))))
                else:
                    self.assertEqual(set(x.flatten().tolist()),
                                     set(range(n_classes)))

            # check log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(t.shape) - int(is_one_hot),
                Z_log_prob_fn=partial(
                    (T.random.one_hot_categorical_log_prob
                     if is_one_hot else T.random.categorical_log_prob),
                    logits=logits_t
                ),
                np_log_prob=log_prob(x, probs, n_classes, is_one_hot)
            )

        # overall test on various arguments
        for is_one_hot in (True, False):
            for n_z in (None, 100):
                do_test_sample(is_one_hot, n_z, None, T.float32)

        for dtype in (None,) + number_dtypes:
            for float_dtype in float_dtypes:
                do_test_sample(True, 100, dtype, float_dtype)

        # sample with 2d probs
        for Z_sample_fn in (T.random.categorical, T.random.one_hot_categorical):
            is_one_hot = Z_sample_fn == T.random.one_hot_categorical
            this_probs = probs[0, 0]
            t = Z_sample_fn(
                probs=T.as_tensor(this_probs),
                n_samples=100
            )
            self.assertEqual(
                T.shape(t),
                [100, 4] + ([n_classes] if is_one_hot else [])
            )

            x = T.to_numpy(t)
            logits_t = T.as_tensor(np.log(this_probs))
            do_check_log_prob(
                given=t,
                batch_ndims=len(t.shape) - int(is_one_hot),
                Z_log_prob_fn=partial(
                    (T.random.one_hot_categorical_log_prob
                     if is_one_hot else T.random.categorical_log_prob),
                    logits=logits_t
                ),
                np_log_prob=log_prob(x, this_probs, n_classes, is_one_hot)
            )

        # given has lower rank than params, broadcasted to match param
        for is_one_hot in (False, True):
            logits_t = T.as_tensor(logits, dtype=T.float32)
            for val in range(n_classes):
                given = (one_hot(np.asarray(val), n_classes) if is_one_hot
                         else np.asarray(val))
                given_t = T.as_tensor(given)
                Z_log_prob_fn = (
                    T.random.one_hot_categorical_log_prob
                    if is_one_hot else T.random.categorical_log_prob)
                assert_allclose(
                    Z_log_prob_fn(given_t, logits_t),
                    log_prob(given, probs, n_classes, is_one_hot),
                    rtol=1e-4
                )

        # argument error
        for Z_sample_fn in (T.random.categorical, T.random.one_hot_categorical):
            with pytest.raises(Exception, match='`n_samples` must be at least 1'):
                _ = Z_sample_fn(probs=T.as_tensor(probs), n_samples=0)

            with pytest.raises(Exception, match='The rank of `probs` must be at '
                                                'least 1'):
                _ = Z_sample_fn(probs=T.as_tensor(probs[0, 0, 0, 0]))

    def test_discretized_logistic(self):
        np.random.seed(1234)
        next_seed_val = [1234]

        def next_seed():
            ret = next_seed_val[0]
            next_seed_val[0] += 1
            return ret

        def safe_sigmoid(x):
            return np.where(x < 0, np.exp(x) / (1. + np.exp(x)), 1. / (1. + np.exp(-x)))

        def do_discretize(x, bin_size, min_val=None, max_val=None):
            if min_val is not None:
                x = x - min_val
            x = np.floor(x / bin_size + .5) * bin_size
            if min_val is not None:
                x = x + min_val
            if min_val is not None:
                x = np.maximum(x, min_val)
            if max_val is not None:
                x = np.minimum(x, max_val)
            return x

        def naive_discretized_logistic_pdf(
                x, mean, log_scale, bin_size, min_val=None, max_val=None,
                biased_edges=True, discretize_given=True):
            # discretize x
            if discretize_given:
                x = do_discretize(x, bin_size, min_val, max_val)

            # middle pdfs
            half_bin = bin_size * 0.5
            x_hi = (x - mean + half_bin) / np.exp(log_scale)
            x_low = (x - mean - half_bin) / np.exp(log_scale)
            cdf_delta = safe_sigmoid(x_hi) - safe_sigmoid(x_low)
            middle_pdf = np.log(np.maximum(cdf_delta, 1e-7))
            log_prob = middle_pdf

            # left edge
            if min_val is not None and biased_edges:
                log_prob = np.where(
                    x < min_val + half_bin,
                    np.log(safe_sigmoid(x_hi)),
                    log_prob
                )

            # right edge
            if max_val is not None and biased_edges:
                log_prob = np.where(
                    x >= max_val - half_bin,
                    np.log(1. - safe_sigmoid(x_low)),
                    log_prob
                )

            # zero out prob outside of [min_val - half_bin, max_val + half_bin].
            if min_val is not None and max_val is not None:
                log_prob = np.where(
                    np.logical_and(
                        x >= min_val - half_bin,
                        x <= max_val + half_bin,
                    ),
                    log_prob,
                    T.random.LOG_ZERO_VALUE
                )
            return log_prob

        def naive_discretized_logistic_sample(
                uniform_samples, mean, log_scale, bin_size, min_val=None,
                max_val=None, discretize_sample=True):
            u = uniform_samples
            samples = mean + np.exp(log_scale) * (np.log(u) - np.log1p(-u))
            if discretize_sample:
                samples = do_discretize(samples, bin_size, min_val, max_val)
            return samples

        def get_samples(mean, log_scale, n_samples=None, **kwargs):
            seed = next_seed()
            kwargs.setdefault('epsilon', 1e-7)
            sample_shape = T.broadcast_shape(T.shape(mean), T.shape(log_scale))
            if n_samples is not None:
                sample_shape = [n_samples] + sample_shape

            np.random.seed(seed)
            T.random.seed(seed)
            u = T.random.uniform(
                shape=sample_shape, low=kwargs['epsilon'],
                high=1. - kwargs['epsilon'], dtype=T.get_dtype(mean))
            u = T.to_numpy(u)

            np.random.seed(seed)
            T.random.seed(seed)
            r = T.random.discretized_logistic(
                mean, log_scale, n_samples=n_samples, **kwargs)

            return u, r

        mean = 3 * np.random.uniform(size=[2, 1, 4]).astype(np.float64) - 1
        log_scale = np.random.normal(size=[3, 1, 5, 1]).astype(np.float64)

        # sample
        def do_test_sample(bin_size: float,
                           min_val: Optional[float],
                           max_val: Optional[float],
                           discretize_sample: bool,
                           discretize_given: bool,
                           biased_edges: bool,
                           reparameterized: bool,
                           n_samples: Optional[int],
                           validate_tensors: bool,
                           dtype: str):
            mean_t = T.as_tensor(mean, dtype=dtype)
            log_scale_t = T.as_tensor(log_scale, dtype=dtype)
            value_shape = T.broadcast_shape(T.shape(mean_t), T.shape(log_scale_t))

            # sample
            sample_shape = [n_samples] if n_samples is not None else []
            u, t = get_samples(
                mean_t, log_scale_t, n_samples=n_samples, bin_size=bin_size,
                min_val=min_val, max_val=max_val, discretize=discretize_sample,
                reparameterized=reparameterized, epsilon=T.EPSILON,
                validate_tensors=validate_tensors,
            )
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), sample_shape + value_shape)

            # check values
            this_mean = mean.astype(dtype)
            this_log_scale = log_scale.astype(dtype)
            expected_t = naive_discretized_logistic_sample(
                u, this_mean, this_log_scale, bin_size,
                min_val, max_val, discretize_sample=discretize_sample,
            )
            assert_allclose(t, expected_t, rtol=1e-4, atol=1e-6)

            # check log_prob
            do_check_log_prob(
                given=t,
                batch_ndims=len(t.shape),
                Z_log_prob_fn=partial(
                    T.random.discretized_logistic_log_prob,
                    mean=mean_t, log_scale=log_scale_t, bin_size=bin_size,
                    min_val=min_val, max_val=max_val, biased_edges=biased_edges,
                    discretize=discretize_given, validate_tensors=validate_tensors,
                ),
                np_log_prob=naive_discretized_logistic_pdf(
                    x=T.to_numpy(t), mean=this_mean, log_scale=this_log_scale,
                    bin_size=bin_size, min_val=min_val, max_val=max_val,
                    biased_edges=biased_edges, discretize_given=discretize_given,
                )
            )

        for dtype in float_dtypes:
            do_test_sample(bin_size=1 / 31., min_val=None, max_val=None,
                           discretize_sample=True, discretize_given=True,
                           biased_edges=False, reparameterized=False, n_samples=None,
                           validate_tensors=False, dtype=dtype)
            do_test_sample(bin_size=1 / 32., min_val=-3., max_val=2.,
                           discretize_sample=True, discretize_given=True,
                           biased_edges=True, reparameterized=False, n_samples=100,
                           validate_tensors=True, dtype=dtype)

        for discretize, biased_edges, validate_tensors in product(
                    [True, False],
                    [True, False],
                    [True, False],
                ):
            do_test_sample(
                bin_size=1 / 127., min_val=None, max_val=None,
                discretize_sample=discretize, discretize_given=discretize,
                biased_edges=biased_edges, reparameterized=not discretize,
                n_samples=None, validate_tensors=validate_tensors, dtype=T.float32)
            do_test_sample(
                bin_size=1 / 128., min_val=-3., max_val=2.,
                discretize_sample=discretize, discretize_given=discretize,
                biased_edges=biased_edges, reparameterized=not discretize,
                n_samples=None, validate_tensors=validate_tensors, dtype=T.float32)

        mean_t = T.as_tensor(mean, dtype=T.float32)
        log_scale_t = T.as_tensor(log_scale, dtype=T.float32)
        given_t = T.zeros(T.broadcast_shape(T.shape(mean_t), T.shape(log_scale_t)))

        with pytest.raises(Exception,
                           match='`min_val` and `max_val` must be both None or neither None'):
            _ = T.random.discretized_logistic(
                mean_t, log_scale_t, 1 / 255., min_val=-3.)

        with pytest.raises(Exception,
                           match='`min_val` and `max_val` must be both None or neither None'):
            _ = T.random.discretized_logistic(
                mean_t, log_scale_t, 1 / 255., max_val=2.)

        with pytest.raises(Exception,
                           match='`discretize` and `reparameterized` cannot be both True'):
            _ = T.random.discretized_logistic(
                mean_t, log_scale_t, 1 / 255., discretize=True, reparameterized=True)

        with pytest.raises(Exception,
                           match='`mean.dtype` != `log_scale.dtype`'):
            _ = T.random.discretized_logistic(
                mean_t, T.as_tensor(log_scale, dtype=T.float64), 1 / 255.)

        with pytest.raises(Exception,
                           match='`min_val` and `max_val` must be both None or neither None'):
            _ = T.random.discretized_logistic_log_prob(
                given_t, mean_t, log_scale_t, 1 / 255., min_val=-3.)

        with pytest.raises(Exception,
                           match='`min_val` and `max_val` must be both None or neither None'):
            _ = T.random.discretized_logistic_log_prob(
                given_t, mean_t, log_scale_t, 1 / 255., max_val=2.)

    def test_random_init(self):
        np.random.seed(1234)
        T.random.seed(1234)

        for dtype in float_dtypes:
            t = T.variable([n_samples, 2, 3], dtype=dtype)
            for fn, mean, std in [
                        (partial(T.random.normal_init, mean=1., std=2.),
                         1., 2.),
                        (partial(T.random.uniform_init, low=0., high=1.),
                         0.5, 1. / math.sqrt(12)),
                    ]:
                fn(t)
                t_mean = np.mean(T.to_numpy(t))
                self.assertLess(abs(t_mean - mean),
                                3. * std / math.sqrt(n_samples * 2 * 3))
