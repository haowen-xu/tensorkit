import unittest
from functools import partial
from typing import *

import numpy as np
import pytest
from scipy.stats import norm

from tensorkit import tensor as T
from tests.helper import *


def do_check_log_prob(given, batch_ndims, Z_log_prob_fn, np_log_prob):
    # test log_prob
    for group_ndims in range(0, batch_ndims + 1):
        np.testing.assert_allclose(
            T.to_numpy(Z_log_prob_fn(given, group_ndims=group_ndims)),
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
        np.testing.assert_allclose(x, z)

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
        self.assertEqual(
            T.to_numpy(T.random.truncated_randn_log_pdf(
                T.from_numpy(np.array(10000.0)),
                low=9999.0,
                high=10001.0,
            )),
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
            mean_t = T.from_numpy(mean, dtype)
            std_t = T.from_numpy(std, dtype)
            logstd_t = T.from_numpy(logstd, dtype)
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
            mean_t = T.from_numpy(mean, dtype)
            std_t = T.from_numpy(std, dtype)
            logstd_t = T.from_numpy(logstd, dtype)
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
            w_t = T.requires_grad(T.from_numpy(w))
            mean_t = T.requires_grad(T.from_numpy(mean, dtype))
            std_t = T.requires_grad(T.from_numpy(std, dtype))
            t = w_t * T.random.normal(mean_t, std_t)
            [mean_grad, std_grad] = T.grad(
                [t], [mean_t, std_t], [T.ones_like(t)])
            np.testing.assert_allclose(T.to_numpy(mean_grad), w, rtol=1e-4)
            np.testing.assert_allclose(
                T.to_numpy(std_grad),
                np.sum(T.to_numpy((t - w_t * mean_t) / std_t), axis=0),
                rtol=1e-4
            )

        # test not reparameterized
        for dtype in float_dtypes:
            w_t = T.requires_grad(T.from_numpy(w))
            mean_t = T.requires_grad(T.from_numpy(mean, dtype))
            std_t = T.requires_grad(T.from_numpy(std, dtype))
            t = w_t * T.random.normal(mean_t, std_t, reparameterized=False)
            [mean_grad, std_grad] = T.grad(
                [t], [mean_t, std_t], [T.ones_like(t)], allow_unused=True)
            self.assertTrue(T.is_null_grad(mean_t, mean_grad))
            self.assertTrue(T.is_null_grad(std_t, std_grad))

        # given has lower rank than params, broadcasted to match param
        for dtype in float_dtypes:
            mean_t = T.from_numpy(mean, dtype)
            logstd_t = T.from_numpy(logstd, dtype)
            for val in (0., 1., -1.):
                np.testing.assert_allclose(
                    T.to_numpy(T.random.normal_log_pdf(
                        T.float_scalar(val), mean_t, logstd_t)),
                    log_prob(val),
                    rtol=1e-4
                )

        # dtype mismatch
        with pytest.raises(Exception, match='`mean.dtype` != `std.dtype`'):
            _ = T.random.normal(T.from_numpy(mean, T.float32),
                                T.from_numpy(std, T.float64))

        # check numerics
        mean_t = T.from_numpy(mean)
        std_t = T.zeros_like(mean_t)
        logstd_t = T.from_numpy(T.log(std_t))
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

        for low, high in [(-2., 3.), (-2., None), (None, 3.), (None, None)]:
            # test(n_samples=n_samples)
            for dtype in float_dtypes:
                # test sample dtype and shape
                mean_t = T.from_numpy(mean, dtype)
                std_t = T.from_numpy(std, dtype)
                logstd_t = T.from_numpy(logstd, dtype)
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
            for dtype in float_dtypes:
                mean_t = T.from_numpy(mean, dtype)
                std_t = T.from_numpy(std, dtype)
                logstd_t = T.from_numpy(logstd, dtype)
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

            for dtype in float_dtypes:
                w_t = T.requires_grad(T.from_numpy(w))
                mean_t = T.requires_grad(T.from_numpy(mean, dtype))
                std_t = T.requires_grad(T.from_numpy(std, dtype))
                t = w_t * T.random.truncated_normal(mean_t, std_t)
                [mean_grad, std_grad] = T.grad(
                    [t], [mean_t, std_t], [T.ones_like(t)])
                np.testing.assert_allclose(T.to_numpy(mean_grad), w, rtol=1e-4)
                np.testing.assert_allclose(
                    T.to_numpy(std_grad),
                    np.sum(T.to_numpy((t - w_t * mean_t) / std_t), axis=0),
                    rtol=1e-4
                )

            # test not reparameterized
            for dtype in float_dtypes:
                w_t = T.requires_grad(T.from_numpy(w))
                mean_t = T.requires_grad(T.from_numpy(mean, dtype))
                std_t = T.requires_grad(T.from_numpy(std, dtype))
                t = w_t * T.random.truncated_normal(
                    mean_t, std_t, reparameterized=False)
                [mean_grad, std_grad] = T.grad(
                    [t], [mean_t, std_t], [T.ones_like(t)], allow_unused=True)
                self.assertTrue(T.is_null_grad(mean_t, mean_grad))
                self.assertTrue(T.is_null_grad(std_t, std_grad))

            # given has lower rank than params, broadcasted to match param
            for dtype in float_dtypes:
                mean_t = T.from_numpy(mean, dtype)
                std_t = T.from_numpy(std, dtype)
                logstd_t = T.from_numpy(logstd, dtype)
                np.testing.assert_allclose(
                    T.to_numpy(T.random.truncated_normal_log_pdf(
                        T.float_scalar(0.), mean_t, std_t, logstd_t,
                        low=low, high=high, log_zero=log_zero
                    )),
                    log_prob(0., low=low, high=high),
                    rtol=1e-4
                )

            # dtype mismatch
            with pytest.raises(Exception, match='`mean.dtype` != `std.dtype`'):
                _ = T.random.truncated_normal(T.from_numpy(mean, T.float32),
                                              T.from_numpy(std, T.float64),
                                              low=low,
                                              high=high)

            # check numerics
            mean_t = T.from_numpy(mean)
            std_t = T.zeros_like(mean_t)
            logstd_t = T.from_numpy(T.log(std_t))
            t = T.random.normal(mean_t, std_t)
            with pytest.raises(Exception,
                               match='Infinity or NaN value encountered'):
                _ = T.random.truncated_normal_log_pdf(
                    t, mean_t, std_t, logstd_t, validate_tensors=True)

    def test_bernoulli(self):
        def sigmoid(x):
            return np.where(
                x >= 0,
                np.exp(-np.log1p(np.exp(-x))),
                np.exp(x - np.log1p(np.exp(x)))
            )

        def log_prob(given):
            # return np.log(probs ** given * (1 - probs) ** (1 - given))
            return given * np.log(probs) + (1 - given) * np.log1p(-probs)

        np.random.seed(1234)
        T.random.seed(1234)

        logits = np.random.randn(2, 3, 4)
        probs = sigmoid(logits)

        # bernoulli_logits_to_probs and bernoulli_probs_to_logits
        for float_dtype in float_dtypes:
            logits_t = T.from_numpy(logits, dtype=float_dtype)
            probs_t = T.from_numpy(probs, dtype=float_dtype)
            np.testing.assert_allclose(
                T.to_numpy(T.random.bernoulli_probs_to_logits(probs_t)),
                logits,
                rtol=1e-4
            )
            np.testing.assert_allclose(
                T.to_numpy(T.random.bernoulli_logits_to_probs(logits_t)),
                probs,
                rtol=1e-4
            )

        t = T.random.bernoulli_probs_to_logits(T.from_numpy(np.array([0., 1.])))
        T.assert_finite(t, 'logits')

        # sample
        for float_dtype in float_dtypes:
            probs_t = T.from_numpy(probs, dtype=float_dtype)
            logits_t = T.from_numpy(logits, dtype=float_dtype)
            for n_z, sample_shape in [(None, []), (n_samples, [n_samples])]:
                for dtype in number_dtypes:
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

        with pytest.raises(Exception, match='`n_samples` must be at least 1'):
            _ = T.random.bernoulli(probs=T.as_tensor(probs), n_samples=0)

        # given has lower rank than params, broadcasted to match param
        for float_dtype in float_dtypes:
            logits_t = T.from_numpy(logits, dtype=float_dtype)
            for val in (0, 1):
                np.testing.assert_allclose(
                    T.to_numpy(T.random.bernoulli_log_prob(
                        T.int_scalar(val), logits_t)),
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
            logits_t = T.from_numpy(logits, dtype=dtype)
            probs_t = T.from_numpy(probs, dtype=dtype)

            np.testing.assert_allclose(
                T.to_numpy(T.random.categorical_logits_to_probs(logits_t)),
                probs,
                rtol=1e-4
            )
            np.testing.assert_allclose(
                T.to_numpy(T.random.categorical_probs_to_logits(probs_t)),
                np.log(probs),
                rtol=1e-4
            )

        T.assert_finite(
            T.random.categorical_probs_to_logits(
                T.from_numpy(np.array([0., 1.]))),
            'logits'
        )

        # sample
        def do_test_sample(is_one_hot: bool,
                           n_z: Optional[int],
                           sample_shape: List[int],
                           dtype: Optional[str],
                           float_dtype: str):
            probs_t = T.from_numpy(probs, dtype=float_dtype)
            logits_t = T.from_numpy(logits, dtype=float_dtype)
            value_shape = [n_classes] if is_one_hot else []

            if dtype is not None:
                expected_dtype = dtype
            else:
                expected_dtype = T.int32 if is_one_hot else T.categorical_dtype

            # sample
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

        for is_one_hot in (True, False):
            for n_z, sample_shape in [(None, []), (100, [100])]:
                for dtype in (None,) + number_dtypes:
                    for float_dtype in float_dtypes:
                        do_test_sample(is_one_hot, n_z, sample_shape, dtype,
                                       float_dtype)

        # sample with 2d probs
        for Z_sample_fn in (T.random.categorical, T.random.one_hot_categorical):
            is_one_hot = Z_sample_fn == T.random.one_hot_categorical
            this_probs = probs[0, 0]
            t = Z_sample_fn(
                probs=T.as_tensor(this_probs),
                n_samples=n_samples
            )
            self.assertEqual(
                T.shape(t),
                [n_samples, 4] + ([n_classes] if is_one_hot else [])
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
            for float_dtype in float_dtypes:
                logits_t = T.from_numpy(logits, dtype=float_dtype)
                for val in range(n_classes):
                    given = (one_hot(np.asarray(val), n_classes) if is_one_hot
                             else np.asarray(val))
                    given_t = T.from_numpy(given)
                    Z_log_prob_fn = (
                        T.random.one_hot_categorical_log_prob
                        if is_one_hot else T.random.categorical_log_prob)
                    np.testing.assert_allclose(
                        T.to_numpy(Z_log_prob_fn(given_t, logits_t)),
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
