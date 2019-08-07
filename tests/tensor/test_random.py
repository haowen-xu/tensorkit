import unittest

import numpy as np
import pytest

from tensorkit import tensor as T

n_samples = 10000


class TensorRandomTestCase(unittest.TestCase):

    def test_seed(self):
        T.random.seed(1234)
        x = T.to_numpy(T.random.normal(0., 1.))
        y = T.to_numpy(T.random.normal(0., 1.))
        self.assertFalse(np.allclose(x, y))

        T.random.seed(1234)
        z = T.to_numpy(T.random.normal(0., 1.))
        np.testing.assert_allclose(x, z)

    def test_random_state(self):
        T.random.seed(1234)

        rs1 = T.random.new_state(1234)
        x = T.to_numpy(T.random.normal(0., 1., random_state=rs1))

        rs2 = T.random.new_state()
        y = T.to_numpy(T.random.normal(0., 1., random_state=rs2))
        self.assertFalse(np.allclose(x, y))

        rs3 = T.random.new_state(1234)
        z = T.to_numpy(T.random.normal(0., 1., random_state=rs3))
        np.testing.assert_allclose(x, z)

    def test_normal(self):
        np.random.seed(1234)
        T.random.seed(1234)

        mean = np.random.randn(2, 3, 4)
        log_std = np.random.randn(3, 4)
        std = np.exp(log_std)

        samples = []
        for rs in [None, T.random.new_state(1234), T.random.new_state(1234)]:
            x = T.to_numpy(
                T.random.normal(
                    T.expand(mean, [n_samples, 2, 3, 4]),
                    T.expand(std, [n_samples, 1, 3, 4]),
                    random_state=rs
                )
            )
            self.assertEqual(x.shape, (n_samples, 2, 3, 4))
            samples.append(x)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(x_mean - mean),
                np.tile(np.expand_dims(3 * std / np.sqrt(n_samples), axis=0),
                        [2, 1, 1])
            )

        # validate whether or not random state takes effect.
        # do not validate the first sample; not all backend ensure to output
        # the same samples with identical global seed and random state seed.
        np.testing.assert_allclose(*samples[-2:])

    def test_randn(self):
        np.random.seed(1234)
        T.random.seed(1234)

        samples = []
        for rs in [None, T.random.new_state(1234), T.random.new_state(1234)]:
            t = T.random.randn(
                [n_samples, 2, 3, 4],
                dtype=T.float64,
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.float64)
            self.assertEqual(tuple(T.shape(t)), (n_samples, 2, 3, 4))
            x = T.to_numpy(t)
            samples.append(x)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(x_mean),
                3. / np.sqrt(n_samples) * np.ones_like(x_mean)
            )

        np.testing.assert_allclose(*samples[-2:])

    def test_bernoulli(self):
        def sigmoid(x):
            return np.where(
                x >= 0,
                np.exp(-np.log1p(np.exp(-x))),
                np.exp(x - np.log1p(np.exp(x)))
            )

        np.random.seed(1234)
        T.random.seed(1234)

        logits = np.clip(np.random.randn(2, 3, 4) / 10., a_min=-0.3, a_max=0.3)
        probs = sigmoid(logits)

        std = np.exp(
            np.where(
                logits >= 0,
                -logits - 2 * np.log1p(np.exp(-logits)),
                logits - 2 * np.log1p(np.exp(logits)),
            )
        )

        samples_logits = []
        samples_probs = []

        for rs in [None, T.random.new_state(1234), T.random.new_state(1234)]:
            # test sample with logits
            t = T.random.bernoulli(
                logits=T.expand(logits, [n_samples, 2, 3, 4]),
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.int32)
            self.assertEqual(tuple(T.shape(t)), (n_samples, 2, 3, 4))
            x = T.to_numpy(t)
            samples_logits.append(x)
            x_mean = np.mean(x, axis=0)
            np.testing.assert_array_less(
                np.abs(x_mean - probs),
                4 * std / np.sqrt(n_samples)
            )

            # test sample with probs
            t = T.random.bernoulli(
                probs=T.expand(probs, [2, 3, 4]),
                n_samples=n_samples,
                dtype=T.int64,
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.int64)
            self.assertEqual(tuple(T.shape(t)), (2, 3, 4, n_samples))
            x = T.to_numpy(t)
            samples_probs.append(x)
            x_mean = np.mean(x, axis=-1)
            np.testing.assert_array_less(
                np.abs(x_mean - probs),
                4 * std / np.sqrt(n_samples)
            )

        np.testing.assert_allclose(*samples_logits[-2:])
        np.testing.assert_allclose(*samples_probs[-2:])

        with pytest.raises(ValueError,
                           match='Either `logits` or `probs` must be '
                                 'specified, but not both'):
            _ = T.random.bernoulli(logits=logits, probs=probs)

        with pytest.raises(ValueError,
                           match='`n_samples` must be at least 1: got 0'):
            _ = T.random.bernoulli(logits=logits, n_samples=0)

    def test_categorical(self):
        def log_softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_diff = x - x_max
            return x_diff - np.log(
                np.sum(np.exp(x_diff), axis=axis, keepdims=True))

        def softmax(x, axis):
            return np.exp(log_softmax(x, axis))

        np.random.seed(1234)
        T.random.seed(1234)

        logits = np.clip(
            np.random.randn(2, 3, 4, 5) / 10., a_min=-0.3, a_max=0.3)
        probs = softmax(logits, axis=-1)

        samples_logits = []
        samples_logits_2d = []
        samples_probs = []
        samples_probs_2d = []

        for rs in [None, T.random.new_state(1234), T.random.new_state(1234)]:
            # test sample with logits
            t = T.random.categorical(
                logits=T.expand(logits, [n_samples, 2, 3, 4, 5]),
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.random.CATEGORICAL_DTYPE)
            self.assertEqual(tuple(T.shape(t)), (n_samples, 2, 3, 4))
            x = T.to_numpy(t)
            samples_logits.append(x)

            # test sample with 2d logits
            t = T.random.categorical(
                logits=logits[0, 0],
                dtype=T.int16,
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.int16)
            self.assertEqual(tuple(T.shape(t)), (4,))
            x = T.to_numpy(t)
            samples_logits_2d.append(x)

            # test sample with probs
            t = T.random.categorical(
                probs=T.expand(probs, [2, 3, 4, 5]),
                n_samples=n_samples,
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.random.CATEGORICAL_DTYPE)
            self.assertEqual(tuple(T.shape(t)), (2, 3, 4, n_samples))
            x = T.to_numpy(t)
            samples_probs.append(x)

            # test sample with 2d probs
            t = T.random.categorical(
                probs=probs[0, 0],
                dtype=T.int16,
                random_state=rs
            )
            self.assertEqual(T.dtype(t), T.int16)
            self.assertEqual(tuple(T.shape(t)), (4,))
            x = T.to_numpy(t)
            samples_probs_2d.append(x)

        np.testing.assert_allclose(*samples_logits[-2:])
        np.testing.assert_allclose(*samples_logits_2d[-2:])
        np.testing.assert_allclose(*samples_probs[-2:])
        np.testing.assert_allclose(*samples_probs_2d[-2:])

        with pytest.raises(ValueError,
                           match='Either `logits` or `probs` must be '
                                 'specified, but not both'):
            _ = T.random.categorical(logits=logits, probs=probs)

        with pytest.raises(ValueError,
                           match='`n_samples` must be at least 1: got 0'):
            _ = T.random.categorical(logits=logits, n_samples=0)

        with pytest.raises(ValueError,
                           match='The rank of `logits` or `probs` must be at '
                                 'least 1: got 0'):
            _ = T.random.categorical(logits=logits[0, 0, 0, 0])
