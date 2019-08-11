import unittest

import numpy as np
import pytest

from tensorkit import tensor as T

n_samples = 10000


class TensorRandomTestCase(unittest.TestCase):

    def test_seed(self):
        T.random_seed(1234)
        x = T.to_numpy(T.random_normal(T.as_tensor(0.), T.as_tensor(1.)))
        y = T.to_numpy(T.random_normal(T.as_tensor(0.), T.as_tensor(1.)))
        self.assertFalse(np.allclose(x, y))

        T.random_seed(1234)
        z = T.to_numpy(T.random_normal(T.as_tensor(0.), T.as_tensor(1.)))
        np.testing.assert_allclose(x, z)

    def test_normal(self):
        np.random.seed(1234)
        T.random_seed(1234)

        mean = np.random.randn(2, 3, 4)
        log_std = np.random.randn(3, 4)
        std = np.exp(log_std)

        t = T.random_normal(
            T.cast(T.expand(T.as_tensor(mean), [n_samples, 2, 3, 4]),
                   T.float64),
            T.cast(T.expand(T.as_tensor(std), [n_samples, 1, 3, 4]), T.float64)
        )
        x = T.to_numpy(t)
        self.assertEqual(T.dtype(t), T.float64)
        self.assertEqual(x.shape, (n_samples, 2, 3, 4))
        x_mean = np.mean(x, axis=0)
        np.testing.assert_array_less(
            np.abs(x_mean - mean),
            np.tile(np.expand_dims(3 * std / np.sqrt(n_samples), axis=0),
                    [2, 1, 1])
        )

        t = T.random_normal(
            T.cast(T.as_tensor(mean), T.float32),
            T.cast(T.as_tensor(std), T.float32)
        )
        self.assertEqual(T.dtype(t), T.float32)

    def test_randn(self):
        np.random.seed(1234)
        T.random_seed(1234)

        t = T.randn(
            [n_samples, 2, 3, 4],
            dtype=T.float64
        )
        self.assertEqual(T.dtype(t), T.float64)
        self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])
        x = T.to_numpy(t)
        x_mean = np.mean(x, axis=0)
        np.testing.assert_array_less(
            np.abs(x_mean),
            3. / np.sqrt(n_samples) * np.ones_like(x_mean)
        )

    def test_bernoulli(self):
        def sigmoid(x):
            return np.where(
                x >= 0,
                np.exp(-np.log1p(np.exp(-x))),
                np.exp(x - np.log1p(np.exp(x)))
            )

        np.random.seed(1234)
        T.random_seed(1234)

        logits = np.random.randn(2, 3, 4)
        probs = sigmoid(logits)

        # test sample with logits
        t = T.bernoulli(
            logits=T.expand(T.as_tensor(logits), [n_samples, 2, 3, 4]))
        self.assertEqual(T.dtype(t), T.int32)
        self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])
        x = T.to_numpy(t)
        self.assertEqual(set(x.flatten().tolist()), {0, 1})

        # test sample with probs
        t = T.bernoulli(
            probs=T.expand(T.as_tensor(probs), [2, 3, 4]),
            n_samples=n_samples,
            dtype=T.int64
        )
        self.assertEqual(T.dtype(t), T.int64)
        self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])
        x = T.to_numpy(t)
        self.assertEqual(set(x.flatten().tolist()), {0, 1})

        with pytest.raises(Exception, match='Either `logits` or `probs` must '
                                            'be specified, but not both'):
            _ = T.bernoulli(logits=T.as_tensor(logits),
                            probs=T.as_tensor(probs))

        with pytest.raises(Exception, match='`n_samples` must be at least 1'):
            _ = T.bernoulli(logits=T.as_tensor(logits), n_samples=0)

    def test_categorical(self):
        def log_softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_diff = x - x_max
            return x_diff - np.log(
                np.sum(np.exp(x_diff), axis=axis, keepdims=True))

        def softmax(x, axis):
            return np.exp(log_softmax(x, axis))

        np.random.seed(1234)
        T.random_seed(1234)

        logits = np.clip(
            np.random.randn(2, 3, 4, 5) / 10., a_min=-0.3, a_max=0.3)
        probs = softmax(logits, axis=-1)

        # test sample with logits
        t = T.categorical(
            logits=T.expand(T.as_tensor(logits), [n_samples, 2, 3, 4, 5]))
        self.assertEqual(T.dtype(t), T.index_dtype)
        self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])
        x = T.to_numpy(t)
        self.assertEqual(set(x.flatten().tolist()), set(range(5)))

        # test sample with 2d logits
        t = T.categorical(
            logits=T.as_tensor(logits[0, 0]),
            dtype=T.int16
        )
        self.assertEqual(T.dtype(t), T.int16)
        self.assertEqual(T.shape(t), [4])

        # test sample with probs
        t = T.categorical(
            probs=T.expand(T.as_tensor(probs), [2, 3, 4, 5]),
            n_samples=n_samples
        )
        self.assertEqual(T.dtype(t), T.index_dtype)
        self.assertEqual(T.shape(t), [n_samples, 2, 3, 4])
        x = T.to_numpy(t)
        self.assertEqual(set(x.flatten().tolist()), set(range(5)))

        # test sample with 2d probs
        t = T.categorical(
            probs=T.as_tensor(probs[0, 0]),
            dtype=T.int16,
            n_samples=n_samples
        )
        self.assertEqual(T.dtype(t), T.int16)
        self.assertEqual(T.shape(t), [n_samples, 4])

        with pytest.raises(Exception, match='Either `logits` or `probs` must '
                                            'be specified, but not both'):
            _ = T.categorical(logits=T.as_tensor(logits),
                              probs=T.as_tensor(probs))

        with pytest.raises(Exception, match='`n_samples` must be at least 1'):
            _ = T.categorical(logits=T.as_tensor(logits), n_samples=0)

        with pytest.raises(Exception, match='The rank of `logits` or `probs` '
                                            'must be at least 1'):
            _ = T.categorical(logits=T.as_tensor(logits[0, 0, 0, 0]))
