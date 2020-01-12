import unittest

import numpy as np
import pytest

from tensorkit import tensor as T

assert_allclose = np.testing.assert_allclose


class TensorNNTestCase(unittest.TestCase):

    def test_activation_functions(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3, 4)
        x = np.concatenate([x, np.zeros([2, 3, 1])], axis=-1)
        self.assertTrue(np.any(x < 0))
        self.assertTrue(np.any(x > 0))
        self.assertTrue(np.any(x == 0))

        # test relu
        np.testing.assert_allclose(
            T.to_numpy(T.nn.relu(T.as_tensor_jit(x))),
            x * (x >= 0)
        )

        # test leaky_relu
        np.testing.assert_allclose(
            T.to_numpy(T.nn.leaky_relu(T.as_tensor_jit(x))),
            x * (x >= 0) + (0.01 * x * (x < 0))
        )
        np.testing.assert_allclose(
            T.to_numpy(T.nn.leaky_relu(T.as_tensor_jit(x), a=0.02)),
            x * (x >= 0) + (0.02 * x * (x < 0))
        )

        # test sigmoid
        np.testing.assert_allclose(
            T.to_numpy(T.nn.sigmoid(T.as_tensor_jit(x))),
            np.where(x >= 0, 1. / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        )

        # test softmax
        def softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_exp = np.exp(x - x_max)
            return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

        for axis in [-3, -2, -1, 0, 1, 2]:
            np.testing.assert_allclose(
                T.to_numpy(T.nn.softmax(T.as_tensor_jit(x), axis=axis)),
                softmax(x, axis=axis)
            )

        # test log_softmax
        def log_softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_diff = x - x_max
            return x_diff - np.log(
                np.sum(np.exp(x_diff), axis=axis, keepdims=True))

        for axis in [-3, -2, -1, 0, 1, 2]:
            np.testing.assert_allclose(
                T.to_numpy(T.nn.log_softmax(T.as_tensor_jit(x), axis=axis)),
                log_softmax(x, axis=axis)
            )

    def test_binary_cross_entropy(self):
        def sigmoid(x):
            return np.where(x >= 0, 1. / (1 + np.exp(-x)),
                            np.exp(x) / (1 + np.exp(x)))

        def binary_cross_entropy(logits, labels, reduction, negative):
            # for logits >= 0:
            #   p = 1 / (1 + exp(-logits))
            #   log p = -log(1 + exp(-logits))
            #   log (1-p) = -logits - log(1 + exp(-logits))

            # for logits < 0:
            #   p = exp(logits) / (1 + exp(logits))
            #   log p = logits - log(1 + exp(logits))
            #   log (1-p) = -log(1 + exp(logits))

            log_p = np.where(
                logits >= 0,
                -np.log1p(np.exp(-logits)),
                logits - np.log1p(np.exp(logits))
            )
            log_1_minus_p = np.where(
                logits >= 0,
                -logits - np.log1p(np.exp(-logits)),
                -np.log1p(np.exp(logits))
            )

            out = labels * log_p + (1 - labels) * log_1_minus_p
            if reduction == 'mean':
                out = np.average(out)
            elif reduction == 'sum':
                out = np.sum(out)
            else:
                assert(reduction == 'none')

            if not negative:
                out = -out
            return out

        np.random.seed(1234)

        logits = np.random.randn(2, 3, 4)
        sparse_labels = sigmoid(np.random.randn(3, 4))
        labels = (sparse_labels < 0.5).astype(np.int32)
        self.assertEqual(sparse_labels.shape, (3, 4))
        self.assertTrue(np.any((sparse_labels < 1 - 1e-4) &
                               (sparse_labels > 1e-4)))
        self.assertEqual(labels.shape, (3, 4))
        self.assertEqual(set(labels.flatten().tolist()), {0, 1})

        _f = T.as_tensor_jit

        for reduction in ['none', 'mean', 'sum']:
            for negative in [False, True]:
                # test integer labels
                ans = binary_cross_entropy(logits, labels, reduction, negative)
                out = T.nn.binary_cross_entropy_with_logits(
                    _f(logits), _f(labels), reduction, negative)
                np.testing.assert_allclose(ans, T.to_numpy(out))

                # test sparse labels (floating point labels)
                ans = binary_cross_entropy(
                    logits, sparse_labels, reduction, negative)
                out = T.nn.binary_cross_entropy_with_logits(
                    _f(logits), _f(sparse_labels), reduction, negative)
                np.testing.assert_allclose(ans, T.to_numpy(out))

        # invalid `reduction` argument should raise error
        with pytest.raises(Exception):
            _ = T.nn.binary_cross_entropy_with_logits(
                _f(logits), _f(labels), 'invalid')

        # validation for the shape of logits and labels
        with pytest.raises(Exception):
            # logits and labels shape mismatch
            _ = T.nn.binary_cross_entropy_with_logits(
                _f(logits), _f(labels[:-1]))

    def test_cross_entropy(self):
        def softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_exp = np.exp(x - x_max)
            return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

        def sparse_cross_entropy(logits, labels, reduction, negative):
            logits_max = np.max(logits, axis=-1, keepdims=True)
            logits_max_reduced = np.squeeze(logits_max, axis=-1)
            out = np.sum(logits * labels, axis=-1) - logits_max_reduced
            out -= np.log(np.sum(np.exp(logits - logits_max), axis=-1))

            if reduction == 'sum':
                out = np.sum(out)
            elif reduction == 'mean':
                out = np.mean(out)
            else:
                assert(reduction == 'none')

            if not negative:
                out = -out
            return out

        def cross_entropy(logits, labels, reduction, negative):
            k = logits.shape[-1]
            sparse_labels = np.eye(k, dtype=logits.dtype)[labels]
            return sparse_cross_entropy(
                logits, sparse_labels, reduction, negative)

        np.random.seed(1234)

        logits = np.random.randn(2, 3, 4, 5, 6)
        sparse_labels = softmax(np.random.randn(3, 4, 5, 6), axis=-1)
        labels = np.argmax(sparse_labels, axis=-1)

        self.assertEqual(sparse_labels.shape, (3, 4, 5, 6))
        self.assertEqual(labels.shape, (3, 4, 5))
        self.assertEqual(set(labels.flatten().tolist()), {0, 1, 2, 3, 4, 5})

        _f = T.as_tensor_jit

        for reduction in ['none', 'mean', 'sum']:
            for negative in [False, True]:
                # test cross_entropy
                ans = cross_entropy(logits, labels, reduction, negative)
                out = T.nn.cross_entropy_with_logits(
                    _f(logits), _f(labels), reduction, negative)
                np.testing.assert_allclose(ans, T.to_numpy(out))

                # test cross_entropy on 2d
                ans = cross_entropy(
                    logits[0, 0, 0], labels[0, 0], reduction, negative)
                out = T.nn.cross_entropy_with_logits(
                    _f(logits[0, 0, 0]), _f(labels[0, 0]), reduction, negative)
                np.testing.assert_allclose(ans, T.to_numpy(out))

                # test sparse_cross_entropy
                ans = sparse_cross_entropy(
                    logits, sparse_labels, reduction, negative)
                out = T.nn.sparse_cross_entropy_with_logits(
                    _f(logits), _f(sparse_labels), reduction, negative)
                np.testing.assert_allclose(ans, T.to_numpy(out))

                # test sparse_cross_entropy on 2d
                ans = sparse_cross_entropy(
                    logits[0, 0, 0], sparse_labels[0, 0], reduction, negative)
                out = T.nn.sparse_cross_entropy_with_logits(
                    _f(logits[0, 0, 0]), _f(sparse_labels[0, 0]),
                    reduction, negative
                )
                np.testing.assert_allclose(ans, T.to_numpy(out))

        # invalid `reduction` argument should raise error
        with pytest.raises(Exception):
            _ = T.nn.cross_entropy_with_logits(
                _f(logits), _f(labels), 'invalid')

        with pytest.raises(Exception):
            _ = T.nn.sparse_cross_entropy_with_logits(
                _f(logits), _f(labels), 'invalid')

        # validation for the shape of logits and labels
        with pytest.raises(Exception):
            # logits and labels shape mismatch
            _ = T.nn.cross_entropy_with_logits(_f(logits), _f(labels[:-1]))

        with pytest.raises(Exception):
            # logits rank too low
            _ = T.nn.sparse_cross_entropy_with_logits(_f(logits[0, 0, 0, 0]),
                                                      _f(labels))

        with pytest.raises(Exception):
            # logits and labels shape mismatch
            _ = T.nn.sparse_cross_entropy_with_logits(
                _f(logits), _f(labels[:-1]))
