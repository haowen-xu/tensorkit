import unittest
from itertools import product

import numpy as np
import pytest

from tensorkit import tensor as T
from tests import ops
from tests.helper import *
from tests.ops import *


class TensorNNTestCase(unittest.TestCase):

    def test_constants(self):
        self.assertEqual(T.nn.LEAKY_RELU_DEFAULT_SLOPE, 0.01)
        self.assertFalse(T.nn.AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS)

    def test_activation_functions(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3, 4)
        x = np.concatenate([x, np.zeros([2, 3, 1])], axis=-1)
        self.assertTrue(np.any(x < 0))
        self.assertTrue(np.any(x > 0))
        self.assertTrue(np.any(x == 0))
        x_t = T.as_tensor_backend(x)

        # test relu
        assert_allclose(T.nn.relu(x_t), x * (x >= 0))

        # test leaky_relu
        assert_allclose(
            T.nn.leaky_relu(x_t),
            x * (x >= 0) + (T.nn.LEAKY_RELU_DEFAULT_SLOPE * x * (x < 0))
        )
        assert_allclose(
            T.nn.leaky_relu(x_t, negative_slope=0.2),
            x * (x >= 0) + (0.2 * x * (x < 0))
        )

        # test sigmoid
        assert_allclose(
            T.nn.sigmoid(x_t),
            np.where(x >= 0, 1. / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
        )

        # test log_sigmoid
        assert_allclose(
            T.nn.log_sigmoid(x_t),
            np.where(x >= 0, - np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))
        )

        # test softmax
        def softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_exp = np.exp(x - x_max)
            return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

        for axis in [-3, -2, -1, 0, 1, 2]:
            assert_allclose(T.nn.softmax(x_t, axis=axis), softmax(x, axis=axis))

        # test log_softmax
        def log_softmax(x, axis):
            x_max = np.max(x, axis=axis, keepdims=True)
            x_diff = x - x_max
            return x_diff - np.log(
                np.sum(np.exp(x_diff), axis=axis, keepdims=True))

        for axis in [-3, -2, -1, 0, 1, 2]:
            assert_allclose(
                T.nn.log_softmax(x_t, axis=axis),
                log_softmax(x, axis=axis)
            )

        # test softplus
        assert_allclose(T.nn.softplus(x_t), np.log1p(np.exp(x)))

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

        _f = T.as_tensor_backend

        for reduction in ['none', 'mean', 'sum']:
            for negative in [False, True]:
                # test integer labels
                ans = binary_cross_entropy(logits, labels, reduction, negative)
                out = T.nn.binary_cross_entropy_with_logits(
                    _f(logits), _f(labels), reduction, negative)
                assert_allclose(ans, out)

                # test sparse labels (floating point labels)
                ans = binary_cross_entropy(
                    logits, sparse_labels, reduction, negative)
                out = T.nn.binary_cross_entropy_with_logits(
                    _f(logits), _f(sparse_labels), reduction, negative)
                assert_allclose(ans, out)

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

        _f = T.as_tensor_backend

        for reduction in ['none', 'mean', 'sum']:
            for negative in [False, True]:
                # test cross_entropy
                ans = cross_entropy(logits, labels, reduction, negative)
                out = T.nn.cross_entropy_with_logits(
                    _f(logits), _f(labels), reduction, negative)
                assert_allclose(ans, out)

                # test cross_entropy on 2d
                ans = cross_entropy(
                    logits[0, 0, 0], labels[0, 0], reduction, negative)
                out = T.nn.cross_entropy_with_logits(
                    _f(logits[0, 0, 0]), _f(labels[0, 0]), reduction, negative)
                assert_allclose(ans, out)

                # test sparse_cross_entropy
                ans = sparse_cross_entropy(
                    logits, sparse_labels, reduction, negative)
                out = T.nn.sparse_cross_entropy_with_logits(
                    _f(logits), _f(sparse_labels), reduction, negative)
                assert_allclose(ans, out)

                # test sparse_cross_entropy on 2d
                ans = sparse_cross_entropy(
                    logits[0, 0, 0], sparse_labels[0, 0], reduction, negative)
                out = T.nn.sparse_cross_entropy_with_logits(
                    _f(logits[0, 0, 0]), _f(sparse_labels[0, 0]),
                    reduction, negative
                )
                assert_allclose(ans, out)

        # invalid `reduction` argument should raise error
        with pytest.raises(Exception):
            _ = T.nn.cross_entropy_with_logits(
                _f(logits), _f(labels), 'invalid')

        with pytest.raises(Exception):
            _ = T.nn.sparse_cross_entropy_with_logits(
                _f(logits), _f(labels), 'invalid')

        # validation for the shape of logits and labels
        with pytest.raises(Exception, match='cannot broadcast'):
            # logits and labels shape mismatch
            _ = T.nn.cross_entropy_with_logits(_f(logits), _f(labels[..., :-1]))

        with pytest.raises(Exception, match='must be at least 2d'):
            # logits and labels rank too low
            _ = T.nn.cross_entropy_with_logits(_f(logits[0, 0, 0, 0]),
                                               _f(labels[0, 0, 0]))

        with pytest.raises(Exception, match='cannot broadcast'):
            # logits and labels shape mismatch
            _ = T.nn.sparse_cross_entropy_with_logits(_f(logits[..., :-1]),
                                                      _f(sparse_labels))

        with pytest.raises(Exception, match='must be at least 2d'):
            # logits and labels rank too low
            _ = T.nn.sparse_cross_entropy_with_logits(_f(logits[0, 0, 0, 0]),
                                                      _f(sparse_labels[0, 0, 0, 0]))

    def test_conv_shape_utils(self):
        # channels_to_last
        for spatial_ndims in (1, 2, 3):
            T_fn_name = f'channel_first_to_last{spatial_ndims}d'
            T_fn = getattr(T.nn, T_fn_name)
            for ndims in range(spatial_ndims + 1, spatial_ndims + 2):
                x = np.random.randn(*range(3, 3 + ndims))
                T_ret = T_fn(T.as_tensor(x))
                assert_equal(
                    T_ret,
                    channel_to_last_nd(x, spatial_ndims),
                    err_msg=f'{T_fn_name}: {x.shape} -> {T_ret.shape}'
                )
            with pytest.raises(Exception,
                               match='`input` must be at-least .*d'):
                _ = T_fn(T.as_tensor(np.zeros([1] * spatial_ndims)))

        x = np.random.randn(2, 3, 4, 5)
        assert_equal(
            T.nn.channel_first_to_last2d(T.as_tensor(x)),
            np.transpose(x, [0, 2, 3, 1]),
        )

        # channels_to_first
        for spatial_ndims in (1, 2, 3):
            T_fn_name = f'channel_last_to_first{spatial_ndims}d'
            T_fn = getattr(T.nn, T_fn_name)
            for ndims in range(spatial_ndims + 1, spatial_ndims + 2):
                x = np.random.randn(*range(3, 3 + ndims))
                T_ret = T_fn(T.as_tensor(x))
                assert_equal(
                    T_ret,
                    channel_to_first_nd(x, spatial_ndims),
                    err_msg=f'{T_fn_name}: {x.shape} -> {T_ret.shape}'
                )
            with pytest.raises(Exception,
                               match='`input` must be at-least .*d'):
                _ = T_fn(T.as_tensor(np.zeros([1] * spatial_ndims)))

        x = np.random.randn(2, 3, 4, 5)
        assert_equal(
            T.nn.channel_last_to_first2d(T.as_tensor(x)),
            np.transpose(x, [0, 3, 1, 2]),
        )

        # space_to_depth and depth_to_space
        channel_size = 4
        for batch_shape in [[3], [2, 3]]:
            for spatial_shape in [[6], [6, 7], [6, 7, 8]]:
                spatial_ndims = len(spatial_shape)

                for block_size in [1, 2, 3]:
                    x = np.random.randn(
                        *make_conv_shape(
                            batch_shape, channel_size,
                            [a * block_size for a in spatial_shape]
                        )
                    )
                    T_fn1 = getattr(T.nn, f'space_to_depth{spatial_ndims}d')
                    T_fn2 = getattr(T.nn, f'depth_to_space{spatial_ndims}d')

                    # test space_to_depth
                    fn1_out = T_fn1(T.as_tensor(x), block_size=block_size)
                    assert_equal(
                        fn1_out,
                        space_to_depth_nd(x, block_size, spatial_ndims)
                    )

                    with pytest.raises(Exception,
                                       match='`input` must be at-least .*d'):
                        in_shape = make_conv_shape(
                            [], channel_size, [a * block_size for a in spatial_shape])
                        _ = T_fn1(
                            T.as_tensor(np.random.randn(*in_shape)),
                            block_size
                        )

                    if block_size > 1:
                        with pytest.raises(Exception,
                                           match='multiples of'):
                            for i in range(spatial_ndims):
                                in_shape = make_conv_shape(
                                    batch_shape, channel_size,
                                    [a * block_size + int(i == j)
                                     for j, a in enumerate(spatial_shape)]
                                )
                                _ = T_fn1(
                                    T.as_tensor(np.random.randn(*in_shape)),
                                    block_size
                                )

                    # test depth_to_space
                    fn2_out = T_fn2(fn1_out, block_size=block_size)
                    assert_equal(fn2_out, x)

                    with pytest.raises(Exception,
                                       match='`input` must be at-least .*d'):
                        in_shape = make_conv_shape(
                            [],
                            channel_size * block_size ** spatial_ndims,
                            [a for a in spatial_shape]
                        )
                        _ = T_fn2(
                            T.as_tensor(np.random.randn(*in_shape)),
                            block_size
                        )

                    if block_size > 1:
                        with pytest.raises(Exception,
                                           match='multiples of'):
                            in_shape = make_conv_shape(
                                batch_shape,
                                channel_size * block_size ** spatial_ndims + 1,
                                [a for a in spatial_shape]
                            )
                            for i in range(spatial_ndims):
                                _ = T_fn2(
                                    T.as_tensor(np.random.randn(*in_shape)),
                                    block_size
                                )

    def test_avg_pool(self):
        def is_valid_padding(spatial_ndims, padding, kernel_size):
            if not hasattr(padding, '__iter__'):
                padding = [int(padding)] * spatial_ndims
            if not hasattr(kernel_size, '__iter__'):
                kernel_size = [int(kernel_size)] * spatial_ndims
            for p, k in zip(padding, kernel_size):
                if p >= k / 2.0:
                    return False
            return True

        def do_check(pool_type, spatial_ndims, x, kernel_size, stride, padding,
                     count_padded_zeros):
            kwargs = dict(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            if pool_type == 'avg':
                kwargs['count_padded_zeros'] = count_padded_zeros
            elif not count_padded_zeros:
                return

            assert_allclose(
                getattr(T.nn, f'{pool_type}_pool{spatial_ndims}d')(
                    T.as_tensor(x), **kwargs),
                getattr(ops, f'{pool_type}_pool_nd')(spatial_ndims, x, **kwargs),
                atol=1e-6, rtol=1e-4,
                err_msg=f'pool_type={pool_type}, '
                        f'spatial_ndims={spatial_ndims}, '
                        f'kernel_size={kernel_size}, '
                        f'stride={stride}, '
                        f'padding={padding}, '
                        f'count_padded_zeros={count_padded_zeros}'
            )

        np.random.seed(1234)
        spatial_shape = [12, 13, 14]
        for spatial_ndims in (1, 2):
            x = np.random.uniform(
                size=make_conv_shape([3], 5, spatial_shape[: spatial_ndims]))
            for pool_type, kernel_size, stride, padding, count_padded_zeros in \
                    product(
                        ('avg', 'max'),
                        (1, 2, [5, 3, 1][:spatial_ndims]),
                        (1, 2, [3, 2, 1][:spatial_ndims]),
                        (0, 1, [2, 1, 0][:spatial_ndims]),
                        (True, False),
                    ):
                if not is_valid_padding(spatial_ndims, padding, kernel_size):
                    continue
                do_check(
                    pool_type=pool_type,
                    spatial_ndims=spatial_ndims,
                    x=x,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    count_padded_zeros=count_padded_zeros
                )

        # 3d is too slow, just check one situation
        for pool_type in ('avg', 'max'):
            x = np.random.uniform(
                size=make_conv_shape([3], 5, spatial_shape))
            do_check(
                pool_type=pool_type,
                spatial_ndims=3,
                x=x,
                kernel_size=[5, 3, 1],
                stride=[3, 2, 1],
                padding=[2, 1, 0],
                count_padded_zeros=True
            )
