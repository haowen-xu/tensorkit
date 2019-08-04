import unittest
from functools import partial
from typing import *

import numpy as np
import pytest

from tensorkit import settings
from tensorkit import tensor as T

assert_allclose = np.testing.assert_allclose


class TensorCoreTestCase(unittest.TestCase):

    def test_backend_info(self):
        self.assertEqual(T.backend.name, settings.backend)

    def test_typing(self):
        t = T.as_tensor(np.random.randn(2, 3))
        self.assertIsInstance(t, T.Tensor)

        # TODO: check T.Variable

        self.assertIsInstance(T.int32, T.DType)

        s = T.as_shape([1, 2, 3])
        self.assertIsInstance(s, T.Shape)
        self.assertEqual(tuple(s), (1, 2, 3))

    def test_dtypes(self):
        # test dtypes
        dtype_bits = {
            T.int8: 8, T.uint8: 8, T.int16: 16, T.int32: 32, T.int64: 64,
            T.float16: 16, T.float32: 32, T.float64: 64,
        }

        # (the following two sentences verify these dtypes can be used as keys)
        self.assertEqual(dtype_bits[T.int8], 8)
        self.assertEqual(dtype_bits[T.float64], 64)

        for dtype in [T.int8, T.uint8, T.int16, T.int32, T.int64]:
            dtype_obj = T.as_dtype(dtype)
            self.assertIs(dtype_obj, dtype)
            self.assertEqual(T.iinfo(dtype).bits, dtype_bits[dtype])
            self.assertFalse(T.is_floating_point(dtype))

        for dtype in [T.float16, T.float32, T.float64]:
            dtype_obj = T.as_dtype(dtype)
            self.assertIs(dtype_obj, dtype)
            self.assertEqual(T.finfo(dtype).bits, dtype_bits[dtype])
            self.assertTrue(T.is_floating_point(dtype))

        # test str dtypes
        dtype_bits = {
            'int8': 8, 'uint8': 8, 'int16': 16, 'int32': 32, 'int64': 64,
            'float16': 16, 'float32': 32, 'float64': 64,
        }

        for dtype in ['int8', 'uint8', 'int16', 'int32', 'int64']:
            dtype_obj = T.as_dtype(dtype)
            self.assertIsInstance(dtype_obj, T.DType)
            self.assertEqual(T.iinfo(dtype).bits, dtype_bits[dtype])
            self.assertFalse(T.is_floating_point(dtype))

        for dtype in ['float16', 'float32', 'float64']:
            dtype_obj = T.as_dtype(dtype)
            self.assertIsInstance(dtype_obj, T.DType)
            self.assertEqual(T.finfo(dtype).bits, dtype_bits[dtype])
            self.assertTrue(T.is_floating_point(dtype))

        # test numpy dtypes
        for dtype in [int, np.int, np.int8, np.uint8, np.int16, np.int32,
                      np.int64]:
            dtype_obj = T.as_dtype(dtype)
            self.assertIsInstance(dtype_obj, T.DType)
            self.assertEqual(T.iinfo(dtype).bits, np.iinfo(dtype).bits)
            self.assertFalse(T.is_floating_point(dtype))

        for dtype in [float, np.float, np.double, np.float16, np.float32,
                      np.float64]:
            dtype_obj = T.as_dtype(dtype)
            self.assertIsInstance(dtype_obj, T.DType)
            self.assertEqual(T.finfo(dtype).bits, np.finfo(dtype).bits)
            self.assertTrue(T.is_floating_point(dtype))

        # test invalid type
        with pytest.raises(ValueError, match='Not a valid dtype: \'invalid\''):
            _ = T.as_dtype('invalid')

        # test floatx
        self.assertEqual(settings.float_x, 'float32')
        self.assertEqual(T.float_x(), T.float32)
        try:
            settings.float_x = 'float64'
            self.assertEqual(T.float_x(), T.float64)
        finally:
            settings.float_x = 'float32'

        # test cast and get dtype of a tensor
        x = np.asarray([1, 2, 3])

        t = T.as_tensor(x, dtype=T.int32)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.int32)
        np.testing.assert_equal(T.to_numpy(t), x)

        t2 = T.cast(t, T.float32)
        self.assertIsInstance(t2, T.Tensor)
        self.assertEqual(T.dtype(t2), T.float32)
        np.testing.assert_equal(T.to_numpy(t2), x)

    def test_tensor_constructors(self):
        np.random.seed(1234)

        # test as_tensor
        t = T.as_tensor(1)
        self.assertIsInstance(t, T.Tensor)
        self.assertFalse(T.is_floating_point(T.dtype(t)))
        self.assertEqual(T.shape(t), ())
        self.assertEqual(T.to_numpy(t), 1)

        t = T.as_tensor([1., 2., 3.])
        self.assertIsInstance(t, T.Tensor)
        self.assertTrue(T.is_floating_point(T.dtype(t)))
        self.assertEqual(T.shape(t), (3,))
        np.testing.assert_equal(T.to_numpy(t), np.array([1, 2, 3]))

        x = np.random.randn(2, 3).astype(np.float32)
        t = T.as_tensor(x)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float32)
        self.assertEqual(T.shape(t), (2, 3))
        np.testing.assert_equal(T.to_numpy(t), x)

        x = T.as_tensor(np.asarray([1, 2, 3], dtype=np.int32))
        t = T.as_tensor(x)
        self.assertIs(t, x)

        with pytest.raises(Exception):
            _ = T.as_tensor(object())  # not a tensor, should raise error

        # test register_as_tensor
        class MyArray(object):
            def __init__(self, data):
                self.data = data

        x = np.random.normal(size=[1, 2, 3]).astype(np.float32)
        with pytest.raises(Exception):
            _ = T.as_tensor(MyArray(x))

        def to_tensor(data: MyArray, dtype: Optional[T.DType]) -> T.Tensor:
            return T.as_tensor(data.data, dtype)

        T.register_as_tensor(MyArray, to_tensor)

        t = T.as_tensor(MyArray(x))
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float32)
        np.testing.assert_allclose(T.to_numpy(t), x)

        t = T.as_tensor(MyArray(x), T.float64)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float64)
        np.testing.assert_allclose(T.to_numpy(t), x)

        # test zeros
        t = T.zeros([1, 2, 3], T.float16)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float16)
        np.testing.assert_equal(T.to_numpy(t), np.zeros([1, 2, 3]))

        # test ones
        t = T.ones([1, 2, 3], T.float16)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float16)
        np.testing.assert_equal(T.to_numpy(t), np.ones([1, 2, 3]))

        # test arange
        t = T.arange(10)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.int32)
        np.testing.assert_equal(T.to_numpy(t), np.arange(10))

        t = T.arange(1, 10, dtype=T.float32)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float32)
        np.testing.assert_equal(T.to_numpy(t), np.arange(1, 10))

        t = T.arange(10, step=2, dtype=T.float32)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.float32)
        np.testing.assert_equal(T.to_numpy(t), np.arange(10, step=2))

        t = T.arange(-2, -15, -3)
        self.assertIsInstance(t, T.Tensor)
        self.assertEqual(T.dtype(t), T.int32)
        np.testing.assert_equal(T.to_numpy(t), np.arange(-2, -15, -3))

    def test_shape_utils(self):
        # test shape
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor(x)
        s = T.shape(t)
        self.assertIsInstance(s, T.Shape)
        self.assertEqual(tuple(s), (2, 3, 4))

        # test rank
        self.assertEqual(T.rank(t), 3)

        # test reshape
        t2 = T.reshape(t, [3, 8])
        self.assertEqual(tuple(T.shape(t2)), (3, 8))
        np.testing.assert_equal(T.to_numpy(t2), np.reshape(x, [3, 8]))

        with pytest.raises(Exception):
            _ = T.reshape(t, [4, 8])

        # test repeat
        x = np.random.randn(2, 1, 3)
        t = T.as_tensor(x)

        t2 = T.repeat(t, [])
        self.assertEqual(tuple(T.shape(t2)), (2, 1, 3))
        np.testing.assert_equal(T.to_numpy(t2), x)

        t2 = T.repeat(t, [2])
        self.assertEqual(tuple(T.shape(t2)), (2, 1, 6))
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [1, 1, 2]))

        t2 = T.repeat(t, [4, 3, 2])
        self.assertEqual(tuple(T.shape(t2)), (8, 3, 6))
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [4, 3, 2]))

        t2 = T.repeat(t, [4, 1, 3, 1])
        self.assertEqual(tuple(T.shape(t2)), (4, 2, 3, 3))
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [4, 1, 3, 1]))

        t2 = T.repeat(t, [5, 4, 3, 2])
        self.assertEqual(tuple(T.shape(t2)), (5, 8, 3, 6))
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [5, 4, 3, 2]))

        # test expand
        t2 = T.expand(t, [4, -1, 5, -1])
        self.assertEqual(tuple(T.shape(t2)), (4, 2, 5, 3))
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [4, 1, 5, 1]))

        # test squeeze
        x = np.random.randn(1, 2, 1, 3, 1, 4, 1)
        t = T.as_tensor(x)

        t2 = T.squeeze(x)
        s2 = (2, 3, 4)
        self.assertEqual(tuple(T.shape(t2)), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.squeeze(t, -1)
        s2 = (1, 2, 1, 3, 1, 4)
        self.assertEqual(tuple(T.shape(t2)), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.squeeze(t, [-1, 0, 4, 6])
        s2 = (2, 1, 3, 4)
        self.assertEqual(tuple(T.shape(t2)), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        with pytest.raises(Exception, match='Axis .* cannot be squeezed'):
            _ = T.squeeze(t, [-1, -2])

        # test expand dim
        x = np.random.randn(2, 3)
        t = T.as_tensor(x)

        t2 = T.expand_dim(t, -1)
        s2 = (2, 3, 1)
        self.assertEqual(tuple(T.shape(t2)), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.expand_dim(t, -2)
        s2 = (2, 1, 3)
        self.assertEqual(tuple(T.shape(t2)), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.expand_dim(t, 0)
        s2 = (1, 2, 3)
        self.assertEqual(tuple(T.shape(t2)), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        # test broadcast_shape
        self.assertEqual(
            tuple(T.broadcast_shape([3, 4, 2, 1], [4, 1, 5])),
            (3, 4, 2, 5)
        )
        self.assertEqual(
            tuple(T.broadcast_shape((4, 1, 5), (3, 4, 2, 1))),
            (3, 4, 2, 5)
        )

        with pytest.raises(Exception, match='cannot broadcast'):
            _ = T.broadcast_shape([2], [3])

        # test broadcast_to
        x = np.random.randn(1, 2, 1)
        t = T.as_tensor(x)

        t2 = T.broadcast_to(t, [4, 5, 2, 1])
        self.assertEqual(tuple(T.shape(t2)), (4, 5, 2, 1))
        np.testing.assert_equal(
            T.to_numpy(t2),
            np.tile(x.reshape([1, 1, 2, 1]), [4, 5, 1, 1])
        )

        with pytest.raises(Exception,
                           match='`x` cannot be broadcast to `new_shape`'):
            _ = T.broadcast_to(t, [2, 5])

        with pytest.raises(Exception,
                           match='`x` cannot be broadcast to `new_shape`'):
            _ = T.broadcast_to(t, [1, 1, 1])

        with pytest.raises(Exception,
                           match='`x` cannot be broadcast to `new_shape`'):
            _ = T.broadcast_to(t, [1, 5, 1])

        # test explicit_broadcast
        def explicit_broadcast(x, y):
            x = x * np.ones_like(y, dtype=x.dtype)
            y = y * np.ones_like(x, dtype=y.dtype)
            return x, y

        def check_explicit_broadcast(shape1, shape2):
            x = np.asarray(np.random.randn(*shape1))
            y = np.asarray(np.random.randn(*shape2))
            out1, out2 = T.explicit_broadcast(T.as_tensor(x), T.as_tensor(y))
            out1 = T.to_numpy(out1)
            out2 = T.to_numpy(out2)
            ans1, ans2 = explicit_broadcast(x, y)
            np.testing.assert_equal(out1, ans1)
            np.testing.assert_equal(out2, ans2)

        check_explicit_broadcast([2, 3], [2, 3])
        check_explicit_broadcast([1, 2], [5, 3, 1])
        check_explicit_broadcast([5, 3, 1], [1, 2])
        check_explicit_broadcast([], [1, 1, 1, 1])

        # test flatten_to_ndims
        def run_check(x, k):
            t = T.as_tensor(x, dtype=T.int32)

            if len(x.shape) == k:
                tt, s1 = T.flatten_to_ndims(t, k)
                self.assertIs(tt, t)
                self.assertIsNone(s1)
                self.assertIs(T.unflatten_from_ndims(tt, s1), t)
            else:
                if k == 1:
                    front_shape = tuple(x.shape)
                    xx = x.reshape([-1])
                else:
                    front_shape = tuple(x.shape)[: -(k - 1)]
                    xx = x.reshape([-1] + list(x.shape)[-(k - 1):])

                tt, s1 = T.flatten_to_ndims(t, k)
                self.assertEqual(tuple(s1), front_shape)
                np.testing.assert_equal(T.to_numpy(tt), xx)
                np.testing.assert_equal(
                    T.to_numpy(T.unflatten_from_ndims(tt, s1)),
                    x
                )

        x = np.asarray(123)
        run_check(x, 0)

        x = np.arange(120)
        run_check(x, 1)

        x = np.arange(120).reshape([2, 3, 4, 5]).astype(np.int32)
        run_check(x, 1)
        run_check(x, 2)
        run_check(x, 3)
        run_check(x, 4)

        with pytest.raises(Exception,
                           match=r'`ndims >= 1` must hold when '
                                 r'`rank\(x\) >= 1`'):
            _ = T.flatten_to_ndims(T.as_tensor([0.]), 0)

        with pytest.raises(Exception, match=r'rank\(x\) < ndims'):
            _ = T.flatten_to_ndims(T.zeros([3, 4]), 3)

        with pytest.raises(Exception, match=r'rank\(x\) < ndims'):
            _ = T.flatten_to_ndims(T.zeros([3]), 2)

        with pytest.raises(Exception,
                           match=r'Invalid input: rank\(x\) < 1, but '
                                 r'front_shape is not None'):
            t = T.as_tensor(123)
            _ = T.unflatten_from_ndims(t, T.Shape([2, 3]))

    def test_split_etc(self):
        # test index_select
        x = np.random.randn(3, 4, 5)
        t = T.as_tensor(x)

        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(1), 0)),
            x[1, ...]
        )
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(3), 1)),
            x[:, 3, ...]
        )
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(2), -1)),
            x[..., 2]
        )

        i = np.asarray([0, 2, 1, 1, 0, 2])
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(i), 0)),
            x[i, ...]
        )
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(i), 1)),
            x[:, i, ...]
        )
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(i), -1)),
            x[..., i]
        )

        i = np.asarray([[0, 2, 1], [1, 0, 2]])
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(i), 0)),
            x[i, ...]
        )
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(i), 1)),
            x[:, i, ...]
        )
        np.testing.assert_equal(
            T.to_numpy(T.index_select(t, T.as_tensor(i), -1)),
            x[..., i]
        )

        if T.backend.name != 'pytorch':
            # TODO: pytorch currently does not support negative index in many
            # of its functions.  enable these test when supported.
            np.testing.assert_equal(
                T.to_numpy(T.index_select(t, T.as_tensor(-1), 1)),
                x[:, -1]
            )

            i = np.asarray([0, 1, -1, 2, -2, 0])
            np.testing.assert_equal(
                T.to_numpy(T.index_select(t, T.as_tensor(i), 1)),
                x[:, i, ...]
            )

            i = np.asarray([[0, 1, -1], [2, -2, 0]])
            np.testing.assert_equal(
                T.to_numpy(T.index_select(t, T.as_tensor(i), 1)),
                x[:, i, ...]
            )

        with pytest.raises(Exception, match='`axis` out of range'):
            _ = T.index_select(t, T.as_tensor(0), 3)

        with pytest.raises(Exception, match='`axis` out of range'):
            _ = T.index_select(t, T.as_tensor(0), -4)

    def test_math_univariate_op(self):
        np.random.seed(1234)

        x = np.random.randn(2, 3)
        assert_allclose(T.to_numpy(T.abs(x)), np.abs(x))
        assert_allclose(T.to_numpy(T.neg(x)), -x)
        assert_allclose(T.to_numpy(T.exp(x)), np.exp(x))
        assert_allclose(T.to_numpy(T.log(np.abs(x))), np.log(np.abs(x)))
        assert_allclose(T.to_numpy(T.log1p(np.abs(x) - 1. + 1e-7)),
                        np.log1p(np.abs(x) - 1. + 1e-7))
        assert_allclose(T.to_numpy(T.sin(x)), np.sin(x))
        assert_allclose(T.to_numpy(T.cos(x)), np.cos(x))
        assert_allclose(T.to_numpy(T.square(x)), x ** 2)

    def test_math_bivariate_op(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3)
        y = np.random.randn(3)

        assert_allclose(T.to_numpy(T.add(x, y)), x + y)
        assert_allclose(T.to_numpy(T.sub(x, y)), x - y)
        assert_allclose(T.to_numpy(T.mul(x, y)), x * y)
        assert_allclose(T.to_numpy(T.pow(np.abs(x), y)), np.abs(x) ** y)

        # for division, of course y should not equal to zero
        y = np.asarray(y == 0, dtype=y.dtype) + y
        assert_allclose(T.to_numpy(T.div(x, y)), x / y)
        assert_allclose(T.to_numpy(T.truediv(x, y)), x / y)

        # for floordiv and mod, we only require the backend tensor engine
        # to produce identical results with numpy when x > 0 and y > 0
        x = np.abs(x)
        y = np.abs(y)
        assert_allclose(T.to_numpy(T.floordiv(x, y)), x // y)
        assert_allclose(T.to_numpy(T.mod(x, y)), x % y)
        assert_allclose(T.to_numpy(T.fmod(x, y)), x % y)

        # truediv should raise error for dtype mismatch
        with pytest.raises(TypeError, match='x and y must have the same dtype'):
            _ = T.truediv(T.cast(x, dtype=T.float64),
                          T.cast(x, dtype=T.float32))

        # in addition, we need to test truediv when x & y are both integers
        # (which is expected to produce float outputs)
        #
        # input uint8, output float32
        x = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
        y = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        y = y + (y == 0).astype(y.dtype)
        out = T.truediv(x, y)
        self.assertEqual(T.dtype(out), T.float32)
        assert_allclose(T.to_numpy(out),
                        x.astype(np.float32) / y.astype(np.float32))

        # input int16, output float32
        x = np.random.randint(-32768, 32767, size=(2, 3), dtype=np.int16)
        y = np.random.randint(-32768, 32767, size=(3,), dtype=np.int16)
        y = y + (y == 0).astype(y.dtype)
        out = T.truediv(x, y)
        self.assertEqual(T.dtype(out), T.float32)
        assert_allclose(T.to_numpy(out),
                        x.astype(np.float32) / y.astype(np.float32))

        # input int32, output float64
        x = np.random.randint(-100000, 100000, size=(2, 3), dtype=np.int32)
        y = np.random.randint(-100000, 100000, size=(3,), dtype=np.int32)
        y = y + (y == 0).astype(y.dtype)
        out = T.truediv(x, y)
        self.assertEqual(T.dtype(out), T.float64)
        assert_allclose(T.to_numpy(out),
                        x.astype(np.float64) / y.astype(np.float64))

    def test_math_sequential_op(self):
        # test add_n
        x = np.random.randn(2, 3)
        y = np.random.randn(3)
        z = np.random.randn(2, 1)

        np.testing.assert_allclose(
            T.add_n(T.as_tensor(t) for t in (x, y, z)),
            x + y + z
        )

        with pytest.raises(Exception, match='`tensors` must not be empty'):
            _ = T.add_n([])

    def test_reduction_op(self):
        def log_f_exp(f, x, axis=None, keepdims=False):
            x_max_keepdims = np.max(x, axis=axis, keepdims=True)
            if not keepdims:
                x_max = np.squeeze(x_max_keepdims, axis=axis)
            else:
                x_max = x_max_keepdims
            f_exp = f(np.exp(x - x_max_keepdims), axis=axis, keepdims=keepdims)
            return x_max + np.log(f_exp)

        log_sum_exp = partial(log_f_exp, np.sum)
        log_mean_exp = partial(log_f_exp, np.mean)

        # prepare for the data
        np.random.seed(1234)
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor(x)

        # test sum, mean, max, min
        for name in ['sum', 'mean', 'min', 'max',
                     'log_sum_exp', 'log_mean_exp']:
            T_op = getattr(T, 'reduce_' + name, getattr(T, name, None))
            np_op = getattr(np, name,
                            {
                                'log_sum_exp': log_sum_exp,
                                'log_mean_exp': log_mean_exp,
                            }.get(name))

            np.testing.assert_allclose(
                T.to_numpy(T_op(t)),
                np_op(x)
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, keepdims=True)),
                np_op(x, keepdims=True)
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axis=-1)),
                np_op(x, axis=-1)
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axis=-1, keepdims=True)),
                np_op(x, axis=-1, keepdims=True)
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axis=[0, -1])),
                np_op(x, axis=(0, -1))
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axis=[0, -1], keepdims=True)),
                np_op(x, axis=(0, -1), keepdims=True)
            )

    def test_logical_op(self):
        def read_bool(t):
            return T.to_numpy(t).astype(np.bool)

        def with_raise(name, fn):
            with pytest.raises(Exception,
                               match=f'Expected {name} to be {T.boolean}'):
                _ = fn()

        x = np.asarray([[True, True, False, False],
                        [False, False, True, True]])
        y = np.asarray([True, False, False, True])
        t1 = T.to_boolean(x)
        t2 = T.to_boolean(y)

        # test to_boolean
        self.assertEqual(T.dtype(t1), T.boolean)
        np.testing.assert_equal(read_bool(t1), x)

        # test logical_not
        out = T.logical_not(t1)
        np.testing.assert_equal(read_bool(out), np.logical_not(x))
        with_raise('x', lambda: T.logical_not(T.as_tensor([1, 2, 3])))

        # test logical_and
        out = T.logical_and(t1, t2)
        np.testing.assert_equal(read_bool(out), np.logical_and(x, y))
        with_raise('x', lambda: T.logical_and(T.as_tensor([1, 2, 3, 4]), t2))
        with_raise('y', lambda: T.logical_and(t1, T.as_tensor([1, 2, 3, 4])))

        # test logical_or
        out = T.logical_or(t1, t2)
        np.testing.assert_equal(read_bool(out), np.logical_or(x, y))
        with_raise('x', lambda: T.logical_or(T.as_tensor([1, 2, 3, 4]), t2))
        with_raise('y', lambda: T.logical_or(t1, T.as_tensor([1, 2, 3, 4])))

        # test logical_xor
        out = T.logical_xor(t1, t2)
        np.testing.assert_equal(read_bool(out), np.logical_xor(x, y))
        with_raise('x', lambda: T.logical_xor(T.as_tensor([1, 2, 3, 4]), t2))
        with_raise('y', lambda: T.logical_xor(t1, T.as_tensor([1, 2, 3, 4])))

    def test_comparison_op(self):
        def read_bool(t):
            self.assertEqual(T.dtype(t), T.boolean)
            return T.to_numpy(t).astype(np.bool)

        np.random.seed(1234)
        x = np.random.randn(2, 3, 4)
        y = np.random.randn(1, 3, 4)
        x = np.concatenate([y, x], axis=0)
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)

        # test equal
        np.testing.assert_equal(read_bool(T.equal(t1, t2)), (x == y))

        # test not_equal
        np.testing.assert_equal(read_bool(T.not_equal(t1, t2)), (x != y))

        # test less
        np.testing.assert_equal(read_bool(T.less(t1, t2)), (x < y))

        # test less_equal
        np.testing.assert_equal(read_bool(T.less_equal(t1, t2)), (x <= y))

        # test greater
        np.testing.assert_equal(read_bool(T.greater(t1, t2)), (x > y))

        # test greater_equal
        np.testing.assert_equal(read_bool(T.greater_equal(t1, t2)), (x >= y))

        # test minimum
        np.testing.assert_equal(T.to_numpy(T.minimum(t1, t2)), np.minimum(x, y))

        # test maximum
        np.testing.assert_equal(T.to_numpy(T.maximum(t1, t2)), np.maximum(x, y))

        # test clip
        self.assertTrue(np.any(x < -0.5))
        self.assertTrue(np.any(x > 0.5))
        np.testing.assert_equal(
            T.to_numpy(T.clip(t1, -0.5, 0.5)),
            np.clip(x, -0.5, 0.5)
        )

    def test_gradient(self):
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor(x)
        T.requires_grad(t)
        self.assertIsNone(T.grad(t))

        # test back prop
        T.back_prop(T.sin(T.reduce_sum(t * t)))
        np.testing.assert_allclose(
            T.to_numpy(t.grad),
            np.cos(np.sum(x * x)) * 2 * x
        )

        # test grad accumulation
        T.back_prop(T.reduce_sum(t * t))
        np.testing.assert_allclose(
            T.to_numpy(t.grad),
            2 * x + np.cos(np.sum(x * x)) * 2 * x
        )

        # test grad no accumulation after clear_grad
        T.clear_grad(t)
        T.back_prop(T.reduce_sum(t * t))
        np.testing.assert_allclose(
            T.to_numpy(t.grad),
            2 * x
        )

        # test detach
        T.clear_grad(t)
        t2 = T.detach(t)
        T.back_prop(T.sin(T.reduce_sum(t * t2)))
        np.testing.assert_allclose(
            T.to_numpy(t.grad),
            np.cos(np.sum(x * x)) * x
        )
