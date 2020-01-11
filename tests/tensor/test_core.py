import unittest
from functools import partial

import numpy as np
import pytest
from scipy.special import erf, erfc, erfinv

from tensorkit import settings
from tensorkit import tensor as T
from tests.helper import *

assert_allclose = np.testing.assert_allclose


class TensorCoreTestCase(unittest.TestCase):

    def test_backend_info(self):
        self.assertEqual(T.backend_name, settings.backend)

    def test_dtypes(self):
        x = np.asarray([1, 2, 3])

        # various dtypes
        for dtype in [T.int8, T.uint8, T.int16, T.int32, T.int64]:
            self.assertIsInstance(dtype, str)
            self.assertFalse(T.is_floating_point(T.from_numpy(0, dtype=dtype)))
            self.assertEqual(T.get_dtype(T.cast(T.as_tensor(x), dtype)), dtype)

        for dtype in [T.float16, T.float32, T.float64]:
            self.assertIsInstance(dtype, str)
            self.assertTrue(T.is_floating_point(T.from_numpy(0, dtype=dtype)))
            self.assertEqual(T.get_dtype(T.cast(T.as_tensor(x), dtype)), dtype)

        # floatx
        self.assertEqual(settings.float_x, 'float32')
        self.assertEqual(T.float_x(), T.float32)
        try:
            settings.float_x = 'float64'
            self.assertEqual(T.float_x(), T.float64)
        finally:
            settings.float_x = 'float32'

        # as_tensor
        t = T.as_tensor(x)
        self.assertIsInstance(t, T.Tensor)
        np.testing.assert_equal(T.to_numpy(t), x)

        # cast
        for dtype in number_dtypes:
            t2 = T.cast(t, dtype)
            self.assertIsInstance(t2, T.Tensor)
            self.assertEqual(T.get_dtype(t2), dtype)
            np.testing.assert_equal(T.to_numpy(t2), x)

        # cast_like
        for dtype_as in (t, t2):
            t3 = T.cast_like(t, dtype_as)
            self.assertIsInstance(t3, T.Tensor)
            self.assertEqual(T.get_dtype(t3), T.get_dtype(dtype_as))
            np.testing.assert_equal(T.to_numpy(t3), x)

    def test_tensor_constructors(self):
        np.random.seed(1234)

        # as_tensor
        for x in [1., 1, [1., 2., 3.], np.array([1., 2., 3.])]:
            t = T.as_tensor(x)
            self.assertIsInstance(t, T.Tensor)
            np.testing.assert_equal(T.to_numpy(t), x)

        x = T.as_tensor(np.asarray([1, 2, 3], dtype=np.int32))
        t = T.as_tensor(x)
        self.assertIs(t, x)

        with pytest.raises(Exception):
            _ = T.from_numpy(object())  # not a tensor, should raise error

        # from_numpy
        for x in [1., 1, [1., 2., 3.], np.array([1., 2., 3.])]:
            t = T.from_numpy(x)
            self.assertIsInstance(t, T.Tensor)
            np.testing.assert_equal(T.to_numpy(t), x)

            for dtype in number_dtypes:
                t = T.from_numpy(x, dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                np.testing.assert_equal(T.to_numpy(t), x)

        with pytest.raises(Exception):
            _ = T.from_numpy(object())  # not a tensor, should raise error

        # float_scalar
        for value in (1.25, 125):
            for dtype in (T.float16, T.float32, T.float64):
                t = T.float_scalar(value, dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                self.assertEqual(T.to_numpy(t), value)
        self.assertEqual(T.get_dtype(T.float_scalar(1.25)), T.float_x())

        # int_scalar
        for value in (2, 125):
            for dtype in (T.int8, T.int16, T.int32, T.int64):
                t = T.int_scalar(value, dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                self.assertEqual(T.to_numpy(t), value)
        self.assertEqual(T.get_dtype(T.int_scalar(125)), T.int32)

        # zeros
        for shape in ([1, 2, 3], []):
            for dtype in number_dtypes:
                t = T.zeros(shape, dtype=dtype)
                self.assertIsInstance(t, T.Tensor)
                self.assertEqual(T.get_dtype(t), dtype)
                np.testing.assert_equal(T.to_numpy(t), np.zeros(shape))

                # zeros_like
                t2 = T.zeros_like(t)
                self.assertIsInstance(t2, T.Tensor)
                self.assertEqual(T.get_dtype(t2), dtype)
                np.testing.assert_equal(T.to_numpy(t), np.zeros(shape))

                for dtype2 in (None,) + number_dtypes:
                    for shape2 in (None, [7, 8]):
                        t2 = T.zeros_like(t, dtype=dtype2, shape=shape2)
                        self.assertIsInstance(t2, T.Tensor)
                        self.assertEqual(T.get_dtype(t2), dtype2 or dtype)
                        np.testing.assert_equal(T.to_numpy(t2),
                                                np.zeros(shape2 or shape))

        # ones
        for shape in ([1, 2, 3], []):
            for dtype in number_dtypes:
                t = T.ones(shape, dtype=dtype)
                self.assertIsInstance(t, T.Tensor)
                self.assertEqual(T.get_dtype(t), dtype)
                np.testing.assert_equal(T.to_numpy(t), np.ones(shape))

                # ones_like
                t2 = T.ones_like(t)
                self.assertIsInstance(t2, T.Tensor)
                self.assertEqual(T.get_dtype(t2), dtype)
                np.testing.assert_equal(T.to_numpy(t), np.ones(shape))

                for dtype2 in (None,) + number_dtypes:
                    for shape2 in (None, [7, 8]):
                        t2 = T.ones_like(t, dtype=dtype2, shape=shape2)
                        self.assertIsInstance(t2, T.Tensor)
                        self.assertEqual(T.get_dtype(t2), dtype2 or dtype)
                        np.testing.assert_equal(T.to_numpy(t2),
                                                np.ones(shape2 or shape))

        # full
        fill_value = 123
        for shape in ([1, 2, 3], []):
            for dtype in number_dtypes:
                t = T.full(shape, fill_value, dtype=dtype)
                self.assertIsInstance(t, T.Tensor)
                self.assertEqual(T.get_dtype(t), dtype)
                np.testing.assert_equal(
                    T.to_numpy(t),
                    np.full(shape, fill_value))

                # zeros_like
                t2 = T.full_like(t, fill_value)
                self.assertIsInstance(t2, T.Tensor)
                self.assertEqual(T.get_dtype(t2), dtype)
                np.testing.assert_equal(T.to_numpy(t),
                                        np.full(shape, fill_value))

                for dtype2 in (None,) + number_dtypes:
                    for shape2 in (None, [7, 8]):
                        t2 = T.full_like(t, fill_value, dtype=dtype2,
                                         shape=shape2)
                        self.assertIsInstance(t2, T.Tensor)
                        self.assertEqual(T.get_dtype(t2), dtype2 or dtype)
                        np.testing.assert_equal(
                            T.to_numpy(t2),
                            np.full(shape2 or shape, fill_value))

        # arange
        for start, end in [(1, 10), (0, 10)]:
            t = T.arange(start, end)
            self.assertIsInstance(t, T.Tensor)
            self.assertEqual(T.get_dtype(t), T.int32)
            np.testing.assert_equal(T.to_numpy(t), np.arange(start, end))

        for start, end, step in [(0, 10, 2), (-2, -15, -3)]:
            t = T.arange(start, end, step)
            self.assertIsInstance(t, T.Tensor)
            self.assertEqual(T.get_dtype(t), T.int32)
            np.testing.assert_equal(T.to_numpy(t), np.arange(start, end, step))

        for dtype in number_dtypes:
            t = T.arange(0, 10, dtype=dtype)
            self.assertIsInstance(t, T.Tensor)
            self.assertEqual(T.get_dtype(t), dtype)
            np.testing.assert_equal(T.to_numpy(t), np.arange(10))

        # one_hot
        for n_classes in [1, 5]:
            for shape in [[2, 3, 4], []]:
                I = np.eye(n_classes)
                x = np.random.randint(0, n_classes, size=shape)

                t = T.one_hot(T.as_tensor(x), n_classes)
                np.testing.assert_equal(T.to_numpy(t), I[x])

                for dtype in number_dtypes:
                    t = T.one_hot(T.as_tensor(x), n_classes, dtype=dtype)
                    self.assertEqual(T.get_dtype(t), dtype)
                    np.testing.assert_equal(T.to_numpy(t), I[x])

    def test_read_assign(self):
        # test to_numpy
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor(x)
        out = T.to_numpy(t)
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_equal(out, x)

        with pytest.raises(TypeError, match='Not a Tensor'):
            _ = T.to_numpy(object())

        # test to_numpy with bool
        x = np.asarray([True, False])
        t = T.as_tensor(x)
        out = T.to_numpy(t)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.bool)
        np.testing.assert_equal(out, x)

    def test_shape_utils(self):
        # test shape
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor(x)
        s = T.shape(t)
        self.assertEqual(s, [2, 3, 4])

        # test rank
        self.assertEqual(T.rank(t), 3)

        # test reshape
        t2 = T.reshape(t, [3, 8])
        self.assertEqual(T.shape(t2), [3, 8])
        np.testing.assert_equal(T.to_numpy(t2), np.reshape(x, [3, 8]))

        with pytest.raises(Exception):
            _ = T.reshape(t, [4, 8])

        # test repeat
        x = np.random.randn(2, 1, 3)
        t = T.as_tensor(x)

        t2 = T.repeat(t, [])
        self.assertEqual(T.shape(t2), [2, 1, 3])
        np.testing.assert_equal(T.to_numpy(t2), x)

        t2 = T.repeat(t, [2])
        self.assertEqual(T.shape(t2), [2, 1, 6])
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [1, 1, 2]))

        t2 = T.repeat(t, [4, 3, 2])
        self.assertEqual(T.shape(t2), [8, 3, 6])
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [4, 3, 2]))

        t2 = T.repeat(t, [4, 1, 3, 1])
        self.assertEqual(T.shape(t2), [4, 2, 3, 3])
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [4, 1, 3, 1]))

        t2 = T.repeat(t, [5, 4, 3, 2])
        self.assertEqual(T.shape(t2), [5, 8, 3, 6])
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [5, 4, 3, 2]))

        # test expand
        t2 = T.expand(t, [4, -1, 5, -1])
        self.assertEqual(T.shape(t2), [4, 2, 5, 3])
        np.testing.assert_equal(T.to_numpy(t2), np.tile(x, [4, 1, 5, 1]))

        # test squeeze
        x = np.random.randn(1, 2, 1, 3, 1, 4, 1)
        t = T.as_tensor(x)

        t2 = T.squeeze(T.as_tensor(x))
        s2 = [2, 3, 4]
        self.assertEqual(T.shape(t2), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.squeeze(t, [-1])
        s2 = [1, 2, 1, 3, 1, 4]
        self.assertEqual(T.shape(t2), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.squeeze(t, [-1, 0, 4, 6])
        s2 = [2, 1, 3, 4]
        self.assertEqual(T.shape(t2), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        with pytest.raises(Exception, match='Axis .* cannot be squeezed'):
            _ = T.squeeze(t, [-1, -2])

        # test expand dim
        x = np.random.randn(2, 3)
        t = T.as_tensor(x)

        t2 = T.expand_dim(t, -1)
        s2 = [2, 3, 1]
        self.assertEqual(T.shape(t2), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.expand_dim(t, -2)
        s2 = [2, 1, 3]
        self.assertEqual(T.shape(t2), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        t2 = T.expand_dim(t, 0)
        s2 = [1, 2, 3]
        self.assertEqual(T.shape(t2), s2)
        np.testing.assert_equal(T.to_numpy(t2), x.reshape(s2))

        # test broadcast_shape
        self.assertEqual(
            T.broadcast_shape([3, 4, 2, 1], [4, 1, 5]),
            [3, 4, 2, 5]
        )
        self.assertEqual(
            T.broadcast_shape([4, 1, 5], [3, 4, 2, 1]),
            [3, 4, 2, 5]
        )
        self.assertEqual(
            T.broadcast_shape([3, 4, 2, 1], []),
            [3, 4, 2, 1]
        )
        self.assertEqual(
            T.broadcast_shape([], [4, 1, 5]),
            [4, 1, 5]
        )

        with pytest.raises(Exception, match='cannot broadcast'):
            _ = T.broadcast_shape([2], [3])

        # test broadcast_to
        x = np.random.randn(1, 2, 1)
        t = T.as_tensor(x)

        t2 = T.broadcast_to(t, [4, 5, 2, 1])
        self.assertEqual(T.shape(t2), [4, 5, 2, 1])
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
            t = T.from_numpy(x, dtype=T.int32)

            if len(x.shape) == k:
                tt, s1 = T.flatten_to_ndims(t, k)
                self.assertIs(tt, t)
                self.assertIsNone(s1)
                self.assertIs(T.unflatten_from_ndims(tt, s1), t)
            else:
                if k == 1:
                    front_shape = list(x.shape)
                    xx = x.reshape([-1])
                else:
                    front_shape = list(x.shape)[: -(k - 1)]
                    xx = x.reshape([-1] + list(x.shape)[-(k - 1):])

                tt, s1 = T.flatten_to_ndims(t, k)
                self.assertEqual(s1, front_shape)
                np.testing.assert_equal(T.to_numpy(tt), xx)
                np.testing.assert_equal(
                    T.to_numpy(T.unflatten_from_ndims(tt, s1)),
                    x
                )

        x = np.arange(120)
        run_check(x, 1)

        x = np.arange(120).reshape([2, 3, 4, 5]).astype(np.int32)
        run_check(x, 1)
        run_check(x, 2)
        run_check(x, 3)
        run_check(x, 4)

        with pytest.raises(Exception,
                           match='`ndims` must be at least 1'):
            _ = T.flatten_to_ndims(T.as_tensor([0.]), 0)

        with pytest.raises(Exception, match=r'rank\(x\) < ndims'):
            _ = T.flatten_to_ndims(T.zeros([3, 4]), 3)

        with pytest.raises(Exception, match=r'rank\(x\) < ndims'):
            _ = T.flatten_to_ndims(T.zeros([3]), 2)

        with pytest.raises(Exception,
                           match=r'Invalid input: rank\(x\) < 1, but '
                                 r'front_shape is not None'):
            t = T.as_tensor(123)
            _ = T.unflatten_from_ndims(t, [2, 3])

    def test_index_select(self):
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

        if T.backend_name != 'PyTorch':
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

    def test_concat(self):
        x = np.random.randn(2, 3, 4)
        y = np.random.randn(2, 5, 4)
        z = np.random.randn(2, 3, 5)

        for arrays, axis in [([x, x, y], -2), ([x, y, y], 1),
                             ([x, x, z], -1), ([x, z, z], 2)]:
            t = T.concat([T.as_tensor(arr) for arr in arrays], axis=axis)
            expected = np.concatenate(arrays, axis=axis)
            np.testing.assert_equal(T.to_numpy(t), expected)

    def test_math_univariate_op(self):
        np.random.seed(1234)

        x = np.random.randn(2, 3)
        u = np.random.rand(2, 3)
        x_t = T.as_tensor(x)
        u_t = T.as_tensor(u)

        assert_allclose(T.to_numpy(T.abs(x_t)), np.abs(x))
        assert_allclose(T.to_numpy(T.neg(x_t)), -x)
        assert_allclose(T.to_numpy(T.square(x_t)), x ** 2)

        assert_allclose(T.to_numpy(T.exp(x_t)), np.exp(x))
        assert_allclose(T.to_numpy(T.log(T.as_tensor(np.abs(x)))),
                        np.log(np.abs(x)))
        assert_allclose(T.to_numpy(T.log1p(T.as_tensor(np.abs(x) - 1. + 1e-7))),
                        np.log1p(np.abs(x) - 1. + 1e-7))

        assert_allclose(T.to_numpy(T.sin(x_t)), np.sin(x))
        assert_allclose(T.to_numpy(T.cos(x_t)), np.cos(x))

        assert_allclose(T.to_numpy(T.erf(x_t)), erf(x))
        assert_allclose(T.to_numpy(T.erfc(x_t)), erfc(x))
        assert_allclose(T.to_numpy(T.erfinv(u_t)), erfinv(u))

    def test_math_bivariate_op(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3)
        y = np.random.randn(3)
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)

        assert_allclose(T.to_numpy(T.add(t1, t2)), x + y)
        assert_allclose(T.to_numpy(T.sub(t1, t2)), x - y)
        assert_allclose(T.to_numpy(T.mul(t1, t2)), x * y)
        assert_allclose(T.to_numpy(T.pow(T.as_tensor(np.abs(x)), t2)),
                        np.abs(x) ** y)

        # for division, of course y should not equal to zero
        y = np.asarray(y == 0, dtype=y.dtype) + y
        assert_allclose(T.to_numpy(T.div(t1, t2)), x / y)
        assert_allclose(T.to_numpy(T.truediv(t1, t2)), x / y)

        # for floordiv and mod, we only require the backend tensor engine
        # to produce identical results with numpy when x > 0 and y > 0
        x = np.abs(x)
        y = np.abs(y)
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)
        assert_allclose(T.to_numpy(T.floordiv(t1, t2)), x // y)
        assert_allclose(T.to_numpy(T.mod(t1, t2)), x % y)

        # truediv should raise error for dtype mismatch
        with pytest.raises(Exception, match='x and y must have the same dtype'):
            _ = T.truediv(T.cast(t1, dtype=T.float64),
                          T.cast(t2, dtype=T.float32))

        # in addition, we need to test truediv when x & y are both integers
        # (which is expected to produce float outputs)
        #
        # input uint8, output float32
        x = np.random.randint(0, 255, size=(2, 3), dtype=np.uint8)
        y = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        y = y + (y == 0).astype(y.dtype)
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)
        out = T.truediv(t1, t2)
        self.assertEqual(T.get_dtype(out), T.float32)
        assert_allclose(T.to_numpy(out),
                        x.astype(np.float32) / y.astype(np.float32))

        # input int16, output float32
        x = np.random.randint(-32768, 32767, size=(2, 3), dtype=np.int16)
        y = np.random.randint(-32768, 32767, size=(3,), dtype=np.int16)
        y = y + (y == 0).astype(y.dtype)
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)
        out = T.truediv(t1, t2)
        self.assertEqual(T.get_dtype(out), T.float32)
        assert_allclose(T.to_numpy(out),
                        x.astype(np.float32) / y.astype(np.float32))

        # input int32, output float64
        x = np.random.randint(-100000, 100000, size=(2, 3), dtype=np.int32)
        y = np.random.randint(-100000, 100000, size=(3,), dtype=np.int32)
        y = y + (y == 0).astype(y.dtype)
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)
        out = T.truediv(t1, t2)
        self.assertEqual(T.get_dtype(out), T.float64)
        assert_allclose(T.to_numpy(out),
                        x.astype(np.float64) / y.astype(np.float64))

    def test_math_sequential_op(self):
        # test add_n
        x = np.random.randn(2, 3)
        y = np.random.randn(3)
        z = np.random.randn(2, 1)

        np.testing.assert_allclose(
            T.to_numpy(T.add_n([T.as_tensor(t) for t in (x, y, z)])),
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
                T.to_numpy(T_op(t, axes=[-1])),
                np_op(x, axis=-1)
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axes=[-1], keepdims=True)),
                np_op(x, axis=-1, keepdims=True)
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axes=[0, -1])),
                np_op(x, axis=(0, -1))
            )
            np.testing.assert_allclose(
                T.to_numpy(T_op(t, axes=[0, -1], keepdims=True)),
                np_op(x, axis=(0, -1), keepdims=True)
            )

    def test_logical_op(self):
        def read_bool(t):
            return T.to_numpy(t)

        def with_raise(name, fn):
            with pytest.raises(Exception, match=f'Expected {name} to be .*, '
                                                f'got .* of type'):
                _ = fn()

        x = np.asarray([[True, True, False, False],
                        [False, False, True, True]])
        y = np.asarray([True, False, False, True])
        t1 = T.as_tensor(x)
        t2 = T.as_tensor(y)

        # test as_boolean
        self.assertEqual(T.get_dtype(t1), T.boolean)
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

        # test multiply_mask
        def test_multiply_mask(x, y, dtype, mask_dtype):
            t = T.multiply_mask(
                T.from_numpy(x, dtype=dtype),
                T.from_numpy(y, dtype=mask_dtype)
            )
            self.assertEqual(T.get_dtype(t), dtype)
            np.testing.assert_allclose(
                T.to_numpy(t),
                np.asarray(x * np.asarray(y, dtype=x.dtype),
                           dtype=dtype)
            )

        for dtype in ['float32', 'int32']:
            for mask_dtype in ['bool', 'int32']:
                test_multiply_mask(
                    np.random.randn(5, 3),
                    np.array([True, False, True]),
                    dtype=dtype,
                    mask_dtype=mask_dtype,
                )
                test_multiply_mask(
                    np.random.randn(3),
                    np.tile(np.array([[True, False, True]]), [5, 1]),
                    dtype=dtype,
                    mask_dtype=mask_dtype,
                )

        # test where
        def do_test_where(condition, x=None, y=None, dtype=None):
            condition = np.asarray(condition, dtype=np.bool)
            if x is not None and y is not None:
                x = np.asarray(x, dtype=dtype)
                y = np.asarray(y, dtype=dtype)
                expected = np.where(condition, x, y)
                ret = T.where(
                    T.from_numpy(condition, dtype=T.boolean),
                    T.from_numpy(x, dtype=dtype),
                    T.from_numpy(y, dtype=dtype),
                )
                self.assertEqual(T.get_dtype(ret), dtype)
                np.testing.assert_equal(T.to_numpy(ret), expected)
            else:
                expected = np.where(condition)
                self.assertEqual(len(expected), len(condition.shape))
                ret = T.where(T.from_numpy(condition, dtype=T.boolean))
                self.assertEqual(len(ret), len(condition.shape))
                for a, b in zip(ret, expected):
                    np.testing.assert_equal(T.to_numpy(a), b)

        do_test_where([True, False])
        do_test_where([[True, False], [False, True]])
        do_test_where(np.random.binomial(1, 0.5, [2, 3, 4, 5]).astype(np.bool))

        for dtype in number_dtypes:
            for shape in ([5], [2, 3]):
                # without broadcast
                do_test_where(
                    np.random.binomial(1, 0.5, shape),
                    np.random.randn(*shape),
                    np.random.randn(*shape),
                    dtype=dtype
                )

                # with broadcast
                do_test_where(
                    np.random.binomial(1, 0.5, shape),
                    np.random.randn(7, *shape),
                    2.0,
                    dtype=dtype
                )

    def test_comparison_op(self):
        def read_bool(t):
            self.assertEqual(T.get_dtype(t), T.boolean)
            return T.to_numpy(t)

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
        y = np.random.randn(2, 3, 4)

        # requires_grad
        yt = T.requires_grad(T.as_tensor(y))

        xt = T.as_tensor(x)
        xt_copy = T.requires_grad(xt, copy=False)
        self.assertIs(xt_copy, xt)
        l_sum = T.reduce_sum(xt + xt_copy)
        # xtt and xt are the same tensor, thus gradient should pass along the both paths
        [x_grad] = T.grad([l_sum], [xt_copy])
        np.testing.assert_allclose(T.to_numpy(x_grad), np.full_like(x, 2))

        xt_copy = T.requires_grad(xt, copy=True)
        self.assertIsNot(xt_copy, xt)
        l_sum = T.reduce_sum(xt + xt_copy)
        # xttt is a copy of xt, thus grad should pass to xt along both paths
        # when taking derivative against xt
        [x_grad] = T.grad([l_sum], [xt])
        np.testing.assert_allclose(T.to_numpy(x_grad), np.full_like(x, 2))
        # but grad should not pass to xt if taking derivative against xttt
        [x_grad] = T.grad([l_sum], [xt_copy])
        np.testing.assert_allclose(T.to_numpy(x_grad), np.full_like(x, 1))

        # grad
        l_sum = T.reduce_sum(xt * yt)
        l_squares = 7 * xt ** 3 + 11 * yt ** 3

        [x_grad, y_grad] = T.grad(
            [l_sum, l_squares],
            [xt, yt],
            grad_outputs=[None, T.ones_like(l_squares)],
            keep_graph=True,
            create_graph=True
        )
        np.testing.assert_allclose(T.to_numpy(x_grad), y + 21 * x ** 2)
        np.testing.assert_allclose(T.to_numpy(y_grad), x + 33 * y ** 2)

        # second order grad
        [x_grad_2, y_grad_2] = T.grad(
            [x_grad, y_grad],
            [xt, yt],
            grad_outputs=[T.ones_like(xt), T.ones_like(yt)],
            keep_graph=True,
            create_graph=False
        )
        np.testing.assert_allclose(T.to_numpy(x_grad_2), 42. * x + 1.)
        np.testing.assert_allclose(T.to_numpy(y_grad_2), 66. * y + 1.)

        # get the first order grad again, but once for each of x and y
        [x_grad] = T.grad(
            [l_sum, l_squares],
            [xt],
            grad_outputs=[None, T.ones_like(l_squares)],
            keep_graph=True,
            create_graph=True
        )
        np.testing.assert_allclose(T.to_numpy(x_grad), y + 21 * x ** 2)

        [y_grad] = T.grad(
            [l_sum, l_squares],
            [yt],
            grad_outputs=[None, T.ones_like(l_squares)],
            keep_graph=True,
            create_graph=True
        )
        np.testing.assert_allclose(T.to_numpy(y_grad), x + 33 * y ** 2)

        # stop_grad
        l_sum = T.reduce_sum(T.stop_grad(xt ** 2) * yt)
        [x_grad, y_grad] = T.grad(
            [l_sum],
            [xt, yt],
            keep_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        self.assertTrue(T.is_null_grad(xt, x_grad))
        self.assertFalse(T.is_null_grad(yt, y_grad))
        np.testing.assert_allclose(T.to_numpy(y_grad), x ** 2)

        # is_null_grad counterexample
        self.assertFalse(T.is_null_grad(T.zeros([]), T.zeros([])))
        self.assertFalse(T.is_null_grad(T.random.randn([1, 2]), T.ones([])))

        # stop_grad, but `allow_unused` is False
        l_sum = T.reduce_sum(T.stop_grad(xt ** 2) * yt)
        with pytest.raises(Exception, match='Set allow_unused=True'):
            _ = T.grad(
                [l_sum],
                [xt, yt],
                keep_graph=False,
                create_graph=False,
                allow_unused=False,
            )

    def test_assertions(self):
        # is_finite and assert_finite
        for x in [np.array([-1, 0, 1]), np.array([1., 2., 3.]),
                  np.array([np.inf, 0.]), np.array([np.nan, 0.]),
                  np.array([np.inf, np.nan])]:
            t = T.as_tensor(x)
            np.testing.assert_equal(T.is_finite(t), np.isfinite(x))
            is_finite = np.all(np.isfinite(x))

            if is_finite:
                np.testing.assert_equal(T.to_numpy(T.assert_finite(t, 't')), x)
            else:
                with pytest.raises(Exception,
                                   match='Infinity or NaN value encountered'):
                    _ = T.assert_finite(t, 't')
