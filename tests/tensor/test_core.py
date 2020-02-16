import copy
import itertools
import unittest
from functools import partial

import numpy as np
import pytest
from scipy.special import erf, erfc, erfinv

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tests.helper import *
from tests.ops import *


class TensorCoreTestCase(unittest.TestCase):

    def test_backend_info(self):
        self.assertEqual(T.backend_name, settings.backend)

    def test_jit_compile(self):
        # test compile layer
        layer = tk.layers.Linear(5, 3)
        layer2 = T.jit_compile(layer)
        if not tk.settings.disable_jit:
            self.assertTrue(T.is_jit_layer(layer2))
        else:
            self.assertFalse(T.is_jit_layer(layer2))

        # not supported object
        with pytest.raises(TypeError,
                           match='Not supported by `jit_compile`'):
            _ = T.jit_compile(object())

    def test_utilities(self):
        self.assertEqual(T.int_range(0, 10), list(range(10)))
        self.assertEqual(T.int_range(1, 10, 2), list(range(1, 10, 2)))
        self.assertEqual(T.int_range(-1, -10, -2), list(range(-1, -10, -2)))

        x = T.random.randn([3, 4, 5])
        assert_equal(T.identity(x), x)

    def test_dtypes(self):
        x = np.asarray([1, 2, 3])

        # various dtypes
        for dtype in [T.int8, T.uint8, T.int16, T.int32, T.int64]:
            self.assertIsInstance(dtype, str)
            self.assertFalse(T.is_floating_point(T.as_tensor(0, dtype=dtype)))
            self.assertFalse(T.is_floating_point_dtype(dtype))
            self.assertEqual(T.get_dtype(T.cast(T.as_tensor_backend(x), dtype)), dtype)

        for dtype in [T.float16, T.float32, T.float64]:
            self.assertIsInstance(dtype, str)
            self.assertTrue(T.is_floating_point(T.as_tensor(0, dtype=dtype)))
            self.assertTrue(T.is_floating_point_dtype(dtype))
            self.assertEqual(T.get_dtype(T.cast(T.as_tensor_backend(x), dtype)), dtype)

        # floatx
        self.assertEqual(settings.float_x, 'float32')
        self.assertEqual(T.float_x(), T.float32)
        try:
            settings.float_x = 'float64'
            self.assertEqual(T.float_x(), T.float64)
        finally:
            settings.float_x = 'float32'

        # as_tensor
        t = T.as_tensor_backend(x)
        self.assertIsInstance(t, T.Tensor)
        assert_equal(t, x)

        # cast
        for dtype in number_dtypes:
            t2 = T.cast(t, dtype)
            self.assertIsInstance(t2, T.Tensor)
            self.assertEqual(T.get_dtype(t2), dtype)
            assert_equal(t2, x)

        # cast_like
        for dtype_as in (t, t2):
            t3 = T.cast_like(t, dtype_as)
            self.assertIsInstance(t3, T.Tensor)
            self.assertEqual(T.get_dtype(t3), T.get_dtype(dtype_as))
            assert_equal(t3, x)

    def test_tensor_constructors(self):
        np.random.seed(1234)

        # as_tensor_backend
        for x in [1., 1, [1., 2., 3.], np.array([1., 2., 3.])]:
            t = T.as_tensor_backend(x)
            self.assertIsInstance(t, T.Tensor)
            assert_equal(t, x)

        x = T.as_tensor_backend(np.asarray([1, 2, 3], dtype=np.int32))
        t = T.as_tensor_backend(x)
        self.assertIs(t, x)

        with pytest.raises(Exception):
            _ = T.as_tensor_backend(object())  # not a tensor, should raise error

        # as_tensor
        def copy_tensor(o):
            if isinstance(o, StochasticTensor):
                return StochasticTensor(
                    tensor=T.as_tensor(np.copy(T.to_numpy(o.tensor))),
                    distribution=o.distribution,
                    n_samples=o.n_samples,
                    group_ndims=o.group_ndims,
                    reparameterized=o.reparameterized,
                )
            if isinstance(o, T.Tensor):
                return T.as_tensor(np.copy(T.to_numpy(o)))
            return copy.copy(o)

        stochastic_tensor = UnitNormal(shape=[3]).sample()
        stochastic_tensor.tensor = T.as_tensor([1., 2., 3.])
        for x in [1., 1, [1., 2., 3.], np.array([1., 2., 3.]),
                  T.as_tensor(np.array([1., 2., 3.])),
                  stochastic_tensor]:
            if isinstance(x, StochasticTensor):
                x_value = T.to_numpy(x.tensor)
            elif isinstance(x, T.Tensor):
                x_value = T.to_numpy(x)
            else:
                x_value = copy.copy(x)

            for should_copy in [True, False]:
                for dtype in (None,) + number_dtypes:
                    xx = copy_tensor(x)
                    self.assertIsInstance(xx, type(x))
                    dtype_kwargs = {'dtype': dtype} if dtype is not None else {}

                    t = T.as_tensor(xx, force_copy=should_copy, **dtype_kwargs)
                    self.assertIsInstance(t, T.Tensor)
                    if should_copy:
                        if hasattr(xx, '__setitem__'):
                            xx[0] = 12345.
                    assert_equal(t, x_value,
                                 err_msg=f'{x}, {should_copy}, {dtype}')

        with pytest.raises(Exception):
            _ = T.as_tensor(object())  # not a tensor, should raise error

        # from numpy: force copied
        for x in [np.array([1., 2., 3.])]:
            for dtype in (None,) + number_dtypes:
                xx = copy.copy(x)
                self.assertIsInstance(xx, type(x))
                dtype_kwargs = {'dtype': dtype} if dtype is not None else {}
                t = T.from_numpy(xx, **dtype_kwargs)
                self.assertIsInstance(t, T.Tensor)
                xx[0] = 12345.
                assert_equal(t, x, err_msg=f'{x}, {dtype}')

        with pytest.raises(Exception):
            _ = T.from_numpy(object())  # not a tensor, should raise error

        # float_scalar
        for value in (1.25, 125):
            for dtype in (T.float16, T.float32, T.float64):
                t = T.float_scalar(value, dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                assert_equal(t, value)
        self.assertEqual(T.get_dtype(T.float_scalar(1.25)), T.float_x())

        # int_scalar
        for value in (2, 125):
            for dtype in (T.int8, T.int16, T.int32, T.int64):
                t = T.int_scalar(value, dtype=dtype)
                self.assertEqual(T.get_dtype(t), dtype)
                assert_equal(t, value)
        self.assertEqual(T.get_dtype(T.int_scalar(125)), T.int32)

        # zeros
        for shape in ([1, 2, 3], []):
            for dtype in number_dtypes:
                t = T.zeros(shape, dtype=dtype)
                self.assertIsInstance(t, T.Tensor)
                self.assertEqual(T.get_dtype(t), dtype)
                assert_equal(t, np.zeros(shape))

                # zeros_like
                t2 = T.zeros_like(t)
                self.assertIsInstance(t2, T.Tensor)
                self.assertEqual(T.get_dtype(t2), dtype)
                assert_equal(t, np.zeros(shape))

                for dtype2 in (None,) + number_dtypes:
                    for shape2 in (None, [7, 8]):
                        t2 = T.zeros_like(t, dtype=dtype2, shape=shape2)
                        self.assertIsInstance(t2, T.Tensor)
                        self.assertEqual(T.get_dtype(t2), dtype2 or dtype)
                        assert_equal(t2, np.zeros(shape2 or shape))

        # ones
        for shape in ([1, 2, 3], []):
            for dtype in number_dtypes:
                t = T.ones(shape, dtype=dtype)
                self.assertIsInstance(t, T.Tensor)
                self.assertEqual(T.get_dtype(t), dtype)
                assert_equal(t, np.ones(shape))

                # ones_like
                t2 = T.ones_like(t)
                self.assertIsInstance(t2, T.Tensor)
                self.assertEqual(T.get_dtype(t2), dtype)
                assert_equal(t, np.ones(shape))

                for dtype2 in (None,) + number_dtypes:
                    for shape2 in (None, [7, 8]):
                        t2 = T.ones_like(t, dtype=dtype2, shape=shape2)
                        self.assertIsInstance(t2, T.Tensor)
                        self.assertEqual(T.get_dtype(t2), dtype2 or dtype)
                        assert_equal(t2, np.ones(shape2 or shape))

        # full
        fill_value = 123
        for shape in ([1, 2, 3], []):
            for dtype in number_dtypes:
                t = T.full(shape, fill_value, dtype=dtype)
                self.assertIsInstance(t, T.Tensor)
                self.assertEqual(T.get_dtype(t), dtype)
                assert_equal(t, np.full(shape, fill_value))

                # zeros_like
                t2 = T.full_like(t, fill_value)
                self.assertIsInstance(t2, T.Tensor)
                self.assertEqual(T.get_dtype(t2), dtype)
                assert_equal(t, np.full(shape, fill_value))

                for dtype2 in (None,) + number_dtypes:
                    for shape2 in (None, [7, 8]):
                        t2 = T.full_like(t, fill_value, dtype=dtype2,
                                         shape=shape2)
                        self.assertIsInstance(t2, T.Tensor)
                        self.assertEqual(T.get_dtype(t2), dtype2 or dtype)
                        assert_equal(t2, np.full(shape2 or shape, fill_value))

        # arange
        for start, end in [(1, 10), (0, 10)]:
            t = T.arange(start, end)
            self.assertIsInstance(t, T.Tensor)
            self.assertEqual(T.get_dtype(t), T.int32)
            assert_equal(t, np.arange(start, end))

        for start, end, step in [(0, 10, 2), (-2, -15, -3)]:
            t = T.arange(start, end, step)
            self.assertIsInstance(t, T.Tensor)
            self.assertEqual(T.get_dtype(t), T.int32)
            assert_equal(t, np.arange(start, end, step))

        for dtype in number_dtypes:
            t = T.arange(0, 10, dtype=dtype)
            self.assertIsInstance(t, T.Tensor)
            self.assertEqual(T.get_dtype(t), dtype)
            assert_equal(t, np.arange(10))

        # one_hot
        for n_classes in [1, 5]:
            for shape in [[2, 3, 4], []]:
                I = np.eye(n_classes)
                x = np.random.randint(0, n_classes, size=shape)

                t = T.one_hot(T.as_tensor_backend(x), n_classes)
                assert_equal(t, I[x])

                for dtype in number_dtypes:
                    t = T.one_hot(T.as_tensor_backend(x), n_classes, dtype=dtype)
                    self.assertEqual(T.get_dtype(t), dtype)
                    assert_equal(t, I[x])

                for axis in range(-(len(shape) + 1), len(shape) + 1):
                    t = T.one_hot(T.as_tensor_backend(x), n_classes, axis=axis)
                    expected_t = list(range(0, len(shape)))
                    if axis < 0:
                        expected_t.insert(len(expected_t) + axis + 1, -1)
                    else:
                        expected_t.insert(axis, -1)
                    expected = I[x].transpose(expected_t)
                    assert_equal(t, expected, err_msg=f'shape = {shape}, axis = {axis}')

                for axis in [-(len(shape) + 2), len(shape) + 1]:
                    with pytest.raises(Exception, match='`axis` out of range'):
                        _ = T.one_hot(T.as_tensor_backend(x), n_classes, axis=axis)

    def test_to_numpy(self):
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor_backend(x)
        out = T.to_numpy(t)
        self.assertIsInstance(out, np.ndarray)
        assert_equal(out, x)

        with pytest.raises(TypeError, match='Not a Tensor'):
            _ = T.to_numpy(object())

        x = np.asarray([True, False])
        t = T.as_tensor_backend(x)
        out = T.to_numpy(t)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.dtype, np.bool)
        assert_equal(out, x)

    def test_variable_and_initializer(self):
        def is_requires_grad(t):
            try:
                l = T.reduce_sum(t * t)
                [g] = T.grad([l], [t], allow_unused=False)
                return np.all(T.to_numpy(g) == T.to_numpy(2 * t))
            except Exception:
                return False

        for dtype in number_dtypes:
            t = T.variable([3], dtype=dtype, requires_grad=False)
            self.assertIsInstance(t, T.Variable)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [3])

            t = T.variable([2, 3], dtype=t.dtype, requires_grad=False)
            self.assertIsInstance(t, T.Variable)
            self.assertEqual(T.get_dtype(t), dtype)
            self.assertEqual(T.shape(t), [2, 3])

        for dtype in float_dtypes:
            # scalar initializer
            for v in (123, 123., np.array(123.)):
                for requires_grad in (True, False):
                    t = T.variable([3], dtype=dtype, requires_grad=requires_grad,
                                   initializer=v)
                    self.assertIsInstance(t, T.Variable)
                    self.assertEqual(T.get_dtype(t), dtype)
                    self.assertEqual(is_requires_grad(t), requires_grad)
                    assert_equal(t, np.array([v] * 3))

            # array initializer
            for requires_grad in (True, False):
                t = T.variable([3], dtype=dtype, requires_grad=requires_grad,
                               initializer=np.array([1., 2., 3.]))
                self.assertIsInstance(t, T.Variable)
                self.assertEqual(T.get_dtype(t), dtype)
                self.assertEqual(is_requires_grad(t), requires_grad)
                assert_equal(t, [1., 2., 3.])

            with pytest.raises(ValueError,
                               match=r'`initializer.shape` != `shape`: '
                                     r'\[3\] vs \[4\]'):
                _ = T.variable([4], dtype=dtype, requires_grad=False,
                               initializer=np.array([1., 2., 3.]))

            # callable initializer
            for requires_grad in (True, False):
                t = T.variable([3], dtype=dtype, requires_grad=requires_grad,
                               initializer=partial(T.fill, fill_value=123.))
                self.assertIsInstance(t, T.Variable)
                self.assertEqual(T.get_dtype(t), dtype)
                self.assertEqual(is_requires_grad(t), requires_grad)
                assert_equal(t, [123.] * 3)

        # unsupported initializer
        with pytest.raises(TypeError, match='Unsupported initializer'):
            _ = T.variable([3], dtype=T.float32, initializer=object())

    def test_assignment(self):
        with T.no_grad():
            # fill
            for dtype in float_dtypes:
                x = T.variable([3], dtype=dtype)
                self.assertIs(T.fill(x, 123), x)
                assert_equal(x, np.full([3], 123))

            # fill_zero
            for dtype in float_dtypes:
                x = T.variable([3], dtype=dtype, initializer=0.)
                self.assertIs(T.fill_zeros(x), x)
                assert_equal(x, np.zeros([3]))

            # assign
            for dtype in float_dtypes:
                x = T.variable([3], dtype=dtype)
                y_value = np.random.rand(3).astype(dtype)
                y_tensor = T.from_numpy(np.copy(y_value), dtype=dtype)
                self.assertIs(T.assign(x, y_tensor), x)
                y_tensor[0] = 123.0
                assert_equal(x, y_value)

                with pytest.raises(Exception,
                                   match='`dst.shape` != `src.shape`'):
                    T.assign(x, T.zeros([2], dtype=dtype))

            # assign_data
            for dtype in float_dtypes:
                y_value = np.random.rand(3).astype(dtype)

                # from numpy
                x = T.variable([3], dtype=dtype)
                y_copied = np.copy(y_value)
                self.assertIs(T.assign_data(x, y_copied), x)
                y_copied[0] = 123.
                assert_equal(x, y_value)

                with pytest.raises(Exception,
                                   match='`dst.shape` != `src.shape`'):
                    T.assign_data(x, np.zeros([2], dtype=dtype))

                # from tensor
                x = T.variable([3], dtype=dtype)
                y_tensor = T.from_numpy(np.copy(y_value), dtype=dtype)
                self.assertIs(T.assign_data(x, y_tensor), x)
                y_tensor[0] = 123.0
                assert_equal(x, y_value)

                with pytest.raises(Exception,
                                   match='`dst.shape` != `src.shape`'):
                    T.assign_data(x, T.zeros([2], dtype=dtype))

    def test_shape_utils(self):
        # test shape
        x = np.random.randn(2, 3, 4)
        t = T.as_tensor_backend(x)
        s = T.shape(t)
        self.assertEqual(s, [2, 3, 4])

        # test rank
        self.assertEqual(T.rank(t), 3)

        # test reshape
        t2 = T.reshape(t, [3, 8])
        self.assertEqual(T.shape(t2), [3, 8])
        assert_equal(t2, np.reshape(x, [3, 8]))

        with pytest.raises(Exception):
            _ = T.reshape(t, [4, 8])

        # test repeat
        x = np.random.randn(2, 1, 3)
        t = T.as_tensor_backend(x)

        t2 = T.repeat(t, [])
        self.assertEqual(T.shape(t2), [2, 1, 3])
        assert_equal(t2, x)

        t2 = T.repeat(t, [2])
        self.assertEqual(T.shape(t2), [2, 1, 6])
        assert_equal(t2, np.tile(x, [1, 1, 2]))

        t2 = T.repeat(t, [4, 3, 2])
        self.assertEqual(T.shape(t2), [8, 3, 6])
        assert_equal(t2, np.tile(x, [4, 3, 2]))

        t2 = T.repeat(t, [4, 1, 3, 1])
        self.assertEqual(T.shape(t2), [4, 2, 3, 3])
        assert_equal(t2, np.tile(x, [4, 1, 3, 1]))

        t2 = T.repeat(t, [5, 4, 3, 2])
        self.assertEqual(T.shape(t2), [5, 8, 3, 6])
        assert_equal(t2, np.tile(x, [5, 4, 3, 2]))

        # test expand
        t2 = T.expand(t, [4, -1, 5, -1])
        self.assertEqual(T.shape(t2), [4, 2, 5, 3])
        assert_equal(t2, np.tile(x, [4, 1, 5, 1]))

        # test squeeze
        x = np.random.randn(1, 2, 1, 3, 1, 4, 1)
        t = T.as_tensor_backend(x)

        t2 = T.squeeze(T.as_tensor_backend(x))
        s2 = [2, 3, 4]
        self.assertEqual(T.shape(t2), s2)
        assert_equal(t2, x.reshape(s2))

        t2 = T.squeeze(t, [-1])
        s2 = [1, 2, 1, 3, 1, 4]
        self.assertEqual(T.shape(t2), s2)
        assert_equal(t2, x.reshape(s2))

        t2 = T.squeeze(t, [-1, 0, 4, 6])
        s2 = [2, 1, 3, 4]
        self.assertEqual(T.shape(t2), s2)
        assert_equal(t2, x.reshape(s2))

        with pytest.raises(Exception, match='Axis .* cannot be squeezed'):
            _ = T.squeeze(t, [-1, -2])

        # test expand dim
        x = np.random.randn(2, 3)
        t = T.as_tensor_backend(x)

        t2 = T.expand_dim(t, -1)
        s2 = [2, 3, 1]
        self.assertEqual(T.shape(t2), s2)
        assert_equal(t2, x.reshape(s2))

        t2 = T.expand_dim(t, -2)
        s2 = [2, 1, 3]
        self.assertEqual(T.shape(t2), s2)
        assert_equal(t2, x.reshape(s2))

        t2 = T.expand_dim(t, 0)
        s2 = [1, 2, 3]
        self.assertEqual(T.shape(t2), s2)
        assert_equal(t2, x.reshape(s2))

        # test swap_axes
        x = np.random.randn(2, 3, 4)
        for i in range(-x.ndim, x.ndim):
            for j in range(-x.ndim, x.ndim):
                assert_equal(
                    T.swap_axes(T.as_tensor(x), i, j),
                    np.swapaxes(x, i, j)
                )

        # test transpose
        x = np.random.randn(2, 3, 4)
        indices = [0, 1, 2]
        for perm_indices in itertools.permutations(indices):
            for neg_mark in itertools.product([True] * len(indices),
                                              [False] * len(indices)):
                perm = list(perm_indices)
                for i, m in enumerate(neg_mark):
                    if m:
                        perm[i] -= len(indices)
                assert_equal(
                    T.transpose(T.as_tensor(x), perm),
                    np.transpose(x, perm),
                )

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
        t = T.as_tensor_backend(x)

        t2 = T.broadcast_to(t, [4, 5, 2, 1])
        self.assertEqual(T.shape(t2), [4, 5, 2, 1])
        assert_equal(t2, np.tile(x.reshape([1, 1, 2, 1]), [4, 5, 1, 1]))

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
            out1, out2 = T.explicit_broadcast(T.as_tensor_backend(x), T.as_tensor_backend(y))
            out1 = T.to_numpy(out1)
            out2 = T.to_numpy(out2)
            ans1, ans2 = explicit_broadcast(x, y)
            assert_equal(out1, ans1)
            assert_equal(out2, ans2)

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
                    front_shape = list(x.shape)
                    xx = x.reshape([-1])
                else:
                    front_shape = list(x.shape)[: -(k - 1)]
                    xx = x.reshape([-1] + list(x.shape)[-(k - 1):])

                tt, s1 = T.flatten_to_ndims(t, k)
                self.assertEqual(s1, front_shape)
                assert_equal(tt, xx)
                assert_equal(T.unflatten_from_ndims(tt, s1), x)

        x = np.arange(120)
        run_check(x, 1)

        x = np.arange(120).reshape([2, 3, 4, 5]).astype(np.int32)
        run_check(x, 1)
        run_check(x, 2)
        run_check(x, 3)
        run_check(x, 4)

        with pytest.raises(Exception,
                           match='`ndims` must be at least 1'):
            _ = T.flatten_to_ndims(T.as_tensor_backend([0.]), 0)

        with pytest.raises(Exception, match=r'rank\(x\) < ndims'):
            _ = T.flatten_to_ndims(T.zeros([3, 4]), 3)

        with pytest.raises(Exception, match=r'rank\(x\) < ndims'):
            _ = T.flatten_to_ndims(T.zeros([3]), 2)

        with pytest.raises(Exception,
                           match=r'Invalid input: rank\(x\) < 1, but '
                                 r'front_shape is not None'):
            t = T.as_tensor_backend(123)
            _ = T.unflatten_from_ndims(t, [2, 3])

        # test reshape_tail
        x = np.random.randn(2, 12, 1)
        x_t = T.as_tensor(x)
        assert_equal(T.reshape_tail(x_t, 0, []), x)
        assert_equal(T.reshape_tail(x_t, 0, [1]), x.reshape([2, 12, 1, 1]))
        assert_equal(T.reshape_tail(x_t, 0, [-1]), x.reshape([2, 12, 1, 1]))

        assert_equal(T.reshape_tail(x_t, 1, []), x.reshape([2, 12]))
        assert_equal(T.reshape_tail(x_t, 1, [1, 1]), x.reshape([2, 12, 1, 1]))
        assert_equal(T.reshape_tail(x_t, 1, [-1]), x.reshape([2, 12, 1]))
        assert_equal(T.reshape_tail(x_t, 1, [-1, 1]), x.reshape([2, 12, 1, 1]))

        assert_equal(T.reshape_tail(x_t, 2, [12]), x.reshape([2, 12]))
        assert_equal(T.reshape_tail(x_t, 2, [-1]), x.reshape([2, 12]))
        assert_equal(T.reshape_tail(x_t, 2, [3, 4]), x.reshape([2, 3, 4]))
        assert_equal(T.reshape_tail(x_t, 2, [3, -1]), x.reshape([2, 3, 4]))
        assert_equal(T.reshape_tail(x_t, 2, [-1, 4]), x.reshape([2, 3, 4]))

        assert_equal(T.reshape_tail(x_t, 3, [24]), x.reshape([24]))
        assert_equal(T.reshape_tail(x_t, 3, [-1]), x.reshape([24]))
        assert_equal(T.reshape_tail(x_t, 3, [4, -1]), x.reshape([4, 6]))
        assert_equal(T.reshape_tail(x_t, 3, [-1, 6]), x.reshape([4, 6]))

        with pytest.raises(Exception,
                           match='`input` must be at least `ndims`-dimensional'):
            _ = T.reshape_tail(x_t, 4, [-1])

    def test_index_select_and_others(self):
        # index_select
        x = np.random.randn(3, 4, 5)
        t = T.as_tensor_backend(x)

        assert_equal(
            T.index_select(t, T.as_tensor_backend(1), 0),
            x[1, ...]
        )
        assert_equal(
            T.index_select(t, T.as_tensor_backend(3), 1),
            x[:, 3, ...]
        )
        assert_equal(
            T.index_select(t, T.as_tensor_backend(2), -1),
            x[..., 2]
        )

        i = np.asarray([0, 2, 1, 1, 0, 2])
        assert_equal(
            T.index_select(t, T.as_tensor_backend(i), 0),
            x[i, ...]
        )
        assert_equal(
            T.index_select(t, T.as_tensor_backend(i), 1),
            x[:, i, ...]
        )
        assert_equal(
            T.index_select(t, T.as_tensor_backend(i), -1),
            x[..., i]
        )

        i = np.asarray([[0, 2, 1], [1, 0, 2]])
        assert_equal(
            T.index_select(t, T.as_tensor_backend(i), 0),
            x[i, ...]
        )
        assert_equal(
            T.index_select(t, T.as_tensor_backend(i), 1),
            x[:, i, ...]
        )
        assert_equal(
            T.index_select(t, T.as_tensor_backend(i), -1),
            x[..., i]
        )

        if T.backend_name != 'PyTorch':
            # TODO: pytorch currently does not support negative index in many
            # of its functions.  enable these test when supported.
            assert_equal(
                T.index_select(t, T.as_tensor_backend(-1), 1),
                x[:, -1]
            )

            i = np.asarray([0, 1, -1, 2, -2, 0])
            assert_equal(
                T.index_select(t, T.as_tensor_backend(i), 1),
                x[:, i, ...]
            )

            i = np.asarray([[0, 1, -1], [2, -2, 0]])
            assert_equal(
                T.index_select(t, T.as_tensor_backend(i), 1),
                x[:, i, ...]
            )

        with pytest.raises(Exception, match='`axis` out of range'):
            _ = T.index_select(t, T.as_tensor_backend(0), 3)

        with pytest.raises(Exception, match='`axis` out of range'):
            _ = T.index_select(t, T.as_tensor_backend(0), -4)

        # concat
        x = np.random.randn(2, 3, 4)
        y = np.random.randn(2, 5, 4)
        z = np.random.randn(2, 3, 5)

        for arrays, axis in [([x, x, y], -2), ([x, y, y], 1),
                             ([x, x, z], -1), ([x, z, z], 2)]:
            t = T.concat([T.as_tensor_backend(arr) for arr in arrays], axis=axis)
            expected = np.concatenate(arrays, axis=axis)
            assert_equal(t, expected)

        # split
        x = np.random.randn(2, 3, 4, 5)

        for sections, axis in [([1, 2], 1),
                               ([1, 2], -3),
                               ([2], 0),
                               ([3, 2], 3),
                               ([1, 1, 1, 2], -1)]:
            a = T.split(T.as_tensor(x), sections, axis=axis)
            if len(sections) > 1:
                split_positions = []
                start_pos = 0
                for section in sections[:-1]:
                    split_positions.append(start_pos + section)
                    start_pos += section
                b = np.split(x, split_positions, axis=axis)
            else:
                b = [x]

            err_msg = f'sections {sections}, axis {axis}'
            self.assertEqual(len(a), len(b), msg=err_msg)
            for aa, bb in zip(a, b):
                assert_equal(aa, bb, err_msg=err_msg)

        with pytest.raises(Exception):
            print(T.split(T.as_tensor(x), [], axis=0))
        with pytest.raises(Exception):
            print(T.split(T.as_tensor(x), [1], axis=0))

        # stack
        arrays = [np.random.randn(2, 3, 4, 5) for _ in range(3)]
        for k in range(1, len(arrays)):
            for axis in range(-arrays[0].ndim - 1, arrays[0].ndim + 1):
                assert_equal(
                    T.stack([T.as_tensor(a) for a in arrays[:k]], axis),
                    np.stack(arrays[:k], axis)
                )

        # unstack
        array = np.random.randn(1, 2, 3, 4, 5)
        for axis in range(-len(array.shape), len(array.shape)):
            pos_axis = axis if axis >= 0 else axis + len(array.shape)
            sections = T.unstack(T.as_tensor(array), axis=axis)
            expected = np.split(array, array.shape[axis], axis=axis)
            self.assertEqual(len(sections), array.shape[axis])
            for i, section in enumerate(sections):
                sec_shape = list(array.shape)[:pos_axis] + list(array.shape[pos_axis + 1:])
                self.assertEqual(T.shape(section), sec_shape)
                assert_equal(section, expected[i].reshape(sec_shape))

        # slice
        x = np.random.randn(3, 4)
        x_t = T.as_tensor(x)
        assert_equal(T.slice(x_t, []), x)
        assert_equal(T.slice(x_t, [1]), x[:, 1:])
        assert_equal(T.slice(x_t, [1, -2]), x[1:, -2:])
        assert_equal(T.slice(x_t, [0, 0]), x)

        assert_equal(T.slice(x_t, [], []), x)
        assert_equal(T.slice(x_t, [1], [1]), x[:, 1:2])
        assert_equal(T.slice(x_t, [-1, 2], [1, 2]), x[2:3, 2:4])
        assert_equal(T.slice(x_t, [0, 0], [0, 0]), x[0:0, 0:0])

        with pytest.raises(Exception,
                           match=r'`len\(slice_start\)` must be less or equal to '
                                 r'`rank\(input\)`'):
            _ = T.slice(x_t, [1, 1, 1])

        with pytest.raises(Exception,
                           match=r'`len\(slice_start\)` != `len\(slice_length\)`'):
            _ = T.slice(x_t, [1, 1], [2])

        # slice_axis
        assert_equal(T.slice_axis(x_t, -2, 1), x[1:])
        assert_equal(T.slice_axis(x_t, -1, 2, 1), x[:, 2:3])
        assert_equal(T.slice_axis(x_t, 0, 2, 1), x[2:3])
        assert_equal(T.slice_axis(x_t, 1, 1), x[:, 1:])

        for axis in (-3, 2):
            with pytest.raises(Exception):
                _ = T.slice_axis(x_t, axis, 0)

        # pad
        assert_equal(T.pad(x_t, []), x)
        assert_equal(
            T.pad(x_t, [(1, 2)]),
            np.pad(x, [[0, 0], [1, 2]], mode='constant', constant_values=0.)
        )
        assert_equal(
            T.pad(x_t, [(3, 4), (1, 2)]),
            np.pad(x, [[3, 4], [1, 2]], mode='constant', constant_values=0.)
        )
        assert_equal(
            T.pad(x_t, [(1, 2)], value=1.),
            np.pad(x, [[0, 0], [1, 2]], mode='constant', constant_values=1.)
        )
        with pytest.raises(Exception):
            T.pad(x_t, [(1, 2), 3])
        with pytest.raises(Exception):
            T.pad(x_t, [(1, 2), (3, 4), (5, 6)])

        # pad_axis
        for axis in [-2, -1, 0, 1]:
            assert_equal(T.pad_axis(x_t, axis, (0, 0)), x)
            padding = [[0, 0], [0, 0]]
            padding[axis] = [1, 2]
            assert_equal(
                T.pad_axis(x_t, axis, (1, 2)),
                np.pad(x, padding, mode='constant', constant_values=0.)
            )
            assert_equal(
                T.pad_axis(x_t, axis, (1, 2), 123.),
                np.pad(x, padding, mode='constant', constant_values=123.)
            )

        for axis in (-3, 2):
            with pytest.raises(Exception):
                _ = T.pad_axis(x_t, axis, (0, 0))

        # shift
        assert_equal(T.shift(x_t, []), x)
        assert_equal(T.shift(x_t, [0, 0]), x)
        assert_equal(
            T.shift(x_t, [1]),
            np.pad(x[:, :-1], [[0, 0], [1, 0]], mode='constant', constant_values=0.)
        )
        assert_equal(
            T.shift(x_t, [-1, 0]),
            np.pad(x[1:, :], [[0, 1], [0, 0]], mode='constant', constant_values=0.)
        )
        assert_equal(
            T.shift(x_t, [-2, 1], fill_value=123.),
            np.pad(x[2:, :-1], [[0, 2], [1, 0]], mode='constant', constant_values=123.)
        )

        with pytest.raises(Exception,
                           match=r'`len\(shift\) <= rank\(input\)` does not hold'):
            _ = T.shift(x_t, [1, 2, 3])

        with pytest.raises(Exception,
                           match=r'`shift` out of range at axis .*'):
            _ = T.shift(x_t, [4, 0])

        with pytest.raises(Exception,
                           match=r'`shift` out of range at axis .*'):
            _ = T.shift(x_t, [0, -5])

        # shift_axis
        for axis in (-2, -1, 0, 1):
            assert_equal(T.shift_axis(x_t, axis, 0), x)
        assert_equal(
            T.shift_axis(x_t, -2, 1),
            np.pad(x[:-1, :], [[1, 0], [0, 0]], mode='constant', constant_values=0.)
        )
        assert_equal(
            T.shift_axis(x_t, -1, -2),
            np.pad(x[:, 2:], [[0, 0], [0, 2]], mode='constant', constant_values=0.)
        )
        assert_equal(
            T.shift_axis(x_t, 0, -2, fill_value=123.),
            np.pad(x[2:, :], [[0, 2], [0, 0]], mode='constant', constant_values=123.)
        )
        assert_equal(
            T.shift_axis(x_t, 1, 1, fill_value=123.),
            np.pad(x[:, :-1], [[0, 0], [1, 0]], mode='constant', constant_values=123.)
        )

        with pytest.raises(Exception, match='`shift` out of range'):
            _ = T.shift_axis(x_t, -2, 4)

        with pytest.raises(Exception, match='`shift` out of range'):
            _ = T.shift_axis(x_t, 1, -5)

        for axis in (-3, 2):
            with pytest.raises(Exception):
                _ = T.shift_axis(x_t, axis, 0)

    def test_math_univariate_op(self):
        np.random.seed(1234)

        x = np.random.randn(2, 3)
        u = np.random.rand(2, 3)
        x_t = T.as_tensor_backend(x)
        u_t = T.as_tensor_backend(u)

        assert_allclose(T.floor(x_t), np.floor(x))
        assert_allclose(T.ceil(x_t), np.ceil(x))
        assert_allclose(T.abs(x_t), np.abs(x))
        assert_allclose(T.neg(x_t), -x)
        assert_allclose(T.square(x_t), x ** 2)

        assert_allclose(T.exp(x_t), np.exp(x))
        assert_allclose(T.log(T.as_tensor_backend(np.abs(x))),
                        np.log(np.abs(x)))
        assert_allclose(T.log1p(T.as_tensor_backend(np.abs(x) - 1. + 1e-7)),
                        np.log1p(np.abs(x) - 1. + 1e-7))

        assert_allclose(T.sin(x_t), np.sin(x))
        assert_allclose(T.cos(x_t), np.cos(x))
        assert_allclose(T.tan(x_t), np.tan(x))

        assert_allclose(T.tanh(x_t), np.tanh(x))

        assert_allclose(T.erf(x_t), erf(x))
        assert_allclose(T.erfc(x_t), erfc(x))
        assert_allclose(T.erfinv(u_t), erfinv(u))

    def test_math_bivariate_op(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3)
        y = np.random.randn(3)
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)

        assert_allclose(T.add(t1, t2), x + y)
        assert_allclose(T.sub(t1, t2), x - y)
        assert_allclose(T.mul(t1, t2), x * y)
        assert_allclose(T.pow(T.as_tensor_backend(np.abs(x)), t2),
                        np.abs(x) ** y)
        assert_allclose(T.sqrt(T.as_tensor_backend(np.abs(x))), np.sqrt(np.abs(x)))

        # for division, of course y should not equal to zero
        y = np.asarray(y == 0, dtype=y.dtype) + y
        assert_allclose(T.div(t1, t2), x / y)
        assert_allclose(T.truediv(t1, t2), x / y)

        # for floordiv and mod, we only require the backend tensor engine
        # to produce identical results with numpy when x > 0 and y > 0
        x = np.abs(x)
        y = np.abs(y)
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)
        assert_allclose(T.floordiv(t1, t2), x // y)
        assert_allclose(T.mod(t1, t2), x % y)

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
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)
        out = T.truediv(t1, t2)
        self.assertEqual(T.get_dtype(out), T.float32)
        assert_allclose(out, x.astype(np.float32) / y.astype(np.float32))

        # input int16, output float32
        x = np.random.randint(-32768, 32767, size=(2, 3), dtype=np.int16)
        y = np.random.randint(-32768, 32767, size=(3,), dtype=np.int16)
        y = y + (y == 0).astype(y.dtype)
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)
        out = T.truediv(t1, t2)
        self.assertEqual(T.get_dtype(out), T.float32)
        assert_allclose(out, x.astype(np.float32) / y.astype(np.float32))

        # input int32, output float64
        x = np.random.randint(-100000, 100000, size=(2, 3), dtype=np.int32)
        y = np.random.randint(-100000, 100000, size=(3,), dtype=np.int32)
        y = y + (y == 0).astype(y.dtype)
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)
        out = T.truediv(t1, t2)
        self.assertEqual(T.get_dtype(out), T.float64)
        assert_allclose(out, x.astype(np.float64) / y.astype(np.float64))

    def test_math_sequential_op(self):
        # test add_n
        x = np.random.randn(2, 3)
        y = np.random.randn(3)
        z = np.random.randn(2, 1)

        assert_allclose(
            T.add_n([T.as_tensor_backend(t) for t in (x, y, z)]),
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
        t = T.as_tensor_backend(x)

        # test sum, mean, max, min
        for name in ['sum', 'mean', 'min', 'max',
                     'log_sum_exp', 'log_mean_exp']:
            T_op = getattr(T, 'reduce_' + name, getattr(T, name, None))
            np_op = getattr(np, name,
                            {
                                'log_sum_exp': log_sum_exp,
                                'log_mean_exp': log_mean_exp,
                            }.get(name))

            assert_allclose(T_op(t), np_op(x))
            assert_allclose(T_op(t, keepdims=True), np_op(x, keepdims=True))
            assert_allclose(T_op(t, axis=[-1]), np_op(x, axis=-1))
            assert_allclose(T_op(t, axis=[-1], keepdims=True),
                            np_op(x, axis=-1, keepdims=True))
            assert_allclose(T_op(t, axis=[0, -1]), np_op(x, axis=(0, -1)))
            assert_allclose(T_op(t, axis=[0, -1], keepdims=True),
                            np_op(x, axis=(0, -1), keepdims=True))

            if not name.startswith('log_'):
                assert_allclose(T_op(t, axis=[]), np_op(x, axis=()))
                assert_allclose(T_op(t, axis=[], keepdims=True),
                                np_op(x, axis=(), keepdims=True))
            else:
                with pytest.raises(Exception,
                                   match='`axis` must not be an empty list'):
                    _ = T_op(t, axis=[])

        # test argmax, argmin
        def np_argmaxmin(fn, x, axis, keepdims=False):
            r_shape = list(x.shape)
            r_shape[axis] = 1
            r = fn(x, axis)
            if keepdims:
                r = r.reshape(r_shape)
            return r

        for name in ['argmax', 'argmin']:
            T_op = getattr(T, name, getattr(T, name, None))
            np_op = partial(np_argmaxmin, getattr(np, name))

            for axis in (0, 1, 2, -1, -2, -3):
                assert_allclose(T_op(t, axis=axis), np_op(x, axis=axis))
                assert_allclose(T_op(t, axis=axis, keepdims=True),
                                np_op(x, axis=axis, keepdims=True))

        # test calculate_mean_and_var
        x = np.random.randn(3, 4, 5)
        for dtype in float_dtypes:
            x_t = T.as_tensor(x, dtype=dtype)
            for axis in ([-3], [2], [-1], [2], [-1, -2], None):
                for keepdims in [True, False]:
                    for unbiased in [True, False]:
                        mean_t, var_t = T.calculate_mean_and_var(
                            x_t, axis=axis, keepdims=keepdims,
                            unbiased=unbiased
                        )
                        np_axis = axis and tuple(axis)
                        msg = f'axis={axis}, unbiased={unbiased}, ' \
                              f'keepdims={keepdims}, dtype={dtype}'
                        mean = np.mean(x, axis=np_axis, keepdims=keepdims)
                        var = np.var(x, axis=np_axis, keepdims=keepdims,
                                     ddof=float(int(unbiased)))
                        assert_allclose(mean_t, mean, rtol=1e-4, err_msg=msg)
                        assert_allclose(var_t, var, rtol=1e-4, err_msg=msg)

            with pytest.raises(Exception,
                               match='Variance can only be calculated with at '
                                     'least 2 samples'):
                _ = T.calculate_mean_and_var(x_t, axis=[])

        with pytest.raises(Exception,
                           match='Variance can only be calculated with at '
                                 'least 2 samples'):
            _ = T.calculate_mean_and_var(T.zeros(shape=[]))

        # test norm_except_axis
        x = np.random.randn(3, 4, 5)
        for dtype in float_dtypes:
            x_t = T.as_tensor(x, dtype=dtype)
            for axis, p, keepdims in itertools.product(
                        ([-3], [2], [-1], [2], [-1, -2], None),
                        (-2.0, -1.5, -1.0, 0.5, 1.0, 1.5, 2.0, 3.0),
                        (True, False),
                    ):
                assert_allclose(
                    T.norm_except_axis(x_t, axis, p, keepdims),
                    norm_except_axis(x, axis, p, keepdims),
                    err_msg=f'axis={axis}, p={p}, keepdims={keepdims}',
                    rtol=1e-4, atol=1e-6
                )

            for axis in ([-4], [3], [-1, -4], [0, 3]):
                with pytest.raises(Exception, match='`axis` out of range'):
                    _ = T.norm_except_axis(x_t, axis=axis)

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
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)

        # test as_boolean
        self.assertEqual(T.get_dtype(t1), T.boolean)
        assert_equal(read_bool(t1), x)

        # test logical_not
        out = T.logical_not(t1)
        assert_equal(read_bool(out), np.logical_not(x))
        with_raise('x', lambda: T.logical_not(T.as_tensor_backend([1, 2, 3])))

        # test logical_and
        out = T.logical_and(t1, t2)
        assert_equal(read_bool(out), np.logical_and(x, y))
        with_raise('x', lambda: T.logical_and(T.as_tensor_backend([1, 2, 3, 4]), t2))
        with_raise('y', lambda: T.logical_and(t1, T.as_tensor_backend([1, 2, 3, 4])))

        # test logical_or
        out = T.logical_or(t1, t2)
        assert_equal(read_bool(out), np.logical_or(x, y))
        with_raise('x', lambda: T.logical_or(T.as_tensor_backend([1, 2, 3, 4]), t2))
        with_raise('y', lambda: T.logical_or(t1, T.as_tensor_backend([1, 2, 3, 4])))

        # test logical_xor
        out = T.logical_xor(t1, t2)
        assert_equal(read_bool(out), np.logical_xor(x, y))
        with_raise('x', lambda: T.logical_xor(T.as_tensor_backend([1, 2, 3, 4]), t2))
        with_raise('y', lambda: T.logical_xor(t1, T.as_tensor_backend([1, 2, 3, 4])))

        # test multiply_mask
        def test_multiply_mask(x, y, dtype, mask_dtype):
            t = T.multiply_mask(
                T.as_tensor(x, dtype=dtype),
                T.as_tensor(y, dtype=mask_dtype)
            )
            self.assertEqual(T.get_dtype(t), dtype)
            assert_allclose(t, np.asarray(x * np.asarray(y, dtype=x.dtype),
                                          dtype=dtype))

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
                    T.as_tensor(condition, dtype=T.boolean),
                    T.as_tensor(x, dtype=dtype),
                    T.as_tensor(y, dtype=dtype),
                )
                self.assertEqual(T.get_dtype(ret), dtype)
                assert_equal(ret, expected)
            else:
                expected = np.where(condition)
                self.assertEqual(len(expected), len(condition.shape))
                ret = T.where(T.as_tensor(condition, dtype=T.boolean))
                self.assertEqual(len(ret), len(condition.shape))
                for a, b in zip(ret, expected):
                    assert_equal(a, b)

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
        t1 = T.as_tensor_backend(x)
        t2 = T.as_tensor_backend(y)

        # test equal
        assert_equal(read_bool(T.equal(t1, t2)), (x == y))

        # test not_equal
        assert_equal(read_bool(T.not_equal(t1, t2)), (x != y))

        # test less
        assert_equal(read_bool(T.less(t1, t2)), (x < y))

        # test less_equal
        assert_equal(read_bool(T.less_equal(t1, t2)), (x <= y))

        # test greater
        assert_equal(read_bool(T.greater(t1, t2)), (x > y))

        # test greater_equal
        assert_equal(read_bool(T.greater_equal(t1, t2)), (x >= y))

        # test minimum
        assert_equal(T.minimum(t1, t2), np.minimum(x, y))

        # test maximum
        assert_equal(T.maximum(t1, t2), np.maximum(x, y))

        # test clip
        self.assertTrue(np.any(x < -0.5))
        self.assertTrue(np.any(x > 0.5))
        assert_equal(T.clip(t1, -0.5, 0.5), np.clip(x, -0.5, 0.5))

        # test maybe_clip
        assert_equal(T.maybe_clip(t1), x)
        assert_equal(T.maybe_clip(t1, -0.5, 0.5), np.clip(x, -0.5, 0.5))
        assert_equal(T.maybe_clip(t1, x_min=-0.5), np.maximum(x, -0.5))
        assert_equal(T.maybe_clip(t1, x_max=0.5), np.minimum(x, 0.5))
        assert_equal(T.maybe_clip(t1, x_min=-0.5, x_max=None), np.maximum(x, -0.5))
        assert_equal(T.maybe_clip(t1, x_min=None, x_max=0.5), np.minimum(x, 0.5))

    def test_sort(self):
        x = np.random.randn(5, 6, 7)
        x_t = T.as_tensor(x)

        for axis in range(-len(x.shape), len(x.shape)):
            assert_equal(
                T.sort(x_t, axis=axis, descending=False),
                np.sort(x, axis=axis)
            )
            assert_equal(
                T.sort(x_t, axis=axis, descending=True),
                -np.sort(-x, axis=axis)
            )
            assert_equal(
                T.argsort(x_t, axis=axis, descending=False),
                np.argsort(x, axis=axis)
            )
            assert_equal(
                T.argsort(x_t, axis=axis, descending=True),
                np.argsort(-x, axis=axis)
            )

    def test_matrix_ops(self):
        np.random.seed(1234)

        for k in [1, 5]:
            x = np.random.randn(4, k)
            y = np.random.randn(k, k)
            z = np.random.randn(k, 3)

            for a, b in [(x, y), (y, z), (x, z)]:
                assert_allclose(T.matmul(T.as_tensor(a), T.as_tensor(b)),
                                np.dot(a, b))

            yy = np.linalg.qr(y)[0]
            assert_allclose(T.matrix_inverse(T.as_tensor(yy)),
                            np.linalg.inv(yy))

    def test_gradient(self):
        x = np.random.randn(2, 3, 4)
        y = np.random.randn(2, 3, 4)

        # requires_grad
        yt = T.requires_grad(T.as_tensor_backend(y))

        xt = T.as_tensor_backend(x)
        xt_copy = T.requires_grad(xt, copy=False)
        self.assertIs(xt_copy, xt)
        l_sum = T.reduce_sum(xt + xt_copy)
        # xtt and xt are the same tensor, thus gradient should pass along the both paths
        [x_grad] = T.grad([l_sum], [xt_copy])
        assert_allclose(x_grad, np.full_like(x, 2))

        xt_copy = T.requires_grad(xt, copy=True)
        self.assertIsNot(xt_copy, xt)
        l_sum = T.reduce_sum(xt + xt_copy)
        # xttt is a copy of xt, thus grad should pass to xt along both paths
        # when taking derivative against xt
        [x_grad] = T.grad([l_sum], [xt])
        assert_allclose(x_grad, np.full_like(x, 2))
        # but grad should not pass to xt if taking derivative against xttt
        [x_grad] = T.grad([l_sum], [xt_copy])
        assert_allclose(x_grad, np.full_like(x, 1))

        # grad
        l_sum = T.reduce_sum(xt * yt)
        l_squares = 7 * xt ** 3 + 11 * yt ** 3

        [x_grad, y_grad] = T.grad(
            [l_sum, l_squares],
            [xt, yt],
            grad_outputs=[None, T.ones_like(l_squares)],
            retain_graph=True,
            create_graph=True
        )
        assert_allclose(x_grad, y + 21 * x ** 2)
        assert_allclose(y_grad, x + 33 * y ** 2)

        # second order grad
        [x_grad_2, y_grad_2] = T.grad(
            [x_grad, y_grad],
            [xt, yt],
            grad_outputs=[T.ones_like(xt), T.ones_like(yt)],
            retain_graph=True,
            create_graph=False
        )
        assert_allclose(x_grad_2, 42. * x + 1.)
        assert_allclose(y_grad_2, 66. * y + 1.)

        # get the first order grad again, but once for each of x and y
        [x_grad] = T.grad(
            [l_sum, l_squares],
            [xt],
            grad_outputs=[None, T.ones_like(l_squares)],
            retain_graph=True,
            create_graph=True
        )
        assert_allclose(x_grad, y + 21 * x ** 2)

        [y_grad] = T.grad(
            [l_sum, l_squares],
            [yt],
            grad_outputs=[None, T.ones_like(l_squares)],
            retain_graph=True,
            create_graph=True
        )
        assert_allclose(y_grad, x + 33 * y ** 2)

        # stop_grad
        l_sum = T.reduce_sum(T.stop_grad(xt ** 2) * yt)
        [x_grad, y_grad] = T.grad(
            [l_sum],
            [xt, yt],
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )
        self.assertTrue(T.is_null_grad(xt, x_grad))
        self.assertFalse(T.is_null_grad(yt, y_grad))
        assert_allclose(y_grad, x ** 2)

        # is_null_grad counterexample
        self.assertFalse(T.is_null_grad(T.zeros([]), T.zeros([])))
        self.assertFalse(T.is_null_grad(T.random.randn([1, 2]), T.ones([])))

        # stop_grad, but `allow_unused` is False
        l_sum = T.reduce_sum(T.stop_grad(xt ** 2) * yt)
        with pytest.raises(Exception):
            _ = T.grad(
                [l_sum],
                [xt, yt],
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

    def test_assertions(self):
        # is_finite and assert_finite
        for x in [np.array([-1, 0, 1]), np.array([1., 2., 3.]),
                  np.array([np.inf, 0.]), np.array([np.nan, 0.]),
                  np.array([np.inf, np.nan])]:
            t = T.as_tensor_backend(x)
            assert_equal(T.is_finite(t), np.isfinite(x))
            is_finite = np.all(np.isfinite(x))

            if is_finite:
                assert_equal(T.assert_finite(t, 't'), x)
            else:
                with pytest.raises(Exception,
                                   match='Infinity or NaN value encountered'):
                    _ = T.assert_finite(t, 't')
