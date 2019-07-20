import unittest

import numpy as np
import pytest

from tensorkit import settings
from tensorkit import tensor as T

assert_allclose = np.testing.assert_allclose


class TensorCoreTestCase(unittest.TestCase):

    def test_typing(self):
        t = T.as_tensor(np.random.randn(2, 3))
        self.assertIsInstance(t, T.Tensor)

        s = T.as_shape([1, 2, 3])
        self.assertIsInstance(s, T.Shape)

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
        np.testing.assert_equal(T.read(t), x)

        t2 = T.cast(t, T.float32)
        self.assertIsInstance(t2, T.Tensor)
        self.assertEqual(T.dtype(t2), T.float32)
        np.testing.assert_equal(T.read(t2), x)

    def test_math_univariate_op(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3)

        assert_allclose(T.read(T.abs(x)), np.abs(x))
        assert_allclose(T.read(T.neg(x)), -x)
        assert_allclose(T.read(T.exp(x)), np.exp(x))
        assert_allclose(T.read(T.log(np.abs(x))), np.log(np.abs(x)))
        assert_allclose(T.read(T.log1p(np.abs(x) - 1. + 1e-7)),
                        np.log1p(np.abs(x) - 1. + 1e-7))
        assert_allclose(T.read(T.sin(x)), np.sin(x))
        assert_allclose(T.read(T.cos(x)), np.cos(x))
        assert_allclose(T.read(T.square(x)), x ** 2)

    def test_math_bivariate_op(self):
        np.random.seed(1234)
        x = np.random.randn(2, 3)
        y = np.random.randn(3)

        assert_allclose(T.read(T.add(x, y)), x + y)
        assert_allclose(T.read(T.sub(x, y)), x - y)
        assert_allclose(T.read(T.mul(x, y)), x * y)
        assert_allclose(T.read(T.pow(np.abs(x), y)), np.abs(x) ** y)

        # for division, of course y should not equal to zero
        y = np.asarray(y == 0, dtype=y.dtype) + y
        assert_allclose(T.read(T.div(x, y)), x / y)
        assert_allclose(T.read(T.truediv(x, y)), x / y)

        # for floordiv and mod, we only require the backend tensor engine
        # to produce identical results with numpy when x > 0 and y > 0
        x = np.abs(x)
        y = np.abs(y)
        assert_allclose(T.read(T.floordiv(x, y)), x // y)
        assert_allclose(T.read(T.mod(x, y)), x % y)
        assert_allclose(T.read(T.fmod(x, y)), x % y)

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
        assert_allclose(T.read(out),
                        x.astype(np.float32) / y.astype(np.float32))

        # input int16, output float32
        x = np.random.randint(-32768, 32767, size=(2, 3), dtype=np.int16)
        y = np.random.randint(-32768, 32767, size=(3,), dtype=np.int16)
        y = y + (y == 0).astype(y.dtype)
        out = T.truediv(x, y)
        self.assertEqual(T.dtype(out), T.float32)
        assert_allclose(T.read(out),
                        x.astype(np.float32) / y.astype(np.float32))

        # input int32, output float64
        x = np.random.randint(-100000, 100000, size=(2, 3), dtype=np.int32)
        y = np.random.randint(-100000, 100000, size=(3,), dtype=np.int32)
        y = y + (y == 0).astype(y.dtype)
        out = T.truediv(x, y)
        self.assertEqual(T.dtype(out), T.float64)
        assert_allclose(T.read(out),
                        x.astype(np.float64) / y.astype(np.float64))
