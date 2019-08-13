import unittest

import numpy as np
import pytest

from tensorkit import tensor as T

assert_allclose = np.testing.assert_allclose


class _SimpleTensor(T.ExtendedTensor):

    _flag_ = None

    def init_extension(self, flag=None):
        self._flag_ = flag

    def get_flag(self):
        return self._flag_


T.register_extended_tensor_class(_SimpleTensor)


class TensorWrapperTestCase(unittest.TestCase):

    def test_unary_op(self):
        def check_op(name, func, x):
            if x.dtype == np.bool:
                as_tensor = T.as_tensor
            else:
                as_tensor = T.as_tensor

            x_tensor = as_tensor(x)
            ans = func(x_tensor)
            res = T.as_tensor(func(T.extend_tensor(x_tensor, _SimpleTensor)))
            self.assertEqual(
                res.dtype, ans.dtype,
                msg=f'Result dtype does not match answer after unary operator '
                    f'{name} is applied: {res.dtype!r} vs {ans.dtype!r} '
                    f'(x is {x!r})'
            )
            res_val = T.to_numpy(res)
            ans_val = T.to_numpy(ans)
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg=f'Result value does not match answer after unary '
                        f'operator {name} is applied: {res_val!r} vs '
                        f'{ans_val!r} (x is {x!r})'
            )

        int_data = np.asarray([1, -2, 3], dtype=np.int32)
        float_data = np.asarray([1.1, -2.2, 3.3], dtype=np.float32)
        bool_data = np.asarray([True, False, True], dtype=np.bool)

        check_op('abs', abs, int_data)
        check_op('abs', abs, float_data)
        check_op('neg', (lambda v: -v), int_data)
        check_op('neg', (lambda v: -v), float_data)
        check_op('invert', (lambda v: ~v), bool_data)

    def test_binary_op(self):
        def check_op(name, func, x, y):
            if x.dtype == np.bool:
                as_tensor = T.as_tensor
            else:
                as_tensor = T.as_tensor

            x_tensor = as_tensor(x)
            y_tensor = as_tensor(y)
            ans = func(x_tensor, y_tensor)
            res_1 = T.as_tensor(
                func(T.extend_tensor(x_tensor, _SimpleTensor), y_tensor))
            res_2 = T.as_tensor(
                func(
                    T.extend_tensor(x_tensor, _SimpleTensor),
                    T.extend_tensor(y_tensor, _SimpleTensor)
                )
            )
            res_3 = T.as_tensor(
                func(x_tensor, T.extend_tensor(y_tensor, _SimpleTensor)))

            outputs = [('TensorWrapper + Tensor', res_1),
                       ('TensorWrapper + TensorWrapper', res_2),
                       ('Tensor + TensorWrapper', res_3)]

            # not all backends support np.ndarray
            if T.backend.name != 'pytorch':
                res_4 = T.as_tensor(
                    func(
                        T.extend_tensor(x_tensor, _SimpleTensor),
                        # y -> Tensor -> np.ndarray,
                        # in case T.boolean != np.bool
                        T.to_numpy(as_tensor(y))
                    )
                )
                res_5 = T.as_tensor(
                    func(
                        T.to_numpy(as_tensor(x)),
                        T.extend_tensor(y_tensor, _SimpleTensor)
                    )
                )
                outputs.extend([
                    ('TensorWrapper + np.ndarray', res_4),
                    ('np.ndarray + TensorWrapper', res_5),
                ])

            for tag, res in outputs:
                self.assertEqual(
                    res.dtype, ans.dtype,
                    msg=f'Result dtype does not match answer after {tag} '
                        f'binary operator {name} is applied: {res.dtype!r} vs '
                        f'{ans.dtype!r} (x is {x!r}, y is {y!r})'
                )
                res_val = T.to_numpy(res)
                ans_val = T.to_numpy(ans)
                np.testing.assert_equal(
                    res_val, ans_val,
                    err_msg=f'Result value does not match answer after {tag} '
                            f'binary operator {name} is applied: {res_val!r} '
                            f'vs {ans_val!r} (x is {x!r}, y is {y!r}).'
                )

        def run_ops(x, y, ops):
            for name, func in ops.items():
                check_op(name, func, x, y)

        arith_ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
        }
        arith_ops2 = {
            'floordiv': lambda x, y: x // y,
            'mod': lambda x, y: x % y,
        }
        arith_ops3 = {
            'div': lambda x, y: x / y,
        }

        logical_ops = {
            'and': lambda x, y: x & y,
            'or': lambda x, y: x | y,
            'xor': lambda x, y: x ^ y,
        }

        relation_ops = {
            'lt': lambda x, y: x < y,
            'le': lambda x, y: x <= y,
            'gt': lambda x, y: x > y,
            'ge': lambda x, y: x >= y,
        }

        # arithmetic operators
        run_ops(np.asarray([-4, 5, 6], dtype=np.int32),
                np.asarray([1, -2, 3], dtype=np.int32),
                arith_ops)
        run_ops(np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                np.asarray([1.1, -2.2, 3.4], dtype=np.float32),
                arith_ops)
        run_ops(np.asarray([4, 5, 6], dtype=np.int32),
                np.asarray([1, 2, 3], dtype=np.int32),
                arith_ops2)
        run_ops(np.asarray([4.4, 5.5, 6.6], dtype=np.float32),
                np.asarray([1.1, 2.2, 3.4], dtype=np.float32),
                arith_ops2)
        run_ops(np.asarray([4.4, 5.5, 6.6], dtype=np.float32),
                np.asarray([1.1, 2.2, 3.4], dtype=np.float32),
                arith_ops3)

        check_op('pow',
                 (lambda x, y: x ** y),
                 np.asarray([-4, 5, 6], dtype=np.int32),
                 np.asarray([1, 2, 3], dtype=np.int32))
        check_op('pow',
                 (lambda x, y: x ** y),
                 np.asarray([-4.4, 5.5, 6.6], dtype=np.float32),
                 np.asarray([1.1, -2.2, 3.3], dtype=np.float32))

        # logical operators
        run_ops(np.asarray([True, False, True, False], dtype=np.bool),
                np.asarray([True, True, False, False], dtype=np.bool),
                logical_ops)

        # relation operators
        run_ops(np.asarray([1, -2, 3, -4, 5, 6, -4, 5, 6], dtype=np.int32),
                np.asarray([1, -2, 3, 1, -2, 3, -4, 5, 6], dtype=np.int32),
                relation_ops)
        run_ops(
            np.asarray([1.1, -2.2, 3.3, -4.4, 5.5, 6.6, -4.4, 5.5, 6.6],
                       dtype=np.float32),
            np.asarray([1.1, -2.2, 3.3, 1.1, -2.2, 3.3, -4.4, 5.5, 6.6],
                       dtype=np.float32),
            relation_ops
        )

    def test_getitem(self):
        def check_getitem(x, y, xx, yy):
            ans = T.as_tensor(x[y])
            print(xx, yy)
            res = xx[yy]

            self.assertEqual(
                res.dtype, ans.dtype,
                msg=f'Result dtype does not match answer after getitem '
                    f'is applied: {res.dtype!r} vs {ans.dtype!r} (x is {x!r}, '
                    f'y is {y!r}, xx is {xx!r}, yy is {yy!r}).'
            )
            res_val = T.to_numpy(res)
            ans_val = T.to_numpy(ans)
            np.testing.assert_equal(
                res_val, ans_val,
                err_msg=f'Result value does not match answer after '
                        f'getitem is applied: {res_val!r} vs {ans_val!r} '
                        f'(x is {x!r}, y is {y!r}, xx is {xx!r}, yy is {yy!r}).'
            )

        class _SliceGenerator(object):
            def __getitem__(self, item):
                return item
        sg = _SliceGenerator()

        data = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
        indices_or_slices = [
            0,
            -1,
            np.asarray([0, 3, 2, 6], dtype=int),
            np.asarray([-1, -2, -3], dtype=int),
            sg[0:],
            sg[:1],
            sg[:: 2],
        ]

        # pytorch currently does not support negative strides
        if T.backend.name != 'pytorch':
            indices_or_slices.extend([
                sg[-1:],
                sg[: -1],
                sg[:: -1],
            ])

        for s in indices_or_slices:
            x_tensor = T.as_tensor(data)
            x_simple_tensor = T.extend_tensor(x_tensor, _SimpleTensor)
            check_getitem(data, s, x_simple_tensor, s)

            if not isinstance(s, slice):
                y_tensor = T.as_tensor(s)
                y_simple_tensor = T.extend_tensor(y_tensor, _SimpleTensor)
                check_getitem(data, s, x_simple_tensor, y_tensor)
                check_getitem(data, s, x_simple_tensor, y_simple_tensor)

                # not all backends support inverse indexing
                if T.backend.name != 'pytorch':
                    check_getitem(data, s, x_tensor, y_simple_tensor)

    def test_bool(self):
        self.assertTrue(bool(T.extend_tensor(T.as_tensor(True), _SimpleTensor)))
        self.assertFalse(not T.extend_tensor(T.as_tensor(True), _SimpleTensor))
        self.assertFalse(
            bool(T.extend_tensor(T.as_tensor(False), _SimpleTensor)))
        self.assertTrue(not T.extend_tensor(T.as_tensor(False), _SimpleTensor))

        flag = []
        if T.extend_tensor(T.as_tensor(True), _SimpleTensor):
            flag.append(1)
        if T.extend_tensor(T.as_tensor(False), _SimpleTensor):
            flag.append(2)
        self.assertListEqual(flag, [1])

    def test_iter(self):
        t = T.extend_tensor(T.arange(10), _SimpleTensor)
        self.assertEqual(len(t), 10)

        arr = list(a for a in t)
        for i, a in enumerate(t):
            self.assertIsInstance(a, T.Tensor)
            self.assertEqual(T.to_numpy(a), i)

    def test_as_tensor(self):
        t = T.extend_tensor(T.as_tensor(123., dtype=T.float32), _SimpleTensor)

        t2 = T.as_tensor(t)
        self.assertIsInstance(t2, T.Tensor)
        self.assertEqual(T.dtype(t2), T.float32)
        self.assertNotIsInstance(t2, _SimpleTensor)
        self.assertEqual(T.to_numpy(t2), 123)

        t2 = T.as_tensor(t, dtype=T.int32)
        self.assertEqual(T.dtype(t2), T.int32)
        self.assertEqual(T.to_numpy(t2), 123)

    def test_get_attributes(self):
        t = T.extend_tensor(T.as_tensor([1., 2., 3.]), _SimpleTensor, flag=123)
        self.assertEqual(t._flag_, 123)
        self.assertEqual(t.get_flag(), 123)
        members = dir(t)
        for member in ['_flag_', 'get_flag']:
            self.assertIn(
                member, members,
                msg=f'{members!r} should in dir(t), but not'
            )
            self.assertTrue(
                hasattr(t, member),
                msg=f'_SimpleTensor should has member {member!r}, but not.'
            )
        t0 = t.as_tensor()
        for member in dir(t0):
            if not member.startswith('_'):
                self.assertIn(
                    member, members,
                    msg=f'{members!r} should in dir(t), but not'
                )
                self.assertTrue(
                    hasattr(t, member),
                    msg=f'_SimpleTensor should has member {member!r}, but not.'
                )
                try:
                    self.assertEqual(getattr(t, member),
                                     getattr(t0, member))
                except Exception:
                    pass  # some object may not be comparable

    def test_set_attributes(self):
        t = T.extend_tensor(T.as_tensor([1., 2., 3.]), _SimpleTensor)
        t0 = t.as_tensor()

        self.assertTrue(hasattr(t, '_flag_'))
        t._flag_ = 123
        self.assertEqual(t._flag_, 123)

        self.assertTrue(hasattr(t, 'get_flag'))
        t.get_flag = 456
        self.assertEqual(t.get_flag, 456)
        self.assertTrue(hasattr(t, 'get_flag'))

        t.abc = 1001
        self.assertEqual(t.abc, 1001)
        self.assertTrue(hasattr(t, 'abc'))

    def test_del_attributes(self):
        t = T.extend_tensor(T.as_tensor([1., 2., 3.]), _SimpleTensor)
        t.abc = 1001
        del t.abc
        self.assertFalse(hasattr(t, 'abc'))

    def test_register_non_extended_tensor_class(self):
        class _NonExtendedTensorClass(object):
            pass

        with pytest.raises(TypeError, match='`cls` is not a class, or not a '
                                            'subclass of `ExtendedTensor`: '
                                            'got .*_NonExtendedTensorClass.*'):
            T.register_extended_tensor_class(_NonExtendedTensorClass)

        with pytest.raises(TypeError, match='`cls` is not a class, or not a '
                                            'subclass of `ExtendedTensor`: '
                                            'got 123'):
            T.register_extended_tensor_class(123)
