import operator
import re
import unittest
from functools import reduce

import numpy as np
import pytest
from mock import Mock

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tensorkit.distributions.utils import *
from tests.helper import *


class BaseSink(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in 'cdefgh':
                setattr(self, f'_{k}', v)
            elif k == 'before_map':
                setattr(self, 'after_map', v)
            elif k == 'after_map':
                raise KeyError(k)
            else:
                setattr(self, k, v)

    def __getattr__(self, item):
        return object.__getattribute__(self, f'_{item}')

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.__dict__ == other.__dict__

    def __repr__(self):
        attrs = ', '.join(f'{k}={repr(getattr(self, k))}'
                          for k in sorted(self.__dict__))
        return f'{self.__class__.__qualname__}({attrs})'


class Sink(BaseSink):
    pass


class Sink2(BaseSink):
    pass


class DistributionUtilsTestCase(TestCase):

    def test_get_overrided_parameterized(self):
        cls = Mock(__qualname__='xyz')

        self.assertTrue(get_overrided_parameterized(True, True))
        self.assertTrue(get_overrided_parameterized(True, None))
        self.assertFalse(get_overrided_parameterized(True, False))
        self.assertFalse(get_overrided_parameterized(False, None))
        self.assertFalse(get_overrided_parameterized(False, False))

        with pytest.raises(ValueError,
                           match='Distribution is not re-parameterizable, '
                                 'thus `reparameterized` cannot be set to True'):
            self.assertTrue(get_overrided_parameterized(False, True))

        with pytest.raises(ValueError,
                           match='Distribution `xyz` is not re-parameterizable, '
                                 'thus `reparameterized` cannot be set to True'):
            self.assertTrue(get_overrided_parameterized(False, True, cls))

    def test_get_prob_reduce_ndims(self):
        for sample_shape in ([2, 3, 4, 5, 6, 7], [2, 3], []):
            ndims = len(sample_shape)
            for min_event_ndims in range(0, ndims):
                for event_ndims in range(min_event_ndims, ndims):
                    for group_ndims in range(min_event_ndims - event_ndims,
                                             ndims - event_ndims):
                        self.assertEqual(
                            get_prob_reduce_ndims(
                                sample_ndims=ndims,
                                min_event_ndims=min_event_ndims,
                                event_ndims=event_ndims,
                                group_ndims=group_ndims,
                            ),
                            group_ndims + (event_ndims - min_event_ndims)
                        )
                    for group_ndims in [min_event_ndims - event_ndims - 1,
                                        ndims - event_ndims + 1]:
                        with pytest.raises(
                                Exception,
                                match=re.compile('`min_event_ndims - event_ndims <= group_ndims <= .*'
                                                 'sample_ndims - event_ndims` does not hold: ',
                                                 re.DOTALL)):
                            _ = get_prob_reduce_ndims(
                                sample_ndims=ndims,
                                min_event_ndims=min_event_ndims,
                                event_ndims=event_ndims,
                                group_ndims=group_ndims,
                            )

    def test_get_tail_size(self):
        for shape in ([], [2, 3, 4]):
            for ndims in range(len(shape)):
                self.assertEqual(
                    get_tail_size(shape, ndims),
                    reduce(operator.mul, shape[len(shape) - ndims:], 1)
                )
            with pytest.raises(Exception,
                               match=r'`ndims <= len\(shape\)` does not hold'):
                _ = get_tail_size([], len(shape) + 1)

    def test_log_pdf_mask(self):
        x = np.random.randn(3, 4, 5)

        for dtype in float_dtypes:
            x_t = T.as_tensor(x, dtype=dtype)
            ret = log_pdf_mask(x_t >= 0., x_t ** 2, T.random.LOG_ZERO_VALUE)
            expected = np.where(x >= 0., x ** 2, T.random.LOG_ZERO_VALUE)
            assert_allclose(ret, expected, rtol=1e-4)

    def test_check_tensor_arg_types(self):
        for dtype in float_dtypes:
            # check ordinary usage: mixed floats, numbers, mutual groups
            for specified_dtype in [None, dtype]:
                e_orig = T.as_tensor([1., 2., 3.], dtype=dtype)
                f_orig = StochasticTensor(
                    T.as_tensor([4., 5., 6.], dtype=dtype),
                    UnitNormal([]), None, 0, True
                )
                a, [b, c], [d, e], f = check_tensor_arg_types(
                    ('a', 1.0),
                    [('b', 2.0), ('c', None)],
                    [('d', None), ('e', e_orig)],
                    ('f', f_orig),
                    dtype=specified_dtype
                )
                for t, v in [(a, 1.0), (b, 2.0), (e, e_orig), (f, f_orig.tensor)]:
                    self.assertIsInstance(t, T.Tensor)
                    self.assertEqual(T.get_dtype(t), dtype)
                    self.assertEqual(T.get_device(t), T.current_device())
                    if isinstance(v, float):
                        assert_equal(t, v)
                    else:
                        self.assertIs(t, v)

            # float dtype determined by `dtype` and `default_dtype`
            for arg_name in ('dtype', 'default_dtype'):
                [a] = check_tensor_arg_types(('a', 123.0), **{arg_name: dtype})
                self.assertIsInstance(a, T.Tensor)
                self.assertEqual(T.get_dtype(a), dtype)
                assert_equal(a, 123.0)

            # tensor dtype will ignore `default_dtype`, but checked against `dtype`.
            a_orig = T.as_tensor([1., 2., 3.], dtype=dtype)
            [a] = check_tensor_arg_types(('a', a_orig), default_dtype=T.float32)
            self.assertIs(a, a_orig)

            if dtype != T.float32:
                with pytest.raises(ValueError,
                                   match=f'`a.dtype` != `dtype`: {dtype} vs '
                                         f'{T.float32}'):
                    _ = check_tensor_arg_types(('a', a), dtype=T.float32)

            # check multiple tensors type mismatch
            if dtype != T.float32:
                a_orig = T.as_tensor([1., 2., 3.], dtype=dtype)
                b_orig = T.as_tensor([4., 5., 6.], dtype=T.float32)

                with pytest.raises(ValueError,
                                   match=f'`b.dtype` != `a.dtype`: '
                                         f'{T.float32} vs {dtype}'):
                    _ = check_tensor_arg_types(('a', a_orig), ('b', b_orig))

            # check `device` and `default_device`
            if T.current_device() != T.CPU_DEVICE:
                [a] = check_tensor_arg_types(('a', [1., 2., 3.]), device=T.CPU_DEVICE)
                self.assertEqual(T.get_device(a), T.CPU_DEVICE)

                [a] = check_tensor_arg_types(('a', [1., 2., 3.]), default_device=T.CPU_DEVICE)
                self.assertEqual(T.get_device(a), T.CPU_DEVICE)

                [a] = check_tensor_arg_types(('a', [1., 2., 3.]), device=T.CPU_DEVICE,
                                             default_device=T.current_device())
                self.assertEqual(T.get_device(a), T.CPU_DEVICE)

                a = T.as_tensor([1., 2., 3.], device=T.current_device())
                with pytest.raises(ValueError,
                                   match=f'`a.device` != `device`'):
                    _ = check_tensor_arg_types(('a', a), device=T.CPU_DEVICE)

                b = T.as_tensor([1., 2., 3.], device=T.CPU_DEVICE)
                with pytest.raises(ValueError,
                                   match=f'`b.device` != `a.device`'):
                    _ = check_tensor_arg_types(('a', a), ('b', b))

            # check tensor cannot be None
            with pytest.raises(ValueError,
                               match='`a` must be specified.'):
                _ = check_tensor_arg_types(('a', None))

            # check mutual group must specify exactly one tensor
            for t in [None, T.as_tensor([1., 2., 3.], dtype=dtype)]:
                with pytest.raises(ValueError,
                                   match="Either `a` or `b` must be "
                                         "specified, but not both"):
                    _ = check_tensor_arg_types([('a', t), ('b', t)])
                with pytest.raises(ValueError,
                                   match="One and exactly one of `a`, `b` and "
                                         "`c` must be specified"):
                    _ = check_tensor_arg_types([('a', t), ('b', t), ('c', t)])

    def test_copy_distribution(self):
        # cached attribute copied
        for chk in [True, False]:
            x = Sink2(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8,
                      event_ndims=111, validate_tensors=chk, before_map=123)
            y = copy_distribution(
                cls=Sink,
                base=x,
                attrs=['a', 'b', 'validate_tensors', 'event_ndims',
                       ('before_map', 'after_map')],
                mutual_attrs=[['c', 'd'], ['e', 'f']],
                cached_attrs=['g', 'h'],
                compute_deps={'g': 'a'},
                original_mutual_params={'c': x.c, 'f': x.f},
                overrided_params={'b': 22},
            )
            self.assertEqual(
                y,
                Sink(a=1, b=22, c=3, d=4, e=5, f=6, g=7, h=8,
                     event_ndims=111, validate_tensors=chk, before_map=123)
            )

        for chk in [True, False]:
            x = Sink2(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8,
                      event_ndims=111, validate_tensors=chk, before_map=123)
            y = copy_distribution(
                cls=Sink,
                base=x,
                attrs=['a', 'b', 'validate_tensors', 'event_ndims',
                       ('before_map', 'after_map')],
                mutual_attrs=[['c', 'd'], ['e', 'f']],
                cached_attrs=['g', 'h'],
                compute_deps={'g': 'a'},
                original_mutual_params={'c': x.c, 'f': x.f},
                overrided_params={'b': 22, 'validate_tensors': False,
                                  'before_map': 456},
            )
            self.assertEqual(
                y,
                Sink(a=1, b=22, c=3, d=4, e=5, f=6, g=7, h=8,
                     event_ndims=111, validate_tensors=False, before_map=456)
            )

        # all computed attributes not copied due to validate_tensors
        x = Sink2(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8,
                  event_ndims=111, validate_tensors=False, before_map=123)
        y = copy_distribution(
            cls=Sink,
            base=x,
            attrs=['a', 'b', 'validate_tensors', 'event_ndims',
                   ('before_map', 'after_map')],
            mutual_attrs=[['c', 'd'], ['e', 'f']],
            cached_attrs=['g', 'h'],
            compute_deps={'g': 'a'},
            original_mutual_params={'c': x.c, 'f': x.f},
            overrided_params={'b': 22, 'validate_tensors': True},
        )
        self.assertEqual(
            y,
            Sink(a=1, b=22, c=3, f=6, event_ndims=111, validate_tensors=True,
                 before_map=123)
        )

        # cached attribute not copied due to dependency update
        x = Sink2(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8,
                  event_ndims=111, validate_tensors=True, before_map=123)
        y = copy_distribution(
            cls=Sink,
            base=x,
            attrs=['a', 'b', 'validate_tensors', 'event_ndims',
                   ('before_map', 'after_map')],
            mutual_attrs=[['c', 'd'], ['e', 'f']],
            cached_attrs=['g', 'h'],
            compute_deps={'g': 'a'},
            original_mutual_params={'c': x.c, 'f': x.f},
            overrided_params={'a': 11},
        )
        self.assertEqual(
            y,
            Sink(a=11, b=2, c=3, d=4, e=5, f=6, h=8, event_ndims=111,
                 validate_tensors=True, before_map=123)
        )

        # mutual attributes not copied due to override
        x = Sink2(a=1, b=2, c=3, d=4, e=5, f=6, g=7, h=8,
                  event_ndims=111, validate_tensors=True, before_map=123)
        y = copy_distribution(
            cls=Sink,
            base=x,
            attrs=['a', 'b', 'validate_tensors', 'event_ndims',
                   ('before_map', 'after_map')],
            mutual_attrs=[['c', 'd'], ['e', 'f']],
            cached_attrs=['g', 'h'],
            compute_deps={'g': 'a'},
            original_mutual_params={'c': x.c, 'f': x.f},
            overrided_params={'d': 44, 'f': 55},
        )
        self.assertEqual(
            y,
            Sink(a=1, b=2, d=44, f=55, g=7, h=8, event_ndims=111,
                 validate_tensors=True, before_map=123)
        )
