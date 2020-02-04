import re
import unittest

import numpy as np
import pytest
from mock import Mock

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tests.helper import *


class BaseDistributionTestCase(unittest.TestCase):

    def test_construct(self):
        def check_all_specified_by_constructor(cls):
            d = cls(
                dtype=T.int32, value_shape=[2, 3, 4], continuous=True,
                reparameterized=False, event_ndims=2, min_event_ndims=1,
                validate_tensors=True,
            )
            self.assertEqual(d.dtype, T.int32)
            self.assertEqual(d.continuous, True)
            self.assertEqual(d.reparameterized, False)
            self.assertEqual(d.value_shape, [2, 3, 4])
            self.assertEqual(d.value_ndims, 3)
            self.assertEqual(d.event_shape, [3, 4])
            self.assertEqual(d.event_ndims, 2)
            self.assertEqual(d.batch_shape, [2])
            self.assertEqual(d.batch_ndims, 1)
            self.assertEqual(d.min_event_ndims, 1)
            self.assertEqual(d.validate_tensors, True)
            self.assertIs(d.base_distribution, d)

        # all specified by constructor is okay
        check_all_specified_by_constructor(Distribution)

        # some specified by class attributes is okay
        class MyDistribution(Distribution):
            continuous = True
            reparameterized = True
            min_event_ndims = 1

        d = MyDistribution(dtype=T.int64, value_shape=[2, 3, 4])
        self.assertEqual(d.dtype, T.int64)
        self.assertEqual(d.value_shape, [2, 3, 4])
        self.assertEqual(d.value_ndims, 3)
        self.assertEqual(d.event_shape, [4])
        self.assertEqual(d.event_ndims, 1)
        self.assertEqual(d.batch_shape, [2, 3])
        self.assertEqual(d.batch_ndims, 2)
        self.assertEqual(d.continuous, True)
        self.assertEqual(d.reparameterized, True)
        self.assertEqual(d.min_event_ndims, 1)
        self.assertIs(d.base_distribution, d)

        # class attributes specified but overrided by class constructor
        check_all_specified_by_constructor(MyDistribution)

        # validate_tensors follow `settings.validate_tensors` by default
        old_validate_tensors = settings.validate_tensors
        try:
            settings.validate_tensors = False
            self.assertEqual(MyDistribution(T.int64, [2]).validate_tensors, False)
            settings.validate_tensors = True
            self.assertEqual(MyDistribution(T.int64, [2]).validate_tensors, True)
        finally:
            settings.validate_tensors = old_validate_tensors

        # check `value_shape` against `batch_shape` + event_ndims
        d = MyDistribution(dtype=T.int64, batch_shape=[2, 3], event_ndims=3)
        self.assertEqual(d.batch_shape, [2, 3])
        self.assertEqual(d.batch_ndims, 2)
        self.assertEqual(d.event_ndims, 3)
        self.assertEqual(d.value_ndims, 5)
        self.assertIsNone(d.event_shape)
        self.assertIsNone(d.value_shape)

        with pytest.raises(ValueError,
                           match='Either `value_shape` or `batch_shape` '
                                 'should be specified'):
            _ = MyDistribution(dtype=T.int64)

        with pytest.raises(ValueError,
                           match='The arguments `value_shape`, `batch_shape` '
                                 'and `event_ndims` are not coherent'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2, 3, 4],
                               batch_shape=[2])

        with pytest.raises(ValueError,
                           match='The arguments `value_shape`, `batch_shape` '
                                 'and `event_ndims` are not coherent'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2, 3, 4],
                               batch_shape=[2, 4], event_ndims=2)

        # overrided arguments conflict with class attributes
        class MyDistribution(Distribution):
            continuous = False
            reparameterized = False
            min_event_ndims = 1

        with pytest.raises(ValueError,
                           match='`continuous` has already been defined by '
                                 'class attribute, thus cannot be overrided'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2], continuous=True)

        with pytest.raises(ValueError,
                           match='`min_event_ndims` has already been defined by '
                                 'class attribute, thus cannot be overrided'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2], min_event_ndims=0)

        with pytest.raises(ValueError,
                           match=f'Distribution `{MyDistribution.__qualname__}'
                                 f'` is not re-parameterizable, thus '
                                 f'`reparameterized` cannot be set to True'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2],
                               reparameterized=True)

        with pytest.raises(ValueError,
                           match='`event_ndims >= min_event_ndims` does not '
                                 'hold: `event_ndims` == 0, while '
                                 '`min_event_ndims` == 1'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2], event_ndims=0)

        with pytest.raises(ValueError,
                           match='`event_ndims >= min_event_ndims` does not '
                                 'hold: `event_ndims` == 0, while '
                                 '`min_event_ndims` == 1'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2], event_ndims=0)

        with pytest.raises(ValueError,
                           match=r'`event_ndims <= len\(value_shape\)` does '
                                 r'not hold: `event_ndims` == 2, while '
                                 r'`len\(value_shape\)` == 1'):
            _ = MyDistribution(dtype=T.int64, value_shape=[2], event_ndims=2)

    def test_assert_finite(self):
        d = Distribution(T.int32, [], continuous=True, reparameterized=True,
                         event_ndims=0, min_event_ndims=0)

        d.validate_tensors = False
        t = T.as_tensor(np.nan)
        self.assertIs(d._assert_finite(t, 't'), t)

        d.validate_tensors = True
        with pytest.raises(Exception,
                           match='Infinity or NaN value encountered'):
            _ = d._assert_finite(t, 't')

    def test_sample(self):
        t0 = T.zeros([], T.float32)
        d = Distribution(
            T.float32, [2, 3], continuous=True, reparameterized=True,
            event_ndims=1, min_event_ndims=0)
        d._sample = Mock(return_value=t0)

        # without group_ndims or reparameterized
        t = d.sample()
        self.assertIs(t, t0)
        self.assertEqual(d._sample.call_args, ((None, 0, 1, True), {}))

        t = d.sample(n_samples=123)
        self.assertEqual(d._sample.call_args, ((123, 0, 1, True), {}))

        # with group_ndims
        t = d.sample(group_ndims=1)
        self.assertEqual(d._sample.call_args, ((None, 1, 2, True), {}))

        t = d.sample(n_samples=123, group_ndims=1)
        self.assertEqual(d._sample.call_args, ((123, 1, 2, True), {}))

        t = d.sample(n_samples=123, group_ndims=2)
        self.assertEqual(d._sample.call_args, ((123, 2, 3, True), {}))

        with pytest.raises(Exception,
                           match=re.compile('`min_event_ndims - event_ndims <= group_ndims <= .*'
                                            'sample_ndims - event_ndims` does not hold: ',
                                            re.DOTALL)):
            _ = d.sample(group_ndims=2)

        with pytest.raises(Exception,
                           match=re.compile('`min_event_ndims - event_ndims <= group_ndims <= .*'
                                            'sample_ndims - event_ndims` does not hold: ',
                                            re.DOTALL)):
            _ = d.sample(n_samples=123, group_ndims=3)

        d._sample.reset_mock()

        # with reparameterized
        t = d.sample(reparameterized=False)
        self.assertEqual(d._sample.call_args, ((None, 0, 1, False), {}))
        d._sample.reset_mock()

        d.reparameterized = False
        t = d.sample(reparameterized=False)
        self.assertEqual(d._sample.call_args, ((None, 0, 1, False), {}))
        d._sample.reset_mock()

        with pytest.raises(ValueError,
                           match='Distribution .* is not re-parameterizable, '
                                 'thus `reparameterized` cannot be set to True'):
            _ = d.sample(reparameterized=True)

    def test_log_prob(self):
        t0 = T.zeros([], T.float32)
        d = Distribution(
            T.float32, [2, 3], continuous=True, reparameterized=True,
            event_ndims=1, min_event_ndims=0)
        d._log_prob = Mock(return_value=t0)

        def do_check(given, group_ndims, args):
            if group_ndims is None:
                ret = d.log_prob(given)
            else:
                ret = d.log_prob(given, group_ndims)
            self.assertIs(ret, t0)
            self.assertEqual(d._log_prob.call_args, ((given,) + args, {}))
            d._log_prob.reset_mock()

        for sample_shape in ([], [1], [1, 1]):
            do_check(T.zeros(sample_shape), None, (0, 1))
            for group_ndims in (-1, 0, 1):
                do_check(T.zeros(sample_shape),
                         group_ndims,
                         (group_ndims, 1 + group_ndims))

            for group_ndims in (-2, 2):
                with pytest.raises(Exception,
                                   match=re.compile('`min_event_ndims - event_ndims <= group_ndims <= .*'
                                                    'sample_ndims - event_ndims` does not hold: ',
                                                    re.DOTALL)):
                    _ = d.log_prob(T.zeros(sample_shape), group_ndims)
            d._log_prob.reset_mock()

        for group_ndims in (-1, 0, 1, 2):
            do_check(T.zeros([5, 2, 3]),
                     group_ndims,
                     (group_ndims, 1 + group_ndims))

    def test_prob(self):
        np.random.seed(1234)
        t00 = np.random.randn(2, 3)
        t0 = T.as_tensor(t00)
        d = Distribution(
            T.float32, [2, 3], continuous=True, reparameterized=True,
            event_ndims=1, min_event_ndims=0)
        d.log_prob = Mock(return_value=t0)

        given = T.random.randn([5, 2, 3])
        ret = d.prob(given, group_ndims=1)
        self.assertEqual(d.log_prob.call_args, ((given, 1), {}))
        assert_allclose(ret, np.exp(t00))
