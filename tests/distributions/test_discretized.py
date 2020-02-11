import copy
import unittest
from itertools import product

import mock
import pytest

from tensorkit import tensor as T
from tensorkit.distributions import *
from tensorkit.distributions.utils import copy_distribution
from tests.helper import *


class DiscretizedLogisticTestCase(unittest.TestCase):

    def test_discretized_logsitic(self):
        T.random.seed(1234)

        mean = T.random.randn([3, 1, 4])
        log_scale = T.random.randn([2, 1])

        def do_check(**kwargs):
            d = DiscretizedLogistic(mean, log_scale, **kwargs)
            event_ndims = kwargs.get('event_ndims', 0)
            value_shape = T.broadcast_shape(T.shape(mean), T.shape(log_scale))
            log_prob_fn_kwargs = copy.copy(kwargs)
            log_prob_fn_kwargs.pop('discretize_sample', None)
            log_prob_fn_kwargs['discretize'] = \
                log_prob_fn_kwargs.pop('discretize_given', True)

            def log_prob_fn(t):
                return T.random.discretized_logistic_log_prob(
                    T.as_tensor(t), mean=mean, log_scale=log_scale,
                    **log_prob_fn_kwargs
                )

            check_distribution_instance(
                ctx=self,
                d=d,
                event_ndims=event_ndims,
                batch_shape=value_shape[: len(value_shape) - event_ndims],
                event_shape=value_shape[len(value_shape) - event_ndims:],
                min_event_ndims=0,
                max_event_ndims=len(value_shape),
                log_prob_fn=log_prob_fn,
                # other attributes,
                **kwargs
            )

        for biased_edges, discretize_given, discretize_sample in product(
                    [True, False],
                    [True, False],
                    [True, False],
                ):
            do_check(bin_size=1. / 255,
                     biased_edges=biased_edges,
                     discretize_given=discretize_given,
                     discretize_sample=discretize_sample)
            do_check(bin_size=1. / 255,
                     biased_edges=biased_edges,
                     discretize_given=discretize_given,
                     discretize_sample=discretize_sample,
                     min_val=-3.,
                     max_val=2.)

        with pytest.raises(ValueError,
                           match='`reparameterized` cannot be True when '
                                 '`discretize_sample` is True'):
            _ = DiscretizedLogistic(mean, log_scale, 1./32, reparameterized=True,
                                    discretize_sample=True)

        with pytest.raises(ValueError,
                           match='`min_val - max_val` must be multiples of '
                                 '`bin_size`'):
            _ = DiscretizedLogistic(mean, log_scale, .3, min_val=.1, max_val=.2)

        with pytest.raises(ValueError,
                           match='`min_val` and `max_val` must be both None or '
                                 'neither None'):
            _ = DiscretizedLogistic(mean, log_scale, 1./32, max_val=2.)

        with pytest.raises(ValueError,
                           match='`min_val` and `max_val` must be both None or '
                                 'neither None'):
            _ = DiscretizedLogistic(mean, log_scale, 1./32, min_val=-3.)

        with pytest.raises(ValueError,
                           match='The shape of `mean` and `log_scale` cannot be '
                                 'broadcasted against each other'):
            _ = DiscretizedLogistic(mean, T.zeros([7]), 1./32)

    def test_copy(self):
        T.random.seed(1234)

        mean = T.random.randn([3, 1, 4])
        log_scale = T.random.randn([2, 1])

        distrib = DiscretizedLogistic(
            mean=mean, log_scale=log_scale, bin_size=1./128,
            min_val=-3., max_val=2., event_ndims=1,
        )
        self.assertTrue(distrib.discretize_sample)
        self.assertFalse(distrib.reparameterized)
        self.assertFalse(distrib.validate_tensors)

        with mock.patch('tensorkit.distributions.discretized.copy_distribution',
                        wraps=copy_distribution) as f_copy:
            distrib2 = distrib.copy(event_ndims=2, discretize_sample=False,
                                    reparameterized=True, validate_tensors=True)
            self.assertIsInstance(distrib2, DiscretizedLogistic)
            self.assertEqual(distrib2.discretize_sample, False)
            self.assertEqual(distrib2.reparameterized, True)
            self.assertEqual(distrib2.event_ndims, 2)
            self.assertTrue(distrib2.validate_tensors)
            self.assertEqual(f_copy.call_args, ((), {
                'cls': DiscretizedLogistic,
                'base': distrib,
                'attrs': (
                    'mean', 'log_scale', 'bin_size', 'min_val', 'max_val',
                    'biased_edges', 'discretize_given', 'discretize_sample',
                    'reparameterized', 'event_ndims', 'epsilon', 'validate_tensors'
                ),
                'overrided_params': {'event_ndims': 2,
                                     'discretize_sample': False,
                                     'reparameterized': True,
                                     'validate_tensors': True},
            }))
