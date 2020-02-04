import unittest

import pytest
import mock

from tensorkit import tensor as T
from tensorkit.distributions import (Categorical, Normal, UnitNormal, Mixture,
                                     OneHotCategorical, FlowDistribution)
from tensorkit.distributions.utils import copy_distribution
from tensorkit.flows import ActNorm
from tests.distributions.test_flow import check_distribution_instance


def check_mixture(ctx,
                  categorical,
                  components):
    min_event_ndims = components[0].event_ndims
    max_event_ndims = components[0].value_ndims

    def fn(cat, components, reparameterized, event_ndims,
           validate_tensors):
        # construct the instance
        kwargs = {}

        if reparameterized is not None:
            kwargs['reparameterized'] = reparameterized
        else:
            reparameterized = False

        if event_ndims is not None:
            kwargs['event_ndims'] = event_ndims
        else:
            event_ndims = components[0].event_ndims

        if validate_tensors is not None:
            kwargs['validate_tensors'] = validate_tensors
        else:
            validate_tensors = cat.validate_tensors or \
                               any(c.validate_tensors for c in components)

        value_shape = None
        for c in components:
            if c.value_shape is not None:
                value_shape = c.value_shape
                break
        batch_shape = components[0].batch_shape[: (len(components[0].batch_shape) -
                                                   (event_ndims - min_event_ndims))]
        event_shape = (value_shape[len(batch_shape):]
                       if value_shape is not None else None)

        mix = Mixture(cat, components, **kwargs)
        ctx.assertIsInstance(mix.categorical, (OneHotCategorical, Categorical))

        # check the instance
        def log_prob_fn(t):
            cat_log_prob = T.nn.log_softmax(cat.logits)
            c_log_probs = [c.log_prob(t.tensor) for c in components]
            c_log_prob = T.stack(c_log_probs, axis=-1)

            log_prob = T.log_sum_exp(cat_log_prob + c_log_prob, axis=[-1])
            return log_prob

        check_distribution_instance(
            ctx=ctx,
            d=mix,
            event_ndims=event_ndims,
            batch_shape=batch_shape,
            event_shape=event_shape,
            min_event_ndims=min_event_ndims,
            max_event_ndims=max_event_ndims,
            log_prob_fn=log_prob_fn,
            # other attributes
            components=components,
            n_components=len(components),
            dtype=components[0].dtype,
            continuous=components[0].continuous,
            reparameterized=reparameterized,
            validate_tensors=validate_tensors,
        )

    for cat in (categorical.to_indexed(), categorical.to_one_hot()):
        fn(cat, components, None, None, None)

    for reparameterized in (True, False):
        if not reparameterized or \
                not any(not c.reparameterized for c in components):
            fn(categorical, components, reparameterized, None, None)

    for event_ndims in (None,
                        min_event_ndims,
                        (min_event_ndims + max_event_ndims) // 2,
                        max_event_ndims):
        fn(categorical, components, None, event_ndims, None)

    for validate_tensors in (None, True, False):
        fn(categorical, components, None, None, validate_tensors)


class MixtureTestCase(unittest.TestCase):

    def test_mixture(self):
        T.random.seed(1234)

        check_mixture(
            self,
            Categorical(logits=T.random.randn([4, 5, 1])),
            components=[
                UnitNormal([4, 5], event_ndims=0),
            ]
        )

        check_mixture(
            self,
            Categorical(logits=T.random.randn([4, 5, 2])),
            components=[
                UnitNormal([4, 5], event_ndims=0),
                Normal(mean=T.random.randn([4, 5]),
                       logstd=T.random.randn([4, 5])),
            ]
        )

        check_mixture(
            self,
            Categorical(logits=T.random.randn([4, 2])),
            components=[
                FlowDistribution(UnitNormal([4, 5]), ActNorm(5)),
                FlowDistribution(UnitNormal([4, 5]), ActNorm(5)),
            ]
        )

        check_mixture(
            self,
            Categorical(logits=T.random.randn([4, 2])),
            components=[
                FlowDistribution(UnitNormal([4, 5]), ActNorm(5)),
                UnitNormal([4, 5], event_ndims=1),
            ]
        )

        check_mixture(
            self,
            OneHotCategorical(logits=T.random.randn([4, 3])),
            components=[
                UnitNormal([4, 5], event_ndims=1, dtype=T.float64),
                Normal(mean=T.random.randn([4, 5], dtype=T.float64),
                       logstd=T.random.randn([4, 5], dtype=T.float64),
                       event_ndims=1),
                Normal(mean=T.random.randn([4, 5], dtype=T.float64),
                       logstd=T.random.randn([4, 5], dtype=T.float64),
                       event_ndims=1,
                       reparameterized=False,
                       validate_tensors=True),
            ]
        )

        with pytest.raises(TypeError,
                           match='`categorical` is not a categorical distribution'):
            _ = Mixture(UnitNormal([3, 1]), [UnitNormal([3, 4])])

        with pytest.raises(ValueError,
                           match='`categorical.event_ndims` does not equal to '
                                 '`categorical.min_event_ndims`: got '
                                 'Categorical instance, with '
                                 '`event_ndims` 1'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2, 3, 1]), event_ndims=1),
                [UnitNormal([2, 3])]
            )

        with pytest.raises(ValueError,
                           match='`categorical.event_ndims` does not equal to '
                                 '`categorical.min_event_ndims`: got '
                                 'OneHotCategorical instance, with '
                                 '`event_ndims` 2'):
            _ = Mixture(
                OneHotCategorical(logits=T.random.randn([2, 3, 1]), event_ndims=2),
                [UnitNormal([2, 3])]
            )

        with pytest.raises(ValueError, match='`components` must not be empty'):
            _ = Mixture(Categorical(logits=T.random.randn([2])), [])

        with pytest.raises(ValueError,
                           match=r'`len\(components\)` != `categorical.n_classes`'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([])]
            )

        with pytest.raises(TypeError,
                           match=r'`components\[1\]` is not an instance of '
                                 r'`Distribution`: got 123'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([]), 123]
            )

        with pytest.raises(ValueError,
                           match=r'`reparameterized` is True, but `components'
                                 r'\[1\]` is not re-parameterizable'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([]), UnitNormal([], reparameterized=False)],
                reparameterized=True
            )

        with pytest.raises(ValueError,
                           match=r'`components\[1\].dtype` != '
                                 r'`components\[0\].dtype`'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([], dtype=T.float32), UnitNormal([], dtype=T.float64)]
            )

        with pytest.raises(ValueError,
                           match=r'`components\[1\].event_ndims` != '
                                 r'`components\[0\].event_ndims`: '
                                 r'1 vs 0'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([]), UnitNormal([2], event_ndims=1)]
            )

        with pytest.raises(ValueError,
                           match=r'`components\[1\].batch_shape` != '
                                 r'`components\[0\].batch_shape`: '
                                 r'\[2\] vs \[\]'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([]), UnitNormal([2])]
            )

        with pytest.raises(ValueError,
                           match=r'`categorical.batch_shape` != the '
                                 r'`batch_shape` of `components`: '
                                 r'\[\] vs \[3\]'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([3]), UnitNormal([3])]
            )

        with pytest.raises(ValueError,
                           match=r'`components\[1\].event_shape` does not '
                                 r'agree with others: '
                                 r'\[4\] vs \[3\]'):
            _ = Mixture(
                Categorical(logits=T.random.randn([2])),
                [UnitNormal([3], event_ndims=1),
                 UnitNormal([4], event_ndims=1)]
            )

        with pytest.raises(ValueError,
                           match=r'`components\[2\].event_shape` does not '
                                 r'agree with others: '
                                 r'\[4\] vs \[3\]'):
            _ = Mixture(
                Categorical(logits=T.random.randn([4, 3])),
                [
                    FlowDistribution(UnitNormal([4, 3]), ActNorm(3)),
                    UnitNormal([4, 3], event_ndims=1),
                    UnitNormal([4, 4], event_ndims=1),
                ]
            )

        with pytest.raises(ValueError,
                           match=r'`event_ndims` out of range: got 0, but '
                                 r'the minimum allowed value is 1, and '
                                 r'the maximum allowed value is 2'):
            _ = Mixture(
                Categorical(logits=T.random.randn([5, 2])),
                [UnitNormal([5, 3], event_ndims=1),
                 UnitNormal([5, 3], event_ndims=1)],
                event_ndims=0,
            )

        with pytest.raises(ValueError,
                           match=r'`event_ndims` out of range: got 3, but '
                                 r'the minimum allowed value is 1, and '
                                 r'the maximum allowed value is 2'):
            _ = Mixture(
                Categorical(logits=T.random.randn([5, 2])),
                [UnitNormal([5, 3], event_ndims=1),
                 UnitNormal([5, 3], event_ndims=1)],
                event_ndims=3,
            )

    def test_copy(self):
        categorical = Categorical(logits=T.random.randn([4, 5, 2]))
        components = [
            UnitNormal([4, 5], event_ndims=0),
            Normal(mean=T.random.randn([4, 5]),
                   logstd=T.random.randn([4, 5])),
        ]
        distrib = Mixture(categorical, components, event_ndims=1,
                          reparameterized=True, validate_tensors=True)
        self.assertEqual(distrib.event_ndims, 1)
        self.assertTrue(distrib.reparameterized)
        self.assertTrue(distrib.validate_tensors)

        with mock.patch('tensorkit.distributions.mixture.copy_distribution',
                        wraps=copy_distribution) as f_copy:
            distrib2 = distrib.copy(event_ndims=2, reparameterized=False,
                                    validate_tensors=False)
            self.assertIsInstance(distrib2, Mixture)
            self.assertIs(distrib2.categorical, categorical)
            self.assertEqual(distrib2.components, components)
            self.assertEqual(distrib2.reparameterized, False)
            self.assertEqual(distrib2.event_ndims, 2)
            self.assertFalse(distrib2.validate_tensors)
            self.assertEqual(f_copy.call_args, ((), {
                'cls': Mixture,
                'base': distrib,
                'attrs': ('categorical', 'components', 'reparameterized',
                          'event_ndims', 'validate_tensors'),
                'overrided_params': {'event_ndims': 2,
                                     'reparameterized': False,
                                     'validate_tensors': False},
            }))
