import math
import unittest
from typing import *

import mock
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.distributions import Categorical, FlowDistribution, UnitNormal
from tensorkit.distributions.utils import copy_distribution
from tensorkit.flows import ReshapeFlow, ActNorm
from tensorkit.tensor import Tensor, as_tensor_backend, int_range
from tests.helper import *


class _MyFlow(tk.flows.Flow):

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = input * 2.0 + 1
            event_ndims = self.x_event_ndims
        else:
            output = (input - 1.0) * 0.5
            event_ndims = self.y_event_ndims

        if compute_log_det:
            if inverse:
                output_log_det = as_tensor_backend(-math.log(2.))
            else:
                output_log_det = as_tensor_backend(math.log(2.))

            for axis in int_range(-event_ndims, 0):
                output_log_det = output_log_det * output.shape[axis]

            if input_log_det is not None:
                output_log_det = output_log_det + input_log_det
        else:
            output_log_det: Optional[Tensor] = None

        return output, output_log_det


def check_flow_distribution(ctx,
                            distribution,
                            flow):
    min_event_ndims = flow.get_y_event_ndims()
    max_event_ndims = (distribution.value_ndims +
                       (flow.get_y_event_ndims() - flow.get_x_event_ndims()))

    def fn(event_ndims, reparameterized, validate_tensors):
        # construct the instance
        kwargs = {}

        if reparameterized is not None:
            kwargs['reparameterized'] = reparameterized
        else:
            reparameterized = distribution.reparameterized

        if event_ndims is not None:
            kwargs['event_ndims'] = event_ndims
        else:
            event_ndims = flow.get_y_event_ndims()

        if validate_tensors is not None:
            kwargs['validate_tensors'] = validate_tensors
        else:
            validate_tensors = distribution.validate_tensors

        d = FlowDistribution(distribution, flow, **kwargs)

        # check the instance
        def log_prob_fn(t):
            log_px = distribution.log_prob(t.transform_origin.tensor,
                                           group_ndims=0)
            y, log_det = flow(t.transform_origin.tensor)  # y and log |dy/dx|
            assert_allclose(y, t.tensor, atol=1e-4, rtol=1e-6)
            ctx.assertEqual(
                T.rank(log_det),
                T.rank(log_px) - (flow.get_x_event_ndims() - distribution.event_ndims)
            )
            return -log_det + T.reduce_sum(
                log_px, T.int_range(
                    -(flow.get_x_event_ndims() - distribution.event_ndims),
                    0
                )
            )

        check_distribution_instance(
            ctx=ctx,
            d=d,
            event_ndims=event_ndims,
            batch_shape=distribution.batch_shape[: max_event_ndims - event_ndims],
            min_event_ndims=min_event_ndims,
            max_event_ndims=max_event_ndims,
            log_prob_fn=log_prob_fn,
            transform_origin_distribution=distribution,
            transform_origin_group_ndims=flow.get_x_event_ndims() - distribution.event_ndims,
            # other attributes
            base_distribution=distribution,
            flow=flow,
            dtype=distribution.dtype,
            continuous=distribution.continuous,
            reparameterized=reparameterized,
            validate_tensors=validate_tensors,
        )

    for event_ndims in (None,
                        min_event_ndims,
                        (min_event_ndims + max_event_ndims) // 2,
                        max_event_ndims):
        fn(event_ndims, None, None)

    for reparameterized in (None, True, False):
        fn(None, reparameterized, None)

    for validate_tensors in (None, True, False):
        fn(None, None, validate_tensors)


class FlowDistributionTestCase(unittest.TestCase):

    def test_FlowDistribution(self):
        T.random.seed(1234)

        check_flow_distribution(
            self,
            UnitNormal([], event_ndims=0),
            _MyFlow(x_event_ndims=0, y_event_ndims=0, explicitly_invertible=True),
        )

        check_flow_distribution(
            self,
            UnitNormal([2, 3, 4], event_ndims=0),
            _MyFlow(x_event_ndims=0, y_event_ndims=0, explicitly_invertible=True),
        )

        check_flow_distribution(
            self,
            UnitNormal([2, 3, 4], event_ndims=0),
            ActNorm(4),
        )

        check_flow_distribution(
            self,
            UnitNormal([2, 3, 4], event_ndims=1),
            ReshapeFlow([-1], [-1, 1]),
        )

        check_flow_distribution(
            self,
            UnitNormal([2, 3, 4], event_ndims=1),
            ReshapeFlow([-1, 1], [-1]),
        )

        # errors in constructor
        with pytest.raises(TypeError,
                           match='`distribution` is not an instance of '
                                 '`Distribution`'):
            _ = FlowDistribution(object(), ActNorm(3))

        with pytest.raises(TypeError, match='`flow` is not a flow'):
            _ = FlowDistribution(UnitNormal([3]), object())

        with pytest.raises(ValueError,
                           match='cannot be transformed by a flow, because '
                                 'it is not continuous'):
            _ = FlowDistribution(Categorical(logits=[0., 1., 2.]), ActNorm(3))

        with pytest.raises(ValueError,
                           match='cannot be transformed by a flow, because '
                                 'its `dtype` is not floating point'):
            normal = UnitNormal([3])
            normal.dtype = T.int32
            _ = FlowDistribution(normal, ActNorm(3))

        with pytest.raises(ValueError,
                           match='`distribution.event_ndims <= flow.'
                                 'x_event_ndims <= distribution.value_ndims` '
                                 'is not satisfied'):
            _ = FlowDistribution(UnitNormal([2, 3, 4], event_ndims=2),
                                 ActNorm(4))

        with pytest.raises(ValueError,
                           match='`distribution.event_ndims <= flow.'
                                 'x_event_ndims <= distribution.value_ndims` '
                                 'is not satisfied'):
            _ = FlowDistribution(UnitNormal([2, 3, 4], event_ndims=2),
                                 _MyFlow(x_event_ndims=4, y_event_ndims=4,
                                         explicitly_invertible=True))

        with pytest.raises(ValueError,
                           match='`event_ndims` out of range: .* '
                                 'minimum allowed value is 2, .* '
                                 'maximum allowed value is 4'):
            _ = FlowDistribution(
                UnitNormal([2, 3, 4]), ReshapeFlow([-1], [-1, 1]), event_ndims=1)

        with pytest.raises(ValueError,
                           match='`event_ndims` out of range: .* '
                                 'minimum allowed value is 2, .* '
                                 'maximum allowed value is 4'):
            _ = FlowDistribution(
                UnitNormal([2, 3, 4]), ReshapeFlow([-1], [-1, 1]), event_ndims=5)

    def test_copy(self):
        normal = UnitNormal([2, 3, 5], dtype=T.float64, validate_tensors=True)
        flow = ActNorm(5)
        distrib = FlowDistribution(normal, flow)
        self.assertEqual(distrib.event_ndims, 1)
        self.assertTrue(distrib.reparameterized)
        self.assertTrue(distrib.validate_tensors)

        with mock.patch('tensorkit.distributions.flow.copy_distribution',
                        wraps=copy_distribution) as f_copy:
            distrib2 = distrib.copy(event_ndims=2, reparameterized=False,
                                    validate_tensors=False)
            self.assertIsInstance(distrib2, FlowDistribution)
            self.assertIs(distrib2.flow, flow)
            self.assertIsInstance(distrib2.base_distribution, UnitNormal)
            self.assertEqual(distrib2.reparameterized, False)
            self.assertEqual(distrib2.event_ndims, 2)
            self.assertFalse(distrib2.validate_tensors)
            self.assertEqual(f_copy.call_args, ((), {
                'cls': FlowDistribution,
                'base': distrib,
                'attrs': (('distribution', '_base_distribution'), 'flow',
                          'reparameterized', 'event_ndims', 'validate_tensors'),
                'overrided_params': {'event_ndims': 2,
                                     'reparameterized': False,
                                     'validate_tensors': False},
            }))
