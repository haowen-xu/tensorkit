from typing import *

from .. import tensor as T
from ..flows import Flow
from ..stochastic import StochasticTensor
from .base import Distribution
from .utils import copy_distribution, get_overrided_parameterized

__all__ = ['FlowDistribution']


class FlowDistribution(Distribution):
    """
    Transform a :class:`Distribution` by a :class:`BaseFlow`, as a new
    distribution.
    """

    _base_distribution: Distribution
    """The base distribution, which is transform by the `flow`."""

    flow: Flow
    """The flow instance, which transforms the `distribution`."""

    _base_group_ndims: int
    """`group_ndims` for computing log p(x) by `base_distribution` p(x)."""

    def __init__(self,
                 distribution: Distribution,
                 flow: Flow,
                 reparameterized: Optional[bool] = None,
                 event_ndims: Optional[int] = None,
                 validate_tensors: Optional[bool] = None):
        # check the type of `distribution` and `flow`
        if not isinstance(distribution, Distribution):
            raise TypeError(f'`distribution` is not an instance of '
                            f'`Distribution`: got {distribution!r}')
        if not isinstance(flow, Flow) and not T.is_jit_layer(flow):
            raise TypeError(f'`flow` is not a flow: {flow!r}')

        # `distribution` is required to be continuous and have float dtype.
        continuous = distribution.continuous
        if not distribution.continuous:
            raise ValueError(
                f'Distribution {distribution!r} cannot be transformed by a '
                f'flow, because it is not continuous.')

        dtype = distribution.dtype
        if not T.is_floating_point_dtype(dtype):
            raise ValueError(
                f'Distribution {distribution!r} cannot be transformed by a '
                f'flow, because its `dtype` is not floating point.'
            )

        # requirement: distribution.event_ndims <= flow.x_event_ndims <= distribution.value_ndims
        #              otherwise the distribution cannot be transformed by the flow
        if not (distribution.event_ndims <= flow.get_x_event_ndims() <=
                distribution.value_ndims):
            raise ValueError(
                f'`distribution.event_ndims <= flow.x_event_ndims <= '
                f'distribution.value_ndims` is not satisfied: '
                f'`distribution.event_ndims` is {distribution.event_ndims}, '
                f'while `flow.x_event_ndims` is {flow.get_x_event_ndims()}.'
            )

        # requirement: min_event_ndims <= event_ndims <= max_event_ndims
        min_event_ndims = flow.get_y_event_ndims()
        max_event_ndims = (distribution.value_ndims +
                           (flow.get_y_event_ndims() - flow.get_x_event_ndims()))
        if event_ndims is not None and \
                not (min_event_ndims <= event_ndims <= max_event_ndims):
            raise ValueError(
                f'`event_ndims` out of range: got {event_ndims}, but '
                f'the minimum allowed value is {min_event_ndims}, '
                f'and the maximum allowed value is {max_event_ndims}.'
            )

        # obtain the arguments
        if event_ndims is None:
            event_ndims = flow.get_y_event_ndims()
        batch_ndims = max_event_ndims - event_ndims
        batch_shape = distribution.batch_shape[:batch_ndims]
        reparameterized = get_overrided_parameterized(
            distribution.reparameterized,
            reparameterized,
            distribution.__class__,
        )
        if validate_tensors is None:
            validate_tensors = distribution.validate_tensors

        base_group_ndims = flow.get_x_event_ndims() - distribution.event_ndims

        # now construct the instance
        super(FlowDistribution, self).__init__(
            dtype=dtype, batch_shape=batch_shape, continuous=continuous,
            reparameterized=reparameterized, event_ndims=event_ndims,
            min_event_ndims=min_event_ndims, validate_tensors=validate_tensors,
        )
        self._base_distribution = distribution
        self.flow = flow
        self._base_group_ndims = base_group_ndims

    @property
    def base_distribution(self):
        """Get the base distribution."""
        return self._base_distribution

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> 'StochasticTensor':
        x = self._base_distribution.sample(
            n_samples=n_samples, group_ndims=self._base_group_ndims,
            reparameterized=reparameterized
        )
        log_px = x.log_prob()

        # y, log |dy/dx|
        reparameterized = x.reparameterized
        y, log_det = self.flow(x.tensor, compute_log_det=True)
        if not reparameterized:
            y = T.stop_grad(y)  # important!

        # compute log p(y) = log p(x) - log |dy/dx|
        # and then apply `group_ndims` on log p(y)
        log_py = log_px - log_det
        if reduce_ndims > 0:
            log_py = T.reduce_sum(log_py, axis=T.int_range(-reduce_ndims, 0))
        log_py.transform_origin = x  # also add `transform_origin` to the log_py tensor; hope it will be accessible

        # compose the transformed tensor
        return StochasticTensor(
            tensor=y,
            distribution=self,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized,
            transform_origin=x,
            log_prob=log_py,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        # x, log |dx/dy|
        x, log_det = self.flow(given, inverse=True)

        # log p(x)
        log_px = self._base_distribution.log_prob(
            x, group_ndims=self._base_group_ndims)

        # compute log p(y) = log p(x) + log |dx/dy|,
        # and then apply `group_ndims` on log p(x)
        log_py = log_px + log_det
        if reduce_ndims > 0:
            log_py = T.reduce_sum(log_py, axis=T.int_range(-reduce_ndims, 0))

        # also add `transform_origin` to the log_py tensor; hope it will be accessible
        log_py.transform_origin = StochasticTensor(
            tensor=x,
            distribution=self._base_distribution,
            n_samples=None,
            group_ndims=self._base_group_ndims,
            reparameterized=self._base_distribution.reparameterized,
        )

        return log_py

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=FlowDistribution,
            base=self,
            attrs=(('distribution', '_base_distribution'), 'flow',
                   'reparameterized', 'event_ndims', 'validate_tensors'),
            overrided_params=overrided_params,
        )
