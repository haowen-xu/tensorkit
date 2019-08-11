from typing import *

from mltk.utils import optional_apply

from . import tensor as T

__all__ = ['StochasticTensor']


class StochasticTensor(T.ExtendedTensor):

    distribution: 'Distribution' = None
    n_samples: Optional[int] = None
    group_ndims: int = 0
    is_reparameterized: Optional[bool] = None
    transform_origin: Optional[T.Tensor] = None
    _cached_log_prob: Optional[T.Tensor] = None
    _cached_prob: Optional[T.Tensor] = None

    def init_extension(self,
                       distribution: 'Distribution',
                       n_samples: Optional[int] = None,
                       group_ndims: int = 0,
                       is_reparameterized: Optional[bool] = None,
                       transform_origin: Optional[T.Tensor] = None,
                       log_prob: Optional[T.Tensor] = None):
        self.distribution = distribution
        self.n_samples = optional_apply(int, n_samples)
        self.group_ndims = group_ndims
        self.is_reparameterized = optional_apply(bool, is_reparameterized)
        self.transform_origin = transform_origin
        self._cached_log_prob = log_prob

    def log_prob(self, group_ndims: Optional[int] = None):
        if group_ndims is None or group_ndims == self.group_ndims:
            if self._cached_log_prob is None:
                self._cached_log_prob = self.distribution.log_prob(
                    self.as_tensor(),
                    self.group_ndims
                )
            return self._cached_log_prob
        else:
            return self.distribution.log_prob(self.as_tensor(), group_ndims)

    def prob(self, group_ndims: Optional[int] = None):
        def compute_prob(log_p):
            prob = T.exp(log_p)
            # if isinstance(log_p, FlowDistributionDerivedTensor):
            #     # copy the `flow origin` information from log_prob to prob
            #     prob = FlowDistributionDerivedTensor(
            #         prob, flow_origin=log_p.flow_origin)
            return prob

        if group_ndims is None or group_ndims == self.group_ndims:
            if self._cached_prob is None:
                self._cached_prob = compute_prob(self.log_prob())
            return self._cached_prob
        else:
            log_p = self.distribution.log_prob(self.as_tensor(), group_ndims)
            return compute_prob(log_p)


T.register_extended_tensor_class(StochasticTensor)

# back reference to the Distribution class
from .distributions.base import Distribution
