from typing import *

from mltk.utils import optional_apply

from . import tensor as T

__all__ = ['StochasticTensor']


class StochasticTensor(T.TensorWrapper):

    # define the attributes of this class, such that `T.TensorWrapper`
    # will treat them as the attributes of `StochasticTensor`, rather
    # than the attributes of `T.Tensor`.
    _distribution: 'Distribution' = None
    _tensor: T.Tensor = None
    _n_samples: Optional[int] = None
    _group_ndims: int = 0
    _is_reparameterized: Optional[bool] = None
    _transform_origin: Optional[T.Tensor] = None
    _log_prob: Optional[T.Tensor] = None
    _prob: Optional[T.Tensor] = None

    def __init__(self,
                 distribution: 'Distribution',
                 tensor: T.Tensor,
                 n_samples: Optional[int] = None,
                 group_ndims: int = 0,
                 is_reparameterized: Optional[bool] = None,
                 transform_origin: Optional[T.Tensor] = None,
                 log_prob: Optional[T.Tensor] = None
                 ):
        self._distribution = distribution
        self._tensor = tensor
        self._n_samples = optional_apply(int, n_samples)
        self._group_ndims = group_ndims
        self._is_reparameterized = optional_apply(bool, is_reparameterized)
        self._transform_origin = transform_origin
        self._log_prob = log_prob

    def __repr__(self):
        return 'StochasticTensor({!r})'.format(self.tensor)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    @property
    def distribution(self) -> 'Distribution':
        return self._distribution

    @property
    def tensor(self) -> T.Tensor:
        return self._tensor

    @property
    def n_samples(self) -> Optional[int]:
        return self._n_samples

    @property
    def group_ndims(self) -> int:
        return self._group_ndims

    @property
    def is_reparameterized(self) -> Optional[bool]:
        return self._is_reparameterized

    @property
    def transform_origin(self) -> Optional[T.Tensor]:
        return self._transform_origin

    def log_prob(self, group_ndims: Optional[int] = None):
        if group_ndims is None or group_ndims == self.group_ndims:
            if self._log_prob is None:
                self._log_prob = \
                    self.distribution.log_prob(self.tensor, self.group_ndims)
            return self._log_prob
        else:
            return self.distribution.log_prob(self.tensor, group_ndims)

    def prob(self, group_ndims: Optional[int] = None):
        def compute_prob(log_p):
            prob = T.exp(log_p)
            # if isinstance(log_p, FlowDistributionDerivedTensor):
            #     # copy the `flow origin` information from log_prob to prob
            #     prob = FlowDistributionDerivedTensor(
            #         prob, flow_origin=log_p.flow_origin)
            return prob

        if group_ndims is None or group_ndims == self.group_ndims:
            if self._prob is None:
                self._prob = compute_prob(self.log_prob())
            return self._prob
        else:
            log_p = self.distribution.log_prob(self.tensor, group_ndims)
            return compute_prob(log_p)


T.register_tensor_wrapper_class(StochasticTensor)

# back reference to the Distribution class
from .distributions.base import Distribution
