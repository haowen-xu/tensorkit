from typing import *

from .. import tensor as T
from ..tensor import typing as Z
from ..stochastic import StochasticTensor
from .base import Distribution

__all__ = ['Bernoulli']


class Bernoulli(Distribution):

    def __init__(self,
                 *,
                 logits: Optional[Z.TensorLike] = None,
                 probs: Optional[Z.TensorLike] = None,
                 dtype: Z.DTypeLike = T.int32,
                 event_ndims: int = 0,
                 check_numerics: Optional[bool] = None,
                 random_state: Optional[T.random.RandomState] = None,
                 epsilon: float = 1e-7):
        # validate the arguments
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')

        epsilon = float(epsilon)

        if logits is not None:
            logits = T.as_tensor(logits)
            param_shape = T.shape(logits)
            probs = None
            original_arg = 'logits'
        else:
            probs = T.as_tensor(probs)
            param_shape = T.shape(probs)
            logits = None
            original_arg = 'probs'

        # construct the instance
        super().__init__(
            dtype=dtype, is_continuous=False, is_reparameterized=False,
            value_shape=param_shape, event_ndims=event_ndims, min_event_ndims=0,
            check_numerics=check_numerics, random_state=random_state,
        )

        self._param_shape = param_shape
        self._logits = logits
        self._probs = probs
        self._epsilon = epsilon
        self._original_arg = original_arg

    @property
    def original_arg(self) -> str:
        return self._original_arg

    @property
    def logits(self) -> T.Tensor:
        if self._logits is None:
            probs_clipped = T.clip(
                self._probs, self._epsilon, 1 - self._epsilon)
            self._logits = T.log(probs_clipped) - T.log1p(-probs_clipped)
        return self._logits

    @property
    def probs(self) -> T.Tensor:
        if self._probs is None:
            self._probs = T.nn.sigmoid(self._logits)
        return self._probs

    def sample(self, n_samples: Optional[int] = None,
               group_ndims: int = 0,
               is_reparameterized: Optional[bool] = None,
               compute_prob: Optional[bool] = None) -> 'StochasticTensor':
        # validate arguments
        is_reparameterized = self._check_reparameterized(is_reparameterized)

        # generate samples
        arg = getattr(self, self.original_arg)
        param_shape = self._param_shape
        if n_samples is not None:
            param_shape = (n_samples,) + param_shape
            arg = T.expand(arg, param_shape)
        samples = T.random.bernoulli(
            dtype=self.dtype, random_state=self.random_state,
            **{self.original_arg: arg}
        )

        # compose the stochastic tensor
        t = StochasticTensor(
            distribution=self,
            tensor=samples,
            n_samples=n_samples,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
        )
        if compute_prob is True:
            _ = t.log_prob()

        return t

    def log_prob(self, given: Z.TensorLike, group_ndims: int = 0) -> T.Tensor:
        return T.nn.binary_cross_entropy_with_logits(
            logits=self.logits, labels=given, negative=True)

    def copy(self, **kwargs):
        return self._copy_helper(
            (self.original_arg, 'dtype', 'event_ndims', 'check_numerics',
             'random_state', 'epsilon'),
            **kwargs
        )
