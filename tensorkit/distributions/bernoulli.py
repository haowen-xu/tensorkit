from typing import *

from .. import tensor as T
from ..stochastic import StochasticTensor
from .base import Distribution
from .utils import copy_distribution

__all__ = ['Bernoulli']


class Bernoulli(Distribution):
    """
    Bernoulli distribution.

    A bernoulli random variable is a discrete random variable, with probability
    `p` of being 1, and probability `1 - p` of being 0.
    """

    continuous = False
    reparameterized = False
    min_event_ndims = 0

    _logits: Optional[T.Tensor]
    """Logits of the probability of being 1."""

    _probs: Optional[T.Tensor]
    """The probability of being 1."""

    epsilon: float
    """The infinitesimal constant, used for computing `logits`."""

    _mutual_params: Dict[str, Any]
    """Dict that stores the original `logits` or `probs` constructor argument."""

    def __init__(self,
                 *,
                 logits: Optional[T.Tensor] = None,
                 probs: Optional[T.Tensor] = None,
                 dtype: str = T.int32,
                 event_ndims: int = 0,
                 epsilon: float = 1e-7,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`Bernoulli` distribution object.

        Args:
            logits: The logits of the probability of being 1.
                ``logits = log(p) - log(1-p)``.
                Either `logits` or `probs` must be specified, but not both.
            probs: The probability `p` of being 1.  ``p = sigmoid(logits)``.
            dtype: The dtype of the samples.
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            epsilon: The infinitesimal constant, used for computing `logits`.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        # validate the arguments
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')

        epsilon = float(epsilon)

        if logits is not None:
            value_shape = T.shape(logits)
            probs = None
            mutual_params = {'logits': logits}
        else:
            value_shape = T.shape(probs)
            logits = None
            mutual_params = {'probs': probs}

        # construct the object
        super().__init__(
            dtype=dtype,
            value_shape=value_shape,
            event_ndims=event_ndims,
            validate_tensors=validate_tensors,
        )
        for k, v in mutual_params.items():
            mutual_params[k] = self._assert_finite(v, k)

        self._logits = logits
        self._probs = probs
        self.epsilon = epsilon
        self._mutual_params = mutual_params

    @property
    def logits(self) -> T.Tensor:
        """Get the logits of the probability of being 1."""
        if self._logits is None:
            self._logits = T.random.bernoulli_probs_to_logits(self._probs,
                                                              self.epsilon)
        return self._logits

    @property
    def probs(self) -> T.Tensor:
        """Get the probability of being 1."""
        if self._probs is None:
            self._probs = T.random.bernoulli_logits_to_probs(self._logits)
        return self._probs

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        # generate samples
        samples = T.random.bernoulli(probs=self.probs,
                                     n_samples=n_samples,
                                     dtype=self.dtype)

        # compose the stochastic tensor
        return StochasticTensor(
            distribution=self,
            tensor=samples,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return T.random.bernoulli_log_prob(
            given=given,
            logits=self.logits,
            group_ndims=reduce_ndims,
        )

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=Bernoulli,
            base=self,
            attrs=('dtype', 'event_ndims', 'validate_tensors', 'epsilon'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params=overrided_params,
        )
