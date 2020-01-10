from typing import *

from .. import backend as Z
from ..stochastic import StochasticTensor
from .base import Distribution
from .utils import copy_distribution

__all__ = ['Categorical', 'OneHotCategorical']


class BaseCategorical(Distribution):

    continuous = False
    reparameterized = False

    _logits: Optional[Z.Tensor]
    """Logits of the probabilities of being each class."""

    _probs: Optional[Z.Tensor]
    """The probabilities of being each class."""

    n_classes: int
    """The number of categorical classes."""

    epsilon: float
    """The infinitesimal constant, used for computing `logits`."""

    _mutual_params: Dict[str, Any]
    """Dict that stores the original `logits` or `probs` constructor argument."""

    def __init__(self,
                 *,
                 logits: Optional[Z.Tensor],
                 probs: Optional[Z.Tensor],
                 dtype: str,
                 event_ndims: int,
                 epsilon: float = 1e-7,
                 validate_tensors: Optional[bool] = None):
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')

        epsilon = float(epsilon)

        if logits is not None:
            param_shape = Z.shape(logits)
            probs = None
            mutual_params = {'logits': logits}
        else:
            param_shape = Z.shape(probs)
            logits = None
            mutual_params = {'probs': probs}

        if len(param_shape) < 1:
            for param_key in mutual_params:
                raise ValueError(f'`{param_key}` must be at least 1d: '
                                 f'got shape {param_shape}.')
        value_shape = (param_shape if self.min_event_ndims == 1
                       else param_shape[:-1])
        n_classes = param_shape[-1]

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
        self.n_classes = n_classes
        self.epsilon = epsilon
        self._mutual_params = mutual_params

    @property
    def logits(self) -> Z.Tensor:
        """Logits of the probabilities of being each class."""
        if self._logits is None:
            self._logits = Z.random.categorical_probs_to_logits(self._probs,
                                                                self.epsilon)
        return self._logits

    @property
    def probs(self) -> Z.Tensor:
        """The probabilities of being each class."""
        if self._probs is None:
            self._probs = Z.random.categorical_logits_to_probs(self._logits)
        return self._probs

    def to_indexed(self, dtype: str = Z.categorical_dtype) -> 'Categorical':
        """
        Get a :class:`Categorical` object according to this distribution.

        Args:
            dtype: The dtype of the returned distribution.

        Returns:
            The current distribution itself, if this is already an instance of
            :class:`Categorical`, and its dtype equals to `dtype`.
            Otherwise returns  a new :class:`Categorical` instance.
        """
        raise NotImplementedError()

    def to_one_hot(self, dtype: str = Z.int32) -> 'OneHotCategorical':
        """
        Get a :class:`OneHotCategorical` object according to this distribution.

        Args:
            dtype: The dtype of the returned distribution.

        Returns:
            The current distribution itself, if this is already an instance of
            :class:`OneHotCategorical`, and its dtype equals to `dtype`.
            Otherwise returns a new :class:`OneHotCategorical` instance.
        """
        raise NotImplementedError()

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=self.__class__,
            base=self,
            attrs=('dtype', 'event_ndims', 'validate_tensors', 'epsilon'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params=overrided_params,
        )


class Categorical(BaseCategorical):
    """
    Categorical distribution.

    A categorical random variable is a discrete random variable being one of
    `1` to `n_classes - 1`.  The probability of being each possible value is
    specified via the `probs`, or the `logits` of the probs.
    """

    min_event_ndims = 0

    def __init__(self,
                 *,
                 logits: Optional[Z.Tensor] = None,
                 probs: Optional[Z.Tensor] = None,
                 dtype: str = Z.categorical_dtype,
                 event_ndims: int = 0,
                 epsilon: float = 1e-7,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`Categorical` distribution object.

        Args:
            logits: The logits of the probabilities of being each possible value.
                ``logits = log(p)``.
                Either `logits` or `probs` must be specified, but not both.
            probs: The probability `p` of being each possible value.
                ``p = softmax(logits)``.
            dtype: The dtype of the samples.
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            epsilon: The infinitesimal constant, used for computing `logits`.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        super().__init__(
            logits=logits,
            probs=probs,
            dtype=dtype,
            event_ndims=event_ndims,
            validate_tensors=validate_tensors,
            epsilon=epsilon,
        )

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        samples = Z.random.categorical(
            probs=self.probs, n_samples=n_samples, dtype=self.dtype)
        return StochasticTensor(
            distribution=self, tensor=samples, n_samples=n_samples,
            group_ndims=group_ndims, reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: Z.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> Z.Tensor:
        return Z.random.categorical_log_prob(
            given=given, logits=self.logits, group_ndims=reduce_ndims)

    def to_indexed(self, dtype: str = Z.categorical_dtype) -> 'Categorical':
        return self if dtype == self.dtype else self.copy(dtype=dtype)

    def to_one_hot(self, dtype: str = Z.int32) -> 'OneHotCategorical':
        return copy_distribution(
            cls=OneHotCategorical,
            base=self,
            attrs=('dtype', 'validate_tensors', 'event_ndims', 'epsilon'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params={'dtype': dtype,
                              'event_ndims': self.event_ndims + 1}
        )


class OneHotCategorical(BaseCategorical):
    """
    One-hot categorical distribution.

    A one-hot categorical random variable is a `n_classes` binary vector,
    with only one element being 1 and all remaining elements being 0.
    The probability of each element being 1 is specified via thevia the
    `probs`, or the `logits` of the probs.
    """

    min_event_ndims = 1

    def __init__(self,
                 *,
                 logits: Optional[Z.Tensor] = None,
                 probs: Optional[Z.Tensor] = None,
                 dtype: str = Z.int32,
                 event_ndims: int = 1,
                 epsilon: float = 1e-7,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`OneHotCategorical` distribution object.

        Args:
            logits: The logits of the probabilities of being each possible value.
                ``logits = log(p)``.
                Either `logits` or `probs` must be specified, but not both.
            probs: The probability `p` of being each possible value.
                ``p = softmax(logits)``.
            dtype: The dtype of the samples.
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            epsilon: The infinitesimal constant, used for computing `logits`.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        super().__init__(
            logits=logits,
            probs=probs,
            dtype=dtype,
            event_ndims=event_ndims,
            validate_tensors=validate_tensors,
            epsilon=epsilon,
        )

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        samples = Z.random.one_hot_categorical(
            probs=self.probs, n_samples=n_samples, dtype=self.dtype)
        return StochasticTensor(
            distribution=self, tensor=samples, n_samples=n_samples,
            group_ndims=group_ndims, reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: Z.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> Z.Tensor:
        return Z.random.one_hot_categorical_log_prob(
            given=given, logits=self.logits, group_ndims=reduce_ndims)

    def to_indexed(self, dtype: str = Z.categorical_dtype) -> 'Categorical':
        return copy_distribution(
            cls=Categorical,
            base=self,
            attrs=('dtype', 'validate_tensors', 'event_ndims', 'epsilon'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params={'dtype': dtype,
                              'event_ndims': self.event_ndims - 1}
        )

    def to_one_hot(self, dtype: str = Z.int32) -> 'OneHotCategorical':
        return self if dtype == self.dtype else self.copy(dtype=dtype)
