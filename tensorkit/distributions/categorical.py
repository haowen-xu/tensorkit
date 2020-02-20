from typing import *

from .. import tensor as T
from ..stochastic import StochasticTensor
from ..typing_ import *
from .base import Distribution
from .utils import copy_distribution, check_tensor_arg_types

__all__ = ['BaseCategorical', 'Categorical', 'OneHotCategorical']


class BaseCategorical(Distribution):

    continuous = False
    reparameterized = False

    _logits: Optional[T.Tensor]
    """Logits of the probabilities of being each class."""

    _probs: Optional[T.Tensor]
    """The probabilities of being each class."""

    n_classes: int
    """The number of categorical classes."""

    epsilon: float
    """The infinitesimal constant, used for computing `logits`."""

    _mutual_params: Dict[str, Any]
    """Dict that stores the original `logits` or `probs` constructor argument."""

    def __init__(self,
                 *,
                 logits: Optional[TensorOrData],
                 probs: Optional[TensorOrData],
                 dtype: str,
                 event_ndims: int,
                 epsilon: float = T.EPSILON,
                 device: Optional[str] = None,
                 validate_tensors: Optional[bool] = None):
        (logits, probs), = check_tensor_arg_types([('logits', logits),
                                                   ('probs', probs)],
                                                  device=device)
        if logits is not None:
            param_shape = T.shape(logits)
            mutual_params = {'logits': logits}
            device = device or T.get_device(logits)
        else:
            param_shape = T.shape(probs)
            mutual_params = {'probs': probs}
            device = device or T.get_device(probs)
        epsilon = float(epsilon)

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
            device=device,
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
    def logits(self) -> T.Tensor:
        """Logits of the probabilities of being each class."""
        if self._logits is None:
            self._logits = T.random.categorical_probs_to_logits(self._probs,
                                                                self.epsilon)
        return self._logits

    @property
    def probs(self) -> T.Tensor:
        """The probabilities of being each class."""
        if self._probs is None:
            self._probs = T.random.categorical_logits_to_probs(self._logits)
        return self._probs

    def to_indexed(self, dtype: str = T.categorical_dtype) -> 'Categorical':
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

    def to_one_hot(self, dtype: str = T.int32) -> 'OneHotCategorical':
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
            attrs=('dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
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
                 logits: Optional[TensorOrData] = None,
                 probs: Optional[TensorOrData] = None,
                 dtype: str = T.categorical_dtype,
                 event_ndims: int = 0,
                 epsilon: float = T.EPSILON,
                 device: Optional[str] = None,
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
            epsilon=epsilon,
            device=device,
            validate_tensors=validate_tensors,
        )

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        samples = T.random.categorical(
            probs=self.probs, n_samples=n_samples, dtype=self.dtype)
        return StochasticTensor(
            distribution=self, tensor=samples, n_samples=n_samples,
            group_ndims=group_ndims, reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return T.random.categorical_log_prob(
            given=given, logits=self.logits, group_ndims=reduce_ndims)

    def to_indexed(self, dtype: str = T.categorical_dtype) -> 'Categorical':
        return self if dtype == self.dtype else self.copy(dtype=dtype)

    def to_one_hot(self, dtype: str = T.int32) -> 'OneHotCategorical':
        return copy_distribution(
            cls=OneHotCategorical,
            base=self,
            attrs=('dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
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
                 logits: Optional[TensorOrData] = None,
                 probs: Optional[TensorOrData] = None,
                 dtype: str = T.int32,
                 event_ndims: int = 1,
                 epsilon: float = T.EPSILON,
                 device: Optional[str] = None,
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
            device: The device where to place new tensors and variables.
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
            epsilon=epsilon,
            device=device,
            validate_tensors=validate_tensors,
        )

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        samples = T.random.one_hot_categorical(
            probs=self.probs, n_samples=n_samples, dtype=self.dtype)
        return StochasticTensor(
            distribution=self, tensor=samples, n_samples=n_samples,
            group_ndims=group_ndims, reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return T.random.one_hot_categorical_log_prob(
            given=given, logits=self.logits, group_ndims=reduce_ndims)

    def to_indexed(self, dtype: str = T.categorical_dtype) -> 'Categorical':
        return copy_distribution(
            cls=Categorical,
            base=self,
            attrs=('dtype', 'event_ndims', 'epsilon', 'device', 'validate_tensors'),
            mutual_attrs=(('logits', 'probs'),),
            compute_deps={'logits': ('epsilon',)},
            original_mutual_params=self._mutual_params,
            overrided_params={'dtype': dtype,
                              'event_ndims': self.event_ndims - 1}
        )

    def to_one_hot(self, dtype: str = T.int32) -> 'OneHotCategorical':
        return self if dtype == self.dtype else self.copy(dtype=dtype)
