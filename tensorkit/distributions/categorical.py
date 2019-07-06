from typing import *

from .. import tensor as T
from ..stochastic import StochasticTensor
from .base import Distribution

__all__ = [
    'Categorical', 'OnehotCategorical',
]


class _BaseCategorical(Distribution):

    def __init__(self,
                 *,
                 logits: Optional[T.TensorLike],
                 probs: Optional[T.TensorLike],
                 dtype: T.DTypeLike,
                 event_ndims: int,
                 min_event_ndims: int,
                 check_numerics: Optional[bool],
                 random_state: Optional[T.random.RandomState],
                 epsilon: float = 1e-7):
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')

        epsilon = float(epsilon)

        if logits is not None:
            logits = T.as_tensor(logits)
            probs = T.nn.softmax(logits)
            original_arg = 'logits'
        else:
            probs = T.as_tensor(probs)
            logits = T.log(T.clip(probs, epsilon, 1 - epsilon))
            original_arg = 'probs'

        param_shape = T.shape(probs)
        if len(param_shape) < 1:
            raise ValueError(f'`logits` must be at least 1d: got shape '
                             f'{param_shape}.')
        value_shape = param_shape if min_event_ndims == 1 else param_shape[:-1]
        n_classes = param_shape[-1]

        super().__init__(
            dtype=dtype,
            is_continuous=False,
            is_reparameterized=False,
            value_shape=value_shape,
            event_ndims=event_ndims,
            min_event_ndims=min_event_ndims,
            check_numerics=check_numerics,
            random_state=random_state,
        )
        self._epsilon = epsilon
        self._logits = logits
        self._probs = probs
        self._param_shape = param_shape
        self._n_classes = n_classes
        self._original_arg = original_arg

    def _sample(self, n_samples: Optional[int]):
        raise NotImplementedError()

    def sample(self,
               n_samples: Optional[int] = None,
               group_ndims: int = 0,
               is_reparameterized: Optional[bool] = None,
               compute_prob: Optional[bool] = None) -> 'StochasticTensor':
        # validate arguments
        is_reparameterized = self._check_reparameterized(is_reparameterized)

        # generate samples
        samples = self._sample(n_samples)

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

    @property
    def logits(self) -> T.Tensor:
        return self._logits

    @property
    def probs(self) -> T.Tensor:
        return self._probs

    @property
    def n_classes(self) -> int:
        return self._n_classes


class Categorical(_BaseCategorical):

    def __init__(self,
                 *,
                 logits: Optional[T.TensorLike] = None,
                 probs: Optional[T.TensorLike] = None,
                 dtype: T.DTypeLike = T.random.CATEGORICAL_DTYPE,
                 event_ndims: int = 0,
                 check_numerics: Optional[bool] = None,
                 random_state: Optional[T.random.RandomState] = None):
        super().__init__(
            logits=logits,
            probs=probs,
            dtype=dtype,
            event_ndims=event_ndims,
            min_event_ndims=0,
            check_numerics=check_numerics,
            random_state=random_state
        )

    def _sample(self, n_samples: Optional[int]):
        param_shape = self._param_shape
        arg = getattr(self, self._original_arg)
        if n_samples is not None:
            param_shape = (n_samples,) + param_shape
            arg = T.expand(arg, param_shape)
        return T.random.categorical(
            dtype=self.dtype, random_state=self.random_state,
            **{self._original_arg: arg}
        )

    def log_prob(self, given: T.TensorLike, group_ndims: int = 0) -> T.Tensor:
        log_p = T.nn.cross_entropy_with_logits(
            logits=self.logits, labels=given, negative=True)
        return log_p


class OnehotCategorical(_BaseCategorical):

    def __init__(self,
                 *,
                 logits: Optional[T.TensorLike] = None,
                 probs: Optional[T.TensorLike] = None,
                 dtype: T.DTypeLike = T.random.CATEGORICAL_DTYPE,
                 event_ndims: int = 1,
                 check_numerics: Optional[bool] = None,
                 random_state: Optional[T.random.RandomState] = None):
        super().__init__(
            logits=logits,
            probs=probs,
            dtype=dtype,
            event_ndims=event_ndims,
            min_event_ndims=1,
            check_numerics=check_numerics,
            random_state=random_state
        )

    def _sample(self, n_samples: Optional[int]):
        param_shape = self._param_shape
        arg = getattr(self, self._original_arg)
        if n_samples is not None:
            param_shape = (n_samples,) + param_shape
            arg = T.expand(arg, param_shape)
        samples = T.random.categorical(
            random_state=self.random_state, **{self._original_arg: arg})
        samples = T.nn.one_hot(samples, self.n_classes)
        if samples.dtype != self.dtype:
            samples = T.cast(samples, self.dtype)
        return samples

    def log_prob(self, given: T.Tensor, group_ndims: int = 0) -> T.Tensor:
        return T.nn.sparse_cross_entropy_with_logits(
            logits=self.logits, labels=given, negative=True)
