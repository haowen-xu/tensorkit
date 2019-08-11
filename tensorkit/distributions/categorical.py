from typing import *

from ..stochastic import StochasticTensor
from ..tensor import *
from .base import Distribution

__all__ = [
    'BaseCategorical', 'Categorical', 'OnehotCategorical',
]


class BaseCategorical(Distribution):

    def __init__(self,
                 *,
                 logits: Optional[Tensor],
                 probs: Optional[Tensor],
                 dtype: str,
                 event_ndims: int,
                 min_event_ndims: int,
                 check_numerics: Optional[bool],
                 epsilon: float = 1e-7):
        if (logits is None) == (probs is None):
            raise ValueError('Either `logits` or `probs` must be specified, '
                             'but not both.')

        epsilon = float(epsilon)

        if logits is not None:
            param_shape = shape(logits)
            probs = None
            original_arg = 'logits'
        else:
            param_shape = shape(probs)
            logits = None
            original_arg = 'probs'

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
    def original_arg(self) -> str:
        return self._original_arg

    @property
    def logits(self) -> Tensor:
        if self._logits is None:
            self._logits = log(
                clip(self._probs, self._epsilon, 1 - self._epsilon))
        return self._logits

    @property
    def probs(self) -> Tensor:
        if self._probs is None:
            self._probs = softmax(self._logits)
        return self._probs

    @property
    def n_classes(self) -> int:
        return self._n_classes

    def to_indexed(self) -> 'Categorical':
        kwargs = {self.original_arg: getattr(self, self.original_arg)}
        event_ndims = self.event_ndims
        if isinstance(self, OnehotCategorical):
            event_ndims -= 1
        return Categorical(
            dtype=self.dtype,
            event_ndims=event_ndims,
            check_numerics=self._check_numerics,
            epsilon=self._epsilon,
            **kwargs,
        )

    def to_one_hot(self) -> 'OnehotCategorical':
        kwargs = {self.original_arg: getattr(self, self.original_arg)}
        event_ndims = self.event_ndims
        if isinstance(self, Categorical):
            event_ndims += 1
        return OnehotCategorical(
            dtype=self.dtype,
            event_ndims=event_ndims,
            check_numerics=self._check_numerics,
            epsilon=self._epsilon,
            **kwargs,
        )


class Categorical(BaseCategorical):

    def __init__(self,
                 *,
                 logits: Optional[Tensor] = None,
                 probs: Optional[Tensor] = None,
                 dtype: str = index_dtype,
                 event_ndims: int = 0,
                 check_numerics: Optional[bool] = None,
                 epsilon: float = 1e-7):
        super().__init__(
            logits=logits,
            probs=probs,
            dtype=dtype,
            event_ndims=event_ndims,
            min_event_ndims=0,
            check_numerics=check_numerics,
            epsilon=epsilon,
        )

    def _sample(self, n_samples: Optional[int]):
        param_shape = self._param_shape
        arg = getattr(self, self.original_arg)
        if n_samples is not None:
            param_shape = (n_samples,) + param_shape
            arg = expand(arg, param_shape)
        return random.categorical(dtype=self.dtype, **{self.original_arg: arg})

    def log_prob(self, given: Tensor, group_ndims: int = 0) -> Tensor:
        log_p = cross_entropy_with_logits(
            logits=self.logits, labels=given, negative=True)
        return log_p

    def to_indexed(self) -> 'Categorical':
        return self

    def copy(self, **kwargs):
        return self._copy_helper(
            (self.original_arg, 'dtype', 'event_ndims', 'check_numerics',
             'random_state', 'epsilon'),
            **kwargs
        )


class OnehotCategorical(BaseCategorical):

    def __init__(self,
                 *,
                 logits: Optional[Tensor] = None,
                 probs: Optional[Tensor] = None,
                 dtype: str = index_dtype,
                 event_ndims: int = 1,
                 check_numerics: Optional[bool] = None,
                 epsilon: float = 1e-7):
        super().__init__(
            logits=logits,
            probs=probs,
            dtype=dtype,
            event_ndims=event_ndims,
            min_event_ndims=1,
            check_numerics=check_numerics,
            epsilon=epsilon,
        )

    def _sample(self, n_samples: Optional[int]):
        param_shape = self._param_shape
        arg = getattr(self, self.original_arg)
        if n_samples is not None:
            param_shape = (n_samples,) + param_shape
            arg = expand(arg, param_shape)
        samples = random.categorical(**{self.original_arg: arg})
        samples = one_hot(samples, self.n_classes)
        if samples.dtype != self.dtype:
            samples = cast(samples, self.dtype)
        return samples

    def log_prob(self, given: Tensor, group_ndims: int = 0) -> Tensor:
        return sparse_cross_entropy_with_logits(
            logits=self.logits, labels=given, negative=True)

    def to_one_hot(self) -> 'OnehotCategorical':
        return self

    def copy(self, **kwargs):
        return self._copy_helper(
            (self.original_arg, 'dtype', 'event_ndims', 'check_numerics',
             'random_state', 'epsilon'),
            **kwargs
        )
