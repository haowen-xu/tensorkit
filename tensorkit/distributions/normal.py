from typing import Optional

import numpy as np

from .. import tensor as T
from ..stochastic import StochasticTensor
from .base import Distribution


__all__ = ['Normal']


class Normal(Distribution):

    def __init__(self,
                mean: T.TensorLike,
                std: Optional[T.TensorLike] = None,
                *,
                logstd: Optional[T.TensorLike] = None,
                is_reparameterized: bool = True,
                event_ndims: int = 0,
                check_numerics: Optional[bool] = None,
                random_state: Optional[T.random.RandomState] = None):
        # validate the arguments
        if (std is None and logstd is None) or \
                (std is not None and logstd is not None):
            raise ValueError('One and only one of `std` and `logstd` should '
                             'be specified.')

        mean = T.as_tensor(mean)
        if std is not None:
            std = T.as_tensor(std)
        if logstd is not None:
            logstd = T.as_tensor(logstd)

        stdx = std if std is not None else logstd
        dtype = T.dtype(mean)
        if T.dtype(stdx) != dtype:
            raise ValueError(f'dtype mismatch between mean and std/logstd: '
                             f'{dtype} vs {T.dtype(stdx)}')

        mean_shape = T.shape(mean)
        stdx_shape = T.shape(stdx)
        param_shape = T.broadcast_shape(mean_shape, stdx_shape)

        is_reparameterized = bool(is_reparameterized)

        event_ndims = int(event_ndims)
        if event_ndims < 0:
            raise ValueError(f'event_ndims < 0 is invalid: got {event_ndims}')
        if event_ndims > len(param_shape):
            raise ValueError(f'event_ndims > len(param_shape) is invalid: '
                             f'event_ndims {event_ndims} vs param_shape '
                             f'{param_shape}')

        if event_ndims > 0:
            batch_shape = param_shape[:-event_ndims]
            event_shape = param_shape[-event_ndims:]
        else:
            batch_shape = param_shape
            event_shape = ()

        # construct the object
        super().__init__(
            dtype=dtype,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            batch_shape=batch_shape,
            event_shape=event_shape,
            min_event_ndims=0,
            check_numerics=check_numerics,
            random_state=random_state,
        )
        self._param_shape = param_shape

        # validate mean, std and logstd
        self._mean = self._maybe_check_numerics('mean', mean)
        if std is not None:
            self._std = self._maybe_check_numerics('std', std)
            self._logstd = self._maybe_check_numerics('logstd', T.exp(std))
        else:
            self._logstd = self._maybe_check_numerics('logstd', logstd)
            self._std = self._maybe_check_numerics('std', T.log(logstd))

    @property
    def mean(self) -> T.Tensor:
        return self._mean

    @property
    def std(self) -> T.Tensor:
        return self._std

    @property
    def logstd(self) -> T.Tensor:
        return self._logstd

    def sample(self,
               n_samples: Optional[int] = None,
               group_ndims: int = 0,
               is_reparameterized: Optional[bool] = None,
               compute_prob: Optional[bool] = None) -> StochasticTensor:
        # validate arguments
        sample_shape = ((n_samples,) if n_samples else ()) + self._param_shape

        if is_reparameterized is None:
            is_reparameterized = self.is_reparamterized
        else:
            is_reparameterized = bool(is_reparameterized)

        # generate the samples
        samples = T.random.randn(shape=sample_shape, dtype=self.dtype,
                                 random_state=self.random_state)
        samples = samples * self.std + self.mean
        if not is_reparameterized:
            samples = T.stop_gradient(samples)

        # compose the stochastic tensor object
        t = StochasticTensor(
            distribution=self,
            tensor=samples,
            n_samples=n_samples,
            group_ndims=group_ndims,
            is_reparameterized=is_reparameterized,
        )
        if compute_prob is not False:
            _ = t.log_prob()

        return t

    def log_prob(self, given: T.TensorLike, group_ndims: int = 0) -> T.Tensor:
        event_ndims = self._add_to_event_ndims(group_ndims)

        c = T.as_tensor(-0.5 * np.log(2 * np.pi))
        precision = T.exp(-2 * self.logstd)
        precision = self._maybe_check_numerics('precision', precision)
        precision = 0.5 * precision

        ret = c - self.logstd - precision * T.square(given - self.mean)
        if event_ndims > 0:
            ret = T.sum(precision, list(range(-event_ndims, 0)))
        return ret
