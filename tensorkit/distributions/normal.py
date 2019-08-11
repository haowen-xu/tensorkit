from typing import Optional

import numpy as np

from ..stochastic import StochasticTensor
from ..tensor import *
from .base import Distribution

__all__ = ['Normal']


class Normal(Distribution):

    def __init__(self,
                mean: Tensor,
                std: Optional[Tensor] = None,
                *,
                logstd: Optional[Tensor] = None,
                is_reparameterized: bool = True,
                event_ndims: int = 0,
                check_numerics: Optional[bool] = None):
        # validate the arguments
        if (std is None) == (logstd is None):
            raise ValueError('Either `std` or `logstd` must be specified, '
                             'but not both.')

        if std is not None:
            original_std_arg = 'std'
        else:
            original_std_arg = 'logstd'

        stdx = std if std is not None else logstd
        dtype_ = dtype(mean)
        if dtype(stdx) != dtype_:
            raise ValueError(f'dtype mismatch between mean and std/logstd: '
                             f'{dtype_} vs {dtype(stdx)}')

        mean_shape = shape(mean)
        stdx_shape = shape(stdx)
        param_shape = broadcast_shape(mean_shape, stdx_shape)

        # construct the object
        super().__init__(
            dtype=dtype_,
            is_continuous=True,
            is_reparameterized=is_reparameterized,
            value_shape=param_shape,
            event_ndims=event_ndims,
            min_event_ndims=0,
            check_numerics=check_numerics,
        )
        self._param_shape = param_shape
        self._original_std_arg = original_std_arg

        # validate mean, std and logstd
        self._mean = self._maybe_check_numerics('mean', mean)
        if std is not None:
            self._std = self._maybe_check_numerics('std', std)
            self._logstd = None
        else:
            self._logstd = self._maybe_check_numerics('logstd', logstd)
            self._std = None

    @property
    def original_std_arg(self) -> str:
        return self._original_std_arg

    @property
    def mean(self) -> Tensor:
        return self._mean

    @property
    def std(self) -> Tensor:
        if self._std is None:
            logstd = log(self._logstd)
            self._std = self._maybe_check_numerics('std', logstd)
        return self._std

    @property
    def logstd(self) -> Tensor:
        if self._logstd is None:
            std = exp(self._std)
            self._logstd = self._maybe_check_numerics('logstd', std)
        return self._logstd

    def sample(self,
               n_samples: Optional[int] = None,
               group_ndims: int = 0,
               is_reparameterized: Optional[bool] = None,
               compute_prob: Optional[bool] = None) -> StochasticTensor:
        # validate arguments
        sample_shape = ((n_samples,) if n_samples else ()) + self._param_shape
        is_reparameterized = self._check_reparameterized(is_reparameterized)

        # generate the samples
        samples = random.randn(shape=sample_shape, dtype=self.dtype)
        samples = samples * self.std + self.mean
        if not is_reparameterized:
            samples = detach(samples)

        # compose the stochastic tensor object
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

    def log_prob(self, given: Tensor, group_ndims: int = 0) -> Tensor:
        event_ndims = self._add_to_event_ndims(group_ndims)

        c = as_tensor(-0.5 * np.log(2 * np.pi))
        precision = exp(-2 * self.logstd)
        precision = self._maybe_check_numerics('precision', precision)
        precision = 0.5 * precision

        ret = c - self.logstd - precision * square(given - self.mean)
        if event_ndims > 0:
            ret = reduce_sum(precision, list(range(-event_ndims, 0)))
        return ret

    def copy(self, **kwargs):
        return self._copy_helper(
            ('mean', self.original_std_arg, 'is_reparameterized',
             'event_ndims', 'check_numerics', 'random_state'),
            **kwargs
        )
