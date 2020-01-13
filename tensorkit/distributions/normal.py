from typing import *

from .. import tensor as T
from ..stochastic import StochasticTensor
from ..typing_ import *
from .base import Distribution
from .utils import copy_distribution, check_tensor_arg_types

__all__ = [
    'UnitNormal',
    'Normal', 'TruncatedNormal',
]


class UnitNormal(Distribution):
    """
    Unit normal distribution, i.e., :math:`\\mathcal{N}(0, 1)`.

    Use ``UnitNormal(shape)`` is mostly equivalent to using
    ``Normal(zeros(shape), ones(shape))``, except that the distribution
    derived from this class should run a little faster.
    """

    continuous = True
    reparameterized = True
    min_event_ndims = 0

    _mean: Optional[T.Tensor] = None
    _std: Optional[T.Tensor] = None
    _logstd: Optional[T.Tensor] = None

    def __init__(self,
                 shape: List[int],
                 dtype: str = T.float_x(),
                 reparameterized: bool = True,
                 event_ndims: int = 0,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`UnitNormal` distribution.

        Args:
            shape: Shape of the unit normal distribution.
            dtype: Dtype of the samples.
            reparameterized: Whether the distribution should be reparameterized?
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        super().__init__(
            dtype=dtype,
            value_shape=shape,
            reparameterized=reparameterized,
            event_ndims=event_ndims,
            validate_tensors=validate_tensors,
        )

    @property
    def mean(self) -> T.Tensor:
        """The mean of the normal distribution."""
        if self._mean is None:
            self._mean = T.zeros(self.value_shape, self.dtype)
        return self._mean

    @property
    def std(self) -> T.Tensor:
        """The standard deviation (std) of the normal distribution."""
        if self._std is None:
            self._std = T.ones(self.value_shape, self.dtype)
        return self._std

    @property
    def logstd(self) -> T.Tensor:
        """The log-std of the normal distribution."""
        if self._logstd is None:
            self._logstd = T.zeros(self.value_shape, self.dtype)
        return self._logstd

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        return StochasticTensor(
            tensor=T.random.randn(
                shape=([n_samples] + self.value_shape if n_samples is not None
                       else self.value_shape),
                dtype=self.dtype,
            ),
            distribution=self,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return T.random.randn_log_pdf(given=given, group_ndims=reduce_ndims)

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=UnitNormal,
            base=self,
            attrs=(('shape', 'value_shape'), 'dtype', 'reparameterized',
                   'event_ndims', 'validate_tensors'),
            cached_attrs=('mean', 'std', 'logstd'),
            compute_deps={
                'mean': ('dtype', 'value_shape'),
                'std': ('dtype', 'value_shape'),
                'logstd': ('dtype', 'value_shape'),
            },
            overrided_params=overrided_params,
        )


class BaseNormal(Distribution):
    """Base class for univariate normal-like distributions."""

    continuous = True
    reparameterized = True
    min_event_ndims = 0

    _extra_args: Tuple[str] = ()
    """
    Extra arguments for the constructor, in addition to the arguments defined
    in the constructor of :class:`BaseNormal`.
    """

    mean: T.Tensor
    """
    The mean of the normal distribution.

    Note this might not be the true mean of the distribution.
    It only serves as the argument for the distribution, as if it was a
    standard normal distribution.
    """

    _mutual_params: Dict[str, Any]
    """Dict that stores the original `std` or `logstd` constructor argument."""

    def __init__(self,
                 mean: T.Tensor,
                 std: Optional[TensorOrData] = None,
                 *,
                 logstd: Optional[TensorOrData] = None,
                 reparameterized: bool = True,
                 event_ndims: int = 0,
                 validate_tensors: Optional[bool] = None):
        # validate the arguments
        mean, (std, logstd) = check_tensor_arg_types(
            ('mean', mean), [('std', std), ('logstd', logstd)])
        if std is not None:
            mutual_params = {'std': std}
            stdx = std
        else:
            mutual_params = {'logstd': logstd}
            stdx = logstd
        mean_shape = T.shape(mean)
        dtype = T.get_dtype(mean)
        stdx_shape = T.shape(stdx)
        value_shape = T.broadcast_shape(mean_shape, stdx_shape)

        # construct the object
        super().__init__(
            dtype=dtype,
            reparameterized=reparameterized,
            value_shape=value_shape,
            event_ndims=event_ndims,
            validate_tensors=validate_tensors,
        )
        for k, v in mutual_params.items():
            mutual_params[k] = self._assert_finite(v, k)
        self._mutual_params = mutual_params
        self.mean = self._assert_finite(mean, 'mean')
        self._std = mutual_params.get('std', None)
        self._logstd = mutual_params.get('logstd', None)

    @property
    def std(self) -> T.Tensor:
        """
        The standard deviation (std) of the normal distribution.

        Note this might not be the true std of the distribution.
        It only serves as the argument for the distribution, as if it was a
        standard normal distribution.
        """
        if self._std is None:
            self._std = self._assert_finite(T.exp(self._logstd), 'std')
        return self._std

    @property
    def logstd(self) -> T.Tensor:
        """
        The log-std of the normal distribution.

        Note this might not be the true logstd of the distribution.
        It only serves as the argument for the distribution, as if it was a
        standard normal distribution.
        """
        if self._logstd is None:
            self._logstd = self._assert_finite(T.log(self._std), 'logstd')
        return self._logstd

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=self.__class__,
            base=self,
            attrs=(
                ('mean', 'reparameterized', 'event_ndims', 'validate_tensors') +
                self._extra_args
            ),
            mutual_attrs=(('std', 'logstd'),),
            original_mutual_params=self._mutual_params,
            overrided_params=overrided_params,
        )


class Normal(BaseNormal):
    """Univariate normal distribution."""

    def __init__(self,
                 mean: T.Tensor,
                 std: Optional[TensorOrData] = None,
                 *,
                 logstd: Optional[TensorOrData] = None,
                 reparameterized: bool = True,
                 event_ndims: int = 0,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`Normal` distribution instance.

        Args:
            mean: The mean of the normal distribution.
            std: The standard deviation (std) of the normal distribution.
            logstd: The log-std of the normal distribution.
            reparameterized: Whether the distribution should be reparameterized?
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        super().__init__(
            mean=mean, std=std, logstd=logstd, reparameterized=reparameterized,
            event_ndims=event_ndims, validate_tensors=validate_tensors,
        )

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        return StochasticTensor(
            tensor=T.random.normal(
                mean=self.mean,
                std=self.std,
                n_samples=n_samples,
                reparameterized=reparameterized,
            ),
            distribution=self,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return T.random.normal_log_pdf(
            given=given,
            mean=self.mean,
            logstd=self.logstd,
            group_ndims=reduce_ndims,
            validate_tensors=self.validate_tensors,
        )


class TruncatedNormal(BaseNormal):
    """
    Univariate truncated normal distribution.

    The density outside of ``[low * std + mean, high * std + mean)`` will be
    always zero, i.e., the generated samples will be ensured to reside within
    this range.
    """

    low: Optional[float]
    high: Optional[float]
    epsilon: float
    log_zero: float

    _extra_args = ('low', 'high', 'epsilon', 'log_zero')

    def __init__(self,
                 mean: T.Tensor,
                 std: Optional[TensorOrData] = None,
                 *,
                 logstd: Optional[TensorOrData] = None,
                 low: Optional[float] = None,
                 high: Optional[float] = None,
                 reparameterized: bool = True,
                 event_ndims: int = 0,
                 epsilon: float = 1e-7,
                 log_zero: float = T.random.LOG_ZERO_VALUE,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`TruncatedNormal` distribution instance.

        Args:
            mean: The mean, as if this is a standard normal distribution.
            std: The standard deviation (std) of the standard normal distribution.
            logstd: The log-std of the standard normal distribution.
            low: The lower-bound of the distribution, see above.
            high: The upper-bound of the distribution, see above.
            reparameterized: Whether the distribution should be reparameterized?
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            epsilon: The infinitesimal constant, used for generating samples.
            log_zero: The value to represent ``log(0)`` in the result of
                :meth:`log_prob()`, instead of using ``-math.inf``, to avoid
                potential numerical issues.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        if low is not None and high is not None:
            if low >= high:
                raise ValueError(f'`low` < `high` does not hold: low == {low}, '
                                 f'and high == {high}.')
        super().__init__(
            mean=mean, std=std, logstd=logstd, reparameterized=reparameterized,
            event_ndims=event_ndims, validate_tensors=validate_tensors,
        )
        self.low = low
        self.high = high
        self.epsilon = epsilon
        self.log_zero = log_zero

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        return StochasticTensor(
            tensor=T.random.truncated_normal(
                mean=self.mean,
                std=self.std,
                low=self.low,
                high=self.high,
                n_samples=n_samples,
                reparameterized=reparameterized,
                epsilon=self.epsilon,
            ),
            distribution=self,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized,
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        return T.random.truncated_normal_log_pdf(
            given=given,
            mean=self.mean,
            std=self.std,
            logstd=self.logstd,
            low=self.low,
            high=self.high,
            group_ndims=reduce_ndims,
            log_zero=self.log_zero,
            validate_tensors=self.validate_tensors,
        )
