from typing import *

from .. import tensor as T
from ..stochastic import StochasticTensor
from ..tensor import Tensor
from ..typing_ import *
from .base import Distribution, check_tensor_arg_types, copy_distribution

__all__ = ['DiscretizedLogistic']


def is_integer_number(n):
    return abs(n - int(n)) < T.EPSILON


class DiscretizedLogistic(Distribution):
    """
    Discretized logistic distribution (Kingma et. al, 2016).

    For discrete value `x` with equal intervals::

        p(x) = sigmoid((x - mean + bin_size * 0.5) / scale) -
            sigmoid((x - mean - bin_size * 0.5) / scale)

    where `delta` is the interval between two possible values of `x`.

    The `min_val` and `max_val` specifies the minimum and maximum possible
    value of `x`.  It should constraint the generated samples, and if
    `biased_edges` is True, then::

        p(x_min) = sigmoid((x_min - mean + bin_size * 0.5) / scale)
        p(x_max) = 1 - sigmoid((x_max - mean - bin_size * 0.5) / scale)
    """

    min_event_ndims = 0

    mean: Tensor
    log_scale: Tensor
    bin_size: float

    min_val: Optional[float]
    """The minimum possible value of samples."""

    max_val: Optional[float]
    """The maximum possible value of samples."""

    biased_edges: bool
    """Whether or not to use biased density for edge values?"""

    discretize_given: bool
    """Whether or not to discretize `given` in `log_prob()` and `prob()`?"""

    discretize_sample: bool
    """Whether or not to discretize the generated samples in :meth:`sample`?"""

    def __init__(self,
                 mean: TensorOrData,
                 log_scale: TensorOrData,
                 bin_size: float,
                 min_val: Optional[float] = None,
                 max_val: Optional[float] = None,
                 biased_edges: bool = True,
                 discretize_given: bool = True,
                 discretize_sample: bool = True,
                 reparameterized: bool = False,
                 event_ndims: int = 0,
                 epsilon: float = T.EPSILON,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`DiscretizedLogistic`.

        Args:
            mean: A Tensor, the `mean`.
            log_scale: A Tensor, the `log(scale)`.
            bin_size: A scalar, the `bin_size`.
            min_val: A scalar, the minimum possible value of `x`.
            max_val: A scalar, the maximum possible value of `x`.
            biased_edges: Whether or not to use bias density for edge values?
                See above.
            discretize_given: Whether or not to discretize `given`
                in :meth:`log_prob` and :meth:`prob`?
            discretize_sample: Whether or not to discretize the
                generated samples in :meth:`sample`?
            reparameterized: Whether or not the samples are reparameterized?
                Can be True only when `discretize_sample` is False.
            event_ndims: The number of dimensions in the samples to be
                considered as an event.
            epsilon: An infinitesimal constant to avoid dividing by zero or
                taking logarithm of zero.
            validate_tensors: Whether or not to check the numerical issues?
                Defaults to ``settings.validate_tensors``.
        """
        # check the arguments
        if reparameterized and discretize_sample:
            raise ValueError('`reparameterized` cannot be True when '
                             '`discretize_sample` is True.')

        mean, log_scale = check_tensor_arg_types(
            ('mean', mean), ('log_scale', log_scale))
        log_scale = T.as_tensor_backend(log_scale, dtype=mean.dtype)
        dtype = T.get_dtype(mean)

        if min_val is not None and max_val is not None:
            if not is_integer_number((max_val - min_val) / bin_size):
                raise ValueError(
                    f'`min_val - max_val` must be multiples of `bin_size`: '
                    f'`max_val - min_val` == {max_val - min_val}, while '
                    f'`bin_size` == {bin_size}'
                )
        elif min_val is not None or max_val is not None:
            raise ValueError('`min_val` and `max_val` must be both None or '
                             'neither None.')

        # infer the batch shape
        try:
            batch_shape = T.broadcast_shape(T.shape(mean), T.shape(log_scale))
        except Exception:
            raise ValueError(
                f'The shape of `mean` and `log_scale` cannot be broadcasted '
                f'against each other: '
                f'mean {T.shape(mean)} vs log_scale {T.shape(log_scale)}.'
            )

        super(DiscretizedLogistic, self).__init__(
            dtype=dtype,
            value_shape=batch_shape,
            continuous=not discretize_sample,
            reparameterized=reparameterized,
            event_ndims=event_ndims,
            validate_tensors=validate_tensors,
        )
        self.mean = mean
        self.log_scale = log_scale
        self.bin_size = bin_size
        self.min_val = min_val
        self.max_val = max_val
        self.biased_edges = bool(biased_edges)
        self.discretize_given = bool(discretize_given)
        self.discretize_sample = bool(discretize_sample)
        self.epsilon = float(epsilon)

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> 'StochasticTensor':
        return StochasticTensor(
            tensor=T.random.discretized_logistic(
                mean=self.mean,
                log_scale=self.log_scale,
                bin_size=self.bin_size,
                min_val=self.min_val,
                max_val=self.max_val,
                discretize=self.discretize_sample,
                reparameterized=self.reparameterized,
                n_samples=n_samples,
                epsilon=self.epsilon,
                validate_tensors=self.validate_tensors,
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
        return T.random.discretized_logistic_log_prob(
            given=given,
            mean=self.mean,
            log_scale=self.log_scale,
            bin_size=self.bin_size,
            min_val=self.min_val,
            max_val=self.max_val,
            biased_edges=self.biased_edges,
            discretize=self.discretize_given,
            group_ndims=reduce_ndims,
            epsilon=self.epsilon,
            validate_tensors=self.validate_tensors,
        )

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=DiscretizedLogistic,
            base=self,
            attrs=(
                'mean', 'log_scale', 'bin_size', 'min_val', 'max_val',
                'biased_edges', 'discretize_given', 'discretize_sample',
                'reparameterized', 'event_ndims', 'epsilon', 'validate_tensors'
            ),
            overrided_params=overrided_params,
        )
