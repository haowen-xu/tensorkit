from typing import *

from . import tensor as T

__all__ = ['StochasticTensor']


class StochasticTensor(object):
    """
    Samples or observations of a random variable.

    It stores the sample or observation of the random variable, as well as
    the distribution instance.  Also, :meth:`log_prob()` and :meth:`prob()`,
    and a few other utilities are provided for convenience.
    """

    __slots__ = ('tensor', 'distribution', 'n_samples', 'group_ndims',
                 'reparameterized', 'transform_origin', '_cached_log_prob',
                 '_cached_prob')

    tensor: T.Tensor
    """The sample or observation."""

    distribution: 'Distribution'
    """The distribution instance."""

    n_samples: Optional[int]
    """
    The argument `n_samples` for :meth:`sample()` when producing the samples.
    """

    group_ndims: int
    """The default ``group_ndims`` to compute the log-prob of the samples."""

    reparameterized: bool
    """Whether or not the samples are reparameterized?"""

    transform_origin: Optional['StochasticTensor']
    """
    If `distribution.base_distribution` is not `distribution`, this attribute
    should be the original stochastic tensor from `distribution.base_distribution`,
    from which this stochastic tensor is transformed.
    """

    _cached_log_prob: Optional[T.Tensor]
    _cached_prob: Optional[T.Tensor]

    def __init__(self,
                 tensor: T.Tensor,
                 distribution: 'Distribution',
                 n_samples: Optional[int],
                 group_ndims: int,
                 reparameterized: bool,
                 transform_origin: Optional['StochasticTensor'] = None,
                 log_prob: Optional[T.Tensor] = None):
        """
        Construct a new :class:`StochasticTensor`.

        Args:
            tensor: The samples or observations tensor.
            distribution: The distribution instance.
            n_samples: The argument `n_samples` for :meth:`sample()` when
                producing the samples.
            group_ndims: The default ``group_ndims`` to compute the log-prob
                of the samples.
            reparameterized: Whether or not the samples are reparameterized?
            transform_origin: If `distribution.base_distribution` is not
                `distribution`, this attribute should be the original
                stochastic tensor from `distribution.base_distribution`,
                from which this stochastic tensor is transformed.
            log_prob: The pre-computed log-probability or log-density.
        """
        n_samples = int(n_samples) if n_samples is not None else None
        group_ndims = int(group_ndims)
        reparameterized = bool(reparameterized)

        self.tensor = tensor
        self.distribution = distribution
        self.n_samples = n_samples
        self.group_ndims = group_ndims
        self.reparameterized = reparameterized
        self.transform_origin = transform_origin
        self._cached_log_prob = log_prob
        self._cached_prob = None

    def __repr__(self):
        return f'StochasticTensor({self.tensor!r})'

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    @property
    def continuous(self) -> bool:
        return self.distribution.continuous

    def log_prob(self, group_ndims: Optional[int] = None) -> T.Tensor:
        """
        Compute the log-probability or log-density of the samples.

        Args:
            group_ndims: If specified, overriding the default `group_ndims`.

        Returns:
            The log-probability or log-density.
        """
        if group_ndims is None or group_ndims == self.group_ndims:
            if self._cached_log_prob is None:
                self._cached_log_prob = self.distribution.log_prob(
                    given=self.tensor,
                    group_ndims=self.group_ndims,
                )
            return self._cached_log_prob
        else:
            return self.distribution.log_prob(self.tensor, group_ndims)

    def prob(self, group_ndims: Optional[int] = None) -> T.Tensor:
        """
        Compute the probability or density of the samples.

        Args:
            group_ndims: If specified, overriding the default `group_ndims`.

        Returns:
            The probability or density.
        """
        if group_ndims is None or group_ndims == self.group_ndims:
            if self._cached_prob is None:
                self._cached_prob = T.exp(self.log_prob())
            return self._cached_prob
        else:
            return self.distribution.prob(
                given=self.tensor, group_ndims=group_ndims)


from .distributions.base import Distribution
