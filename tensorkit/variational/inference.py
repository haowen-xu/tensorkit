import weakref
from typing import *

from .. import tensor as T
from .estimators import *
from .evaluation import *
from .objectives import *

__all__ = [
    'VariationalInference',
    'VariationalLowerBounds',
    'VariationalTrainingObjectives',
    'VariationalEvaluation'
]


class VariationalInference(object):

    __slots__ = ('latent_log_joint', 'log_joint', 'axis')

    log_joint: T.Tensor
    """Joint log-probability or log-density of the generative net."""

    latent_log_joint: T.Tensor
    """Joint log-probability or log-density of the latent variables."""

    axis: Optional[List[int]]
    """
    The axis to be considered as the sampling dimensions of latent variables.
    The specified axis will be summed up in the variational lower-bounds or
    training objectives.  If :obj:`None`, no dimensions will be reduced.
    """

    def __init__(self,
                 log_joint: T.Tensor,
                 latent_log_joint: T.Tensor,
                 axis: Optional[List[int]] = None):
        """
        Construct a new :class:`VariationalInference` instance.

        Args:
            log_joint: The log-joint of model.
            latent_log_joint: The log-joint of latent variables from the
                variational net.
            axis: The axis to be considered as the sampling dimensions
                of latent variables.  The specified axis will be summed up in
                the variational lower-bounds or training objectives.
                Defaults to :obj:`None`, no axis will be reduced.
        """
        if axis is not None:
            axis = list(map(int, axis))
        self.log_joint = log_joint
        self.latent_log_joint = latent_log_joint
        self.axis = axis

    @property
    def lower_bound(self) -> 'VariationalLowerBounds':
        """The factory for variational lower-bounds."""
        return VariationalLowerBounds(self)

    @property
    def training(self) -> 'VariationalTrainingObjectives':
        """The factory for variational training objectives."""
        return VariationalTrainingObjectives(self)

    @property
    def evaluation(self) -> 'VariationalEvaluation':
        """The factory for evaluation outputs."""
        return VariationalEvaluation(self)


class _VariationalFactory(object):

    __slots__ = ('vi',)

    vi: 'VariationalInference'

    def __init__(self, vi: VariationalInference):
        self.vi = vi


class VariationalLowerBounds(_VariationalFactory):
    """Factory for variational lower-bounds."""

    def elbo(self,
             keepdims: bool = False,
             reduction: str = 'none',  # {'sum', 'mean' or 'none'}
             ) -> T.Tensor:
        """
        Get the evidence lower-bound.

        Returns:
            The evidence lower-bound.

        See Also:
            :func:`tensorkit.variational.elbo_objective`
        """
        vi = self.vi
        return elbo_objective(
            log_joint=vi.log_joint,
            latent_log_joint=vi.latent_log_joint,
            axis=vi.axis,
            keepdims=keepdims,
            reduction=reduction,
        )

    def monte_carlo_objective(self,
                              keepdims: bool = False,
                              reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                              ) -> T.Tensor:
        """
        Get the importance weighted lower-bound (Monte Carlo objective).

        Returns:
            The per-data importance weighted lower-bound.

        See Also:
            :func:`tensorkit.variational.monte_carlo_objective`
        """
        vi = self.vi
        return monte_carlo_objective(
            log_joint=vi.log_joint,
            latent_log_joint=vi.latent_log_joint,
            axis=vi.axis,
            keepdims=keepdims,
            reduction=reduction,
        )

    importance_weighted_objective = monte_carlo_objective


class VariationalTrainingObjectives(_VariationalFactory):
    """Factory for variational training objectives."""

    def sgvb(self,
             keepdims: bool = False,
             reduction: str = 'none',  # {'sum', 'mean' or 'none'}
             ) -> T.Tensor:
        """
        Get the SGVB training objective.

        Returns:
            The per-data SGVB training objective, which is the negative of ELBO.

        See Also:
            :func:`tensorkit.variational.sgvb_estimator`
        """
        vi = self.vi
        return sgvb_estimator(
            # -(log p(x,z) - log q(z|x))
            values=vi.latent_log_joint - vi.log_joint,
            axis=vi.axis,
            keepdims=keepdims,
            reduction=reduction,
        )

    def iwae(self,
             keepdims: bool = False,
             reduction: str = 'none',  # {'sum', 'mean' or 'none'}
             ) -> T.Tensor:
        """
        Get the SGVB training objective for importance weighted objective.

        Returns:
            The per-data SGVB training objective for importance weighted
            objective.

        See Also:
            :func:`tensorkit.variational.iwae_estimator`
        """
        vi = self.vi
        return iwae_estimator(
            log_values=vi.log_joint - vi.latent_log_joint,
            axis=vi.axis,
            keepdims=keepdims,
            reduction=reduction,
            negative=True
        )


class VariationalEvaluation(_VariationalFactory):
    """Factory for variational evaluation outputs."""

    def importance_sampling_log_likelihood(self,
                                           keepdims: bool = False,
                                           reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                                           ) -> T.Tensor:
        """
        Compute :math:`log p(x)` by importance sampling.

        Returns:
            The per-data :math:`log p(x)`.

        See Also:
            :func:`tensorkit.variational.importance_sampling_log_likelihood`
        """
        vi = self.vi
        return importance_sampling_log_likelihood(
            log_joint=vi.log_joint,
            latent_log_joint=vi.latent_log_joint,
            axis=vi.axis,
            keepdims=keepdims,
            reduction=reduction,
        )

    is_loglikelihood = importance_sampling_log_likelihood
    """Short-cut for :meth:`importance_sampling_log_likelihood`."""
