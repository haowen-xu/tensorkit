import weakref
from typing import *

from .. import backend as Z
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

    __slots__ = (
        'latent_log_joint', 'log_joint', 'axes', 'lower_bound',
        'training', 'evaluation',
        '__weakref__',  # to support weakref.ref
    )

    log_joint: Z.Tensor
    """Joint log-probability or log-density of the generative net."""

    latent_log_joint: Z.Tensor
    """Joint log-probability or log-density of the latent variables."""

    axes: Optional[List[int]]
    """
    The axes to be considered as the sampling dimensions of latent variables.
    The specified axes will be summed up in the variational lower-bounds or
    training objectives.  If :obj:`None`, no dimensions will be reduced.
    """

    lower_bound: 'VariationalLowerBounds'
    """The factory for variational lower-bounds."""

    training: 'VariationalTrainingObjectives'
    """The factory for variational training objectives."""

    evaluation: 'VariationalEvaluation'
    """The factory for evaluation outputs."""

    def __init__(self,
                 log_joint: Z.Tensor,
                 latent_log_joint: Z.Tensor,
                 axes: Optional[List[int]] = None):
        """
        Construct a new :class:`VariationalInference` instance.

        Args:
            log_joint: The log-joint of model.
            latent_log_joint: The log-joint of latent variables from the
                variational net.
            axes: The axes to be considered as the sampling dimensions
                of latent variables.  The specified axes will be summed up in
                the variational lower-bounds or training objectives.
                Defaults to :obj:`None`, no axes will be reduced.
        """
        if axes is not None:
            axes = list(map(int, axes))
        self.log_joint = log_joint
        self.latent_log_joint = latent_log_joint
        self.axes = axes
        self.lower_bound = VariationalLowerBounds(self)
        self.training = VariationalTrainingObjectives(self)
        self.evaluation = VariationalEvaluation(self)


class VariationalLowerBounds(object):
    """Factory for variational lower-bounds."""

    __slots__ = ('_vi',)

    _vi: weakref.ref

    def __init__(self, vi: VariationalInference):
        self._vi = weakref.ref(vi)

    def elbo(self, keepdims: bool = False) -> Z.Tensor:
        """
        Get the evidence lower-bound.

        Returns:
            The evidence lower-bound.

        See Also:
            :func:`tensorkit.variational.elbo_objective`
        """
        vi: VariationalInference = self._vi()
        return elbo_objective(
            log_joint=vi.log_joint,
            latent_log_joint=vi.latent_log_joint,
            axes=vi.axes,
            keepdims=keepdims,
        )

    def monte_carlo_objective(self, keepdims: bool = False) -> Z.Tensor:
        """
        Get the importance weighted lower-bound (Monte Carlo objective).

        Returns:
            The per-data importance weighted lower-bound.

        See Also:
            :func:`tensorkit.variational.monte_carlo_objective`
        """
        vi: VariationalInference = self._vi()
        return monte_carlo_objective(
            log_joint=vi.log_joint,
            latent_log_joint=vi.latent_log_joint,
            axes=vi.axes,
            keepdims=keepdims,
        )

    importance_weighted_objective = monte_carlo_objective


class VariationalTrainingObjectives(object):
    """Factory for variational training objectives."""

    __slots__ = ('_vi',)

    _vi: weakref.ref

    def __init__(self, vi: VariationalInference):
        self._vi = weakref.ref(vi)

    def sgvb(self, keepdims: bool = False) -> Z.Tensor:
        """
        Get the SGVB training objective.

        Returns:
            The per-data SGVB training objective, which is the negative of ELBO.

        See Also:
            :func:`tensorkit.variational.sgvb_estimator`
        """
        vi: VariationalInference = self._vi()
        return sgvb_estimator(
            # -(log p(x,z) - log q(z|x))
            values=vi.latent_log_joint - vi.log_joint,
            axes=vi.axes,
            keepdims=keepdims,
        )

    def iwae(self, keepdims: bool = False) -> Z.Tensor:
        """
        Get the SGVB training objective for importance weighted objective.

        Returns:
            The per-data SGVB training objective for importance weighted
            objective.

        See Also:
            :func:`tensorkit.variational.iwae_estimator`
        """
        vi: VariationalInference = self._vi()
        return iwae_estimator(
            log_values=vi.log_joint - vi.latent_log_joint,
            axes=vi.axes,
            keepdims=keepdims,
            negative=True
        )


class VariationalEvaluation(object):
    """Factory for variational evaluation outputs."""

    __slots__ = ('_vi',)

    _vi: weakref.ref

    def __init__(self, vi: VariationalInference):
        self._vi = weakref.ref(vi)

    def importance_sampling_log_likelihood(self,
                                           keepdims: bool = False) -> Z.Tensor:
        """
        Compute :math:`log p(x)` by importance sampling.

        Returns:
            The per-data :math:`log p(x)`.

        See Also:
            :func:`tensorkit.variational.importance_sampling_log_likelihood`
        """
        vi: VariationalInference = self._vi()
        return importance_sampling_log_likelihood(
            log_joint=vi.log_joint,
            latent_log_joint=vi.latent_log_joint,
            axes=vi.axes,
            keepdims=keepdims,
        )

    is_loglikelihood = importance_sampling_log_likelihood
    """Short-cut for :meth:`importance_sampling_log_likelihood`."""
