from typing import *

from .. import tensor as T
from ..tensor import typing as Z
from .estimators import *
from .evaluation import *
from .objectives import *
from .utils import _require_multi_samples

__all__ = [
    'VariationalInference',
    'VariationalLowerBounds',
    'VariationalTrainingObjectives',
    'VariationalEvaluation'
]


class VariationalInference(object):

    def __init__(self,
                 log_joint: Z.TensorLike,
                 latent_log_prob: Z.TensorLike,
                 axis: Optional[Z.AxisOrAxes] = None):
        self._log_joint = T.as_tensor(log_joint)
        self._latent_log_prob = T.as_tensor(latent_log_prob)
        self._axis = axis
        self._lower_bound = VariationalLowerBounds(self)
        self._training = VariationalTrainingObjectives(self)
        self._evaluation = VariationalEvaluation(self)

    @property
    def log_joint(self) -> T.Tensor:
        """Get `log p(x|z) + log p(z)`."""
        return self._log_joint

    @property
    def latent_log_prob(self) -> T.Tensor:
        """Get `log q(z|x)`."""
        return self._latent_log_prob

    @property
    def axis(self) -> Optional[Z.AxisOrAxes]:
        return self._axis

    @property
    def lower_bound(self) -> 'VariationalLowerBounds':
        return self._lower_bound

    @property
    def training(self) -> 'VariationalTrainingObjectives':
        return self._training

    @property
    def evaluation(self) -> 'VariationalEvaluation':
        return self._evaluation


class VariationalLowerBounds(object):

    def __init__(self, vi: VariationalInference):
        self._vi = vi

    def elbo(self, keepdims: bool = False) -> T.Tensor:
        return elbo_objective(
            log_joint=self._vi.log_joint,
            latent_log_prob=self._vi.latent_log_prob,
            axis=self._vi.axis,
            keepdims=keepdims,
        )

    def monte_carlo_objective(self, keepdims: bool = False) -> T.Tensor:
        _require_multi_samples(self._vi.axis, 'monte carlo objective')
        return monte_carlo_objective(
            log_joint=self._vi.log_joint,
            latent_log_prob=self._vi.latent_log_prob,
            axis=self._vi.axis,
            keepdims=keepdims,
        )

    importance_weighted_objective = monte_carlo_objective  # Legacy name


class VariationalTrainingObjectives(object):

    def __init__(self, vi: VariationalInference):
        self._vi = vi

    def sgvb(self, keepdims: bool = False) -> T.Tensor:
        return sgvb_estimator(
            # -(log p(x,z) - log q(z|x))
            values=self._vi.latent_log_prob - self._vi.log_joint,
            axis=self._vi.axis,
            keepdims=keepdims,
        )

    def iwae(self, keepdims: bool = False) -> T.Tensor:
        return iwae_estimator(
            log_values=self._vi.log_joint - self._vi.latent_log_prob,
            axis=self._vi.axis,
            keepdims=keepdims,
            neg_grad=True
        )


class VariationalEvaluation(object):

    def __init__(self, vi: VariationalInference):
        self._vi = vi

    def importance_sampling_log_likelihood(self,
                                           keepdims: bool = False) -> T.Tensor:
        _require_multi_samples(
            self._vi.axis, 'importance sampling log-likelihood')
        return importance_sampling_log_likelihood(
            log_joint=self._vi.log_joint,
            latent_log_prob=self._vi.latent_log_prob,
            axis=self._vi.axis,
            keepdims=keepdims,
        )

    is_loglikelihood = importance_sampling_log_likelihood
    """Short-cut for :meth:`importance_sampling_log_likelihood`."""
