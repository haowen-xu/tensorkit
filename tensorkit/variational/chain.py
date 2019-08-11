from typing import *

from ..bayes import BayesianNet
from ..tensor import *
from .inference import VariationalInference

__all__ = ['VariationalChain']


class VariationalChain(object):

    def __init__(self,
                 q: BayesianNet,
                 p: BayesianNet,
                 log_joint: Optional[Tensor] = None,
                 latent_names: Optional[Iterable[str]] = None,
                 latent_axes: Optional[List[int]] = None):
        if latent_names is None:
            latent_names = tuple(q)
        else:
            latent_names = tuple(map(str, latent_names))

        if log_joint is None:
            log_joint = add_n(p.local_log_probs(iter(p)))
        latent_log_prob = add_n(q.local_log_probs(latent_names))

        self._q = q
        self._p = p
        self._log_joint = log_joint
        self._latent_names = latent_names
        self._latent_axes = latent_axes
        self._vi = VariationalInference(
            log_joint=log_joint,
            latent_log_prob=latent_log_prob,
            axes=latent_axes
        )

    @property
    def q(self) -> BayesianNet:
        return self._q

    @property
    def p(self) -> BayesianNet:
        return self._p

    @property
    def log_joint(self) -> Tensor:
        return self._log_joint

    @property
    def latent_names(self) -> Tuple[str, ...]:
        return self._latent_names

    @property
    def latent_axes(self) -> Optional[List[int]]:
        return self._latent_axes

    @property
    def vi(self) -> VariationalInference:
        return self._vi
