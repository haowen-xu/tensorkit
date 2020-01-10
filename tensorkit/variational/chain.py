from typing import *

from .. import backend as Z
from .inference import VariationalInference

__all__ = ['VariationalChain']


class VariationalChain(object):
    """
    Chain of a pair of variational and generative net for variational inference.

    In the context of variational inference, it is a common usage for chaining
    the variational net and the generative net, by feeding the samples of latent
    variables from the variational net as the observations of the generative net.
    :class:`VariationalChain` stores the :class:`BayesianNet` instances of
    the variational and the generative net, and constructs a
    :class:`VariationalInference` object for this chain.

    See Also:
        :meth:`tensorkit.bayes.BayesianNet.variational_chain`
    """

    __slots__ = ('p', 'q', 'latent_names', 'latent_axes', '_log_joint',
                 '_latent_log_joint', '_vi')

    p: 'BayesianNet'
    """The generative net, which is usually written as `p(x,z)`."""

    q: 'BayesianNet'
    """The variational net, which is usually written as `q(z|x)`."""

    latent_names: List[str]
    """The names of the latent variables."""

    latent_axes: Optional[List[int]]
    """The sampling dimensions of the latent variables."""

    _log_joint: Optional[Z.Tensor]
    _latent_log_joint: Optional[Z.Tensor]
    _vi: Optional[VariationalInference]

    def __init__(self,
                 p: 'BayesianNet',
                 q: 'BayesianNet',
                 latent_names: Optional[Sequence[str]] = None,
                 latent_axes: Optional[List[int]] = None,
                 log_joint: Optional[Z.Tensor] = None,
                 latent_log_joint: Optional[Z.Tensor] = None):
        """
        Construct a new :class:`VariationalChain` instance.

        Args:
            p: The generative net.
            q: The variational net.
            latent_names: Names of the latent variables.  If not specified,
                all random variables in `q` net will be considered as latent.
            latent_axes: The axes to be considered as the sampling dimensions
                of latent variables.  The specified axes will be summed up in
                the variational lower-bounds or training objectives.
                Defaults to :obj:`None`, no axes will be reduced.
            log_joint: Pre-computed joint log-probability or log-density of
                the generative net, i.e., ``sum(p.log_probs(list(p)))`.
            latent_log_joint: Pre-computed joint log-probability or log-density of
                the latent variables from the variational net, i.e.,
                ``sum(q.log_probs(latent_names))`.
        """
        latent_names = (list(q) if latent_names is None
                        else list(map(str, latent_names)))
        if latent_axes is not None:
            latent_axes = list(map(int, latent_axes))

        self.p = p
        self.q = q
        self.latent_names = latent_names
        self.latent_axes = latent_axes
        self._log_joint = log_joint
        self._latent_log_joint = latent_log_joint
        self._vi = None  # constructed on demand later

    @property
    def log_joint(self) -> Z.Tensor:
        """The joint log-probability or log-density of the generative net."""
        if self._log_joint is None:
            self._log_joint = Z.add_n(self.p.log_probs(self.p))
        return self._log_joint

    @property
    def latent_log_joint(self) -> Z.Tensor:
        """
        The joint log-probability or log-density of the latent variables
        from the variational net.
        """
        if self._latent_log_joint is None:
            self._latent_log_joint = Z.add_n(self.q.log_probs(self.latent_names))
        return self._latent_log_joint

    @property
    def vi(self) -> VariationalInference:
        if self._vi is None:
            self._vi = VariationalInference(
                log_joint=self.log_joint,
                latent_log_joint=self.latent_log_joint,
                axes=self.latent_axes,
            )
        return self._vi


from ..bayes import BayesianNet
