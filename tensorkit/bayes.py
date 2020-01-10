import warnings
from typing import *

from frozendict import frozendict

from . import backend as Z
from .distributions import Distribution
from .stochastic import StochasticTensor

__all__ = ['BayesianNet']

ModelBuilderFunctionType = Callable[..., 'BayesianNet']


class BayesianNet(Mapping[str, StochasticTensor]):
    """
    Bayesian networks.

    :class:`BayesianNet` is a class which helps to construct Bayesian networks
    and to derive the variational lower-bounds.
    """

    __slots__ = ('_observed', '_original_observed', '_stochastic_tensors')

    _observed: Mapping[str, Z.Tensor]
    """The observation tensors."""

    _original_observed: Dict[str, Union[Z.Tensor, StochasticTensor]]
    """The original `observed` dict specified in the constructor."""

    _stochastic_tensors: Mapping[str, StochasticTensor]
    """The stochastic tensors added to this Bayesian net."""

    def __init__(self,
                 observed: Mapping[str, Union[Z.Tensor, StochasticTensor]] = None):
        """
        Construct a new :class:`BayesianNet` instance.

        Args:
            observed: The observations dict, map from names of stochastic
                nodes to their observation tensors.
        """
        if observed:
            self._original_observed = {
                str(name): tensor
                for name, tensor in observed.items()}
        else:
            self._original_observed = {}
        self._observed = frozendict([
            (str(name), (t.tensor if isinstance(t, StochasticTensor) else t))
            for name, t in self._original_observed.items()
        ])
        self._stochastic_tensors: Dict[str, StochasticTensor] = {}

    @property
    def observed(self) -> Mapping[str, Z.Tensor]:
        """Get the observation tensors."""
        return self._observed

    def add(self,
            name: str,
            distribution: Distribution,
            n_samples: Optional[int] = None,
            group_ndims: int = 0,
            reparameterized: Optional[bool] = None) -> StochasticTensor:
        """
        Add a stochastic node to the Bayesian network.

        A :class:`StochasticTensor` will be created for this node.
        If `name` exists in `observed` dict, its value will be used as the
        observation of this node.  Otherwise samples will be taken from
        the specified `distribution`.

        Args:
            name: Name of the stochastic node.
            distribution: Distribution where the samples should be taken from.
            n_samples: Number of samples to take.
                If specified, `n_samples` of samples will be taken, with a
                dedicated sampling dimension ``[n_samples]`` at the front.
                If not specified, just one sample will be taken, without the
                dedicated dimension.
            group_ndims: Number of dimensions to be considered as events group,
                passed to `distribution.log_prob()` when computing the
                log-probability or log-density of the stochastic tensor.
                Defaults to 0.
            reparameterized: Whether or not the constructed stochastic tensor
                should be reparameterized?  Defaults to the `reparameterized`
                property of the `distribution`.

        Returns:
            The constructed stochastic tensor.
        """
        name = str(name)
        if name in self._stochastic_tensors:
            raise ValueError(f'Stochastic tensor {name!r} already exists.')

        if group_ndims != 0:
            warnings.warn(
                f'`group_ndims != 0` is not a recommended practice. '
                f'Consider setting `event_ndims` on the distribution instance '
                f'{distribution}.',
                UserWarning,
            )

        if name in self._original_observed:
            ob_tensor = self._original_observed[name]
            if isinstance(ob_tensor, StochasticTensor):
                if reparameterized and not ob_tensor.reparameterized:
                    raise ValueError(
                        f'`reparameterized` is True, but the observation '
                        f'for stochastic tensor {name!r} is not '
                        f're-parameterized: got observation {ob_tensor}'
                    )
                if reparameterized is None:
                    reparameterized = ob_tensor.reparameterized
                ob_tensor = ob_tensor.tensor
            else:
                if reparameterized is None:
                    reparameterized = distribution.reparameterized

            if not reparameterized:
                ob_tensor = Z.stop_grad(ob_tensor)

            t = StochasticTensor(
                distribution=distribution,
                tensor=ob_tensor,
                n_samples=n_samples,
                group_ndims=group_ndims,
                reparameterized=reparameterized,
            )
        else:
            t = distribution.sample(
                n_samples=n_samples,
                group_ndims=group_ndims,
                reparameterized=reparameterized,
            )

        self._stochastic_tensors[name] = t
        return t

    def get(self, name: str) -> Optional[StochasticTensor]:
        """
        Get the :class:`StochasticTensor` of a stochastic node.

        Args:
            name: Name of the queried stochastic node.

        Returns:
            The :class:`StochasticTensor` of the queried node, or :obj:`None`
            if no node exists with `name`.
        """
        return self._stochastic_tensors.get(name)

    def __getitem__(self, name) -> StochasticTensor:
        """
        Get the :class:`StochasticTensor` of a stochastic node.

        Args:
            name: Name of the queried stochastic node.

        Returns:
            The :class:`StochasticTensor` of the queried node.
        """
        return self._stochastic_tensors[name]

    def __contains__(self, name) -> bool:
        """Test whether or not a stochastic node with `name` exists."""
        return name in self._stochastic_tensors

    def __iter__(self) -> Iterator[str]:
        """Get an iterator of the stochastic node names."""
        return iter(self._stochastic_tensors)

    def __len__(self) -> int:
        return len(self._stochastic_tensors)

    def outputs(self, names: Iterable[str]) -> List[Z.Tensor]:
        """
        Get the outputs of stochastic nodes.
        The output of a stochastic node is its :attr:`StochasticTensor.tensor`.

        Args:
            names: Names of the queried stochastic nodes.

        Returns:
            Outputs of the queried stochastic nodes.
        """
        return [self._stochastic_tensors[n].tensor for n in names]

    def output(self, name: str) -> Z.Tensor:
        """
        Get the output of a stochastic node.
        The output of a stochastic node is its :attr:`StochasticTensor.tensor`.

        Args:
            name: Name of the queried stochastic node.

        Returns:
            Output of the queried stochastic node.
        """
        return self._stochastic_tensors[name].tensor

    def log_probs(self, names: Iterable[str]) -> List[Z.Tensor]:
        """
        Get the log-probability or log-density of stochastic nodes.

        Args:
            names: Names of the queried stochastic nodes.

        Returns:
            Log-probability or log-density of the queried stochastic nodes.
        """
        ret = []
        for name in names:
            ret.append(self._stochastic_tensors[name].log_prob())
        return ret

    def log_prob(self, name: str) -> Z.Tensor:
        """
        Get the log-probability or log-density of a stochastic node.

        Args:
            name: Name of the queried stochastic node.

        Returns:
            Log-probability or log-density of the queried stochastic node.
        """
        return self._stochastic_tensors[name].log_prob()

    def query_pairs(self, names: Iterable[str]
                    ) -> List[Tuple[Z.Tensor, Z.Tensor]]:
        """
        Get the output and log-probability/log-density of stochastic nodes.

        Args:
            names: Names of the queried stochastic nodes.

        Returns:
            List of ``(output, log-prob)`` pairs of the queried stochastic nodes.
        """
        return [
            (self._stochastic_tensors[n].tensor,
             self._stochastic_tensors[n].log_prob())
            for n in names
        ]

    def query_pair(self, name: str) -> Tuple[Z.Tensor, Z.Tensor]:
        """
        Get the output and log-probability/log-density of a stochastic node.

        Args:
            name: Name of the queried stochastic node.

        Returns:
            The ``(output, log-prob)`` pair of the queried stochastic node.
        """
        return (self._stochastic_tensors[name].tensor,
                self._stochastic_tensors[name].log_prob())

    def chain(self,
              net_builder: ModelBuilderFunctionType,
              latent_names: Optional[Iterable[str]] = None,
              latent_axes: Optional[Union[int, List[int]]] = None,
              observed: Mapping[str, Z.Tensor] = None,
              **kwargs
              ) -> 'VariationalChain':
        """
        Treat this :class:`BayesianNet` as a variational net, and build the
        generative net with observations taken from this variational net.

        Args:
            net_builder: Function which builds the generative net.
                It should receive an optional observation dict as its
                first positional argument, e.g.::

                    def p_net(observed: Optional[Mapping[str, Tensor]] = None):
                        ...

            latent_names (Iterable[str]): Names of the nodes to be considered
                as latent variables in this :class:`BayesianNet`.  All these
                variables will be fed into `model_builder` as observed
                variables, overriding the observations in `observed`.
                (default all the variables in this :class:`BayesianNet`)
            latent_axes: The axis or axes to be considered as the sampling
                dimensions of the latent variables.  The specified axes will
                be summed up in the variational lower-bounds or training
                objectives.  Defaults to :obj:`None`, no axes will be reduced.
            observed: The observation dict fed into `net_builder`, as
                the first positional argument.  Defaults to :obj:`None`.
            \\**kwargs: Additional named arguments passed to `net_builder`.

        Returns:
            The variational chain object, which stores this :class:`BayesianNet`
            as variational net in its `q` attribute, and the constructed
            generative net in its `p` attribute.  It also carries a
            :class:`~tensorkit.variational.VariationalInference` object
            for obtaining the variational lower-bounds and training objectives.

        See Also:
            :class:`tensorkit.variational.VariationalChain`
        """
        if latent_axes is not None and not hasattr(latent_axes, '__iter__'):
            latent_axes = [latent_axes]

        # build the observed dict: observed + latent samples
        merged_obs = {}
        # add the user-provided observed dict
        if observed:
            merged_obs.update(observed)
        # add the latent samples
        if latent_names is None:
            latent_names = tuple(self)
        else:
            latent_names = tuple(map(str, latent_names))
        merged_obs.update({n: self[n] for n in latent_names})

        for n in self:
            if n not in latent_names:  # pragma: no cover
                warnings.warn(f'Stochastic tensor {n!r} in {self!r} is not fed '
                              f'into `model_builder` as observed variable when '
                              f'building the variational chain. I assume you '
                              f'know what you are doing.', UserWarning)

        # build the model and its log-joint
        model = net_builder(merged_obs, **kwargs)

        # build the chain
        return VariationalChain(
            p=model,
            q=self,
            latent_names=latent_names,
            latent_axes=latent_axes,
        )


from .variational.chain import VariationalChain
