import warnings
from typing import *

from frozendict import frozendict

from .distributions import Distribution
from .stochastic import StochasticTensor
from .tensor import *

__all__ = ['BayesianNet']

ModelBuilderFunctionType = Callable[..., 'BayesianNet']


class BayesianNet(object):

    def __init__(self, observed: Mapping[str, Tensor] = None):
        def check_name(s):
            if not isinstance(s, str):
                raise TypeError(f'name must be a str: got {s!r}')
            return s

        super(BayesianNet, self).__init__()
        self._observed = frozendict([
            (check_name(name), tensor)
            for name, tensor in (observed.items() if observed else ())
        ])
        self._stochastic_tensors = {}

    @property
    def observed(self) -> Mapping[str, Tensor]:
        return self._observed

    def add(self,
            name: str,
            distribution: Distribution,
            n_samples: Optional[int] = None,
            group_ndims: int = 0,
            is_reparameterized: Optional[bool] = None) -> StochasticTensor:
        if not isinstance(name, str):
            raise TypeError('`name` must be a str')
        if name in self._stochastic_tensors:
            raise ValueError(f'Stochastic tensor {name!r} already exists.')

        if group_ndims != 0:
            warnings.warn(
                f'`group_ndims != 0` is not a recommended practice. '
                f'Consider setting `event_ndims` on the distribution instance '
                f'{distribution}.')

        if name in self._observed:
            ob_tensor = self._observed[name]
            if isinstance(ob_tensor, StochasticTensor):
                if is_reparameterized and not ob_tensor.is_reparameterized:
                    raise ValueError(
                        f'`is_reparameterized` is True, but the observation '
                        f'for stochastic tensor {name!r} is not '
                        f're-parameterized: got observation {ob_tensor}'
                    )
                if is_reparameterized is None:
                    is_reparameterized = ob_tensor.is_reparameterized

            if not is_reparameterized:
                ob_tensor = detach(ob_tensor)

            t = StochasticTensor(
                distribution=distribution,
                tensor=ob_tensor,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
            )
        else:
            t = distribution.sample(
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized,
            )
            assert(isinstance(t, StochasticTensor))

        self._stochastic_tensors[name] = t
        return t

    def get(self, name: str) -> Optional[StochasticTensor]:
        return self._stochastic_tensors.get(name)

    def __getitem__(self, name) -> StochasticTensor:
        return self._stochastic_tensors[name]

    def __contains__(self, name) -> bool:
        return name in self._stochastic_tensors

    def __iter__(self) -> Iterator[str]:
        return iter(self._stochastic_tensors)

    def outputs(self, names: Iterable[str]) -> List[Tensor]:
        return [self._stochastic_tensors[n].tensor for n in names]

    def output(self, name: str) -> Tensor:
        return self.outputs((name,))[0]

    def local_log_probs(self, names: Iterable[str]) -> List[Tensor]:
        ret = []
        for name in names:
            ret.append(self._stochastic_tensors[name].log_prob())
        return ret

    def local_log_prob(self, name: str):
        return self.local_log_probs((name,))[0]

    def query(self, names: Iterable[str]) -> List[Tuple[Tensor, Tensor]]:
        names = tuple(names)
        return list(zip(self.outputs(names), self.local_log_probs(names)))

    def chain(self,
              model_builder: ModelBuilderFunctionType,
              latent_names: Optional[Iterable[str]] = None,
              latent_axes: Optional[List[int]] = None,
              observed: Mapping[str, Tensor] = None,
              **kwargs):
        from .variational.chain import VariationalChain

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
                              f'know what you are doing.')

        # build the model and its log-joint
        model_and_log_joint = model_builder(merged_obs, **kwargs)
        if isinstance(model_and_log_joint, tuple):
            model, log_joint = model_and_log_joint
        else:
            model, log_joint = model_and_log_joint, None

        # build the VariationalModelChain
        return VariationalChain(
            q=self,
            p=model,
            log_joint=log_joint,
            latent_names=latent_names,
            latent_axes=latent_axes,
        )
