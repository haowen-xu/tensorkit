from typing import *

from .. import tensor as T
from ..stochastic import StochasticTensor
from .base import Distribution
from .categorical import Categorical, OneHotCategorical
from .utils import copy_distribution

__all__ = ['Mixture']


class Mixture(Distribution):
    """
    Mixture distribution.

    Given a categorical distribution, and corresponding component distributions,
    this class derives a mixture distribution, formulated as follows:

    .. math::

        p(x) = \\sum_{k=1}^{K} \\pi(k) p_k(x)

    where :math:`\\pi(k)` is the probability of taking the k-th component,
    derived by the categorical distribution, and :math:`p_k(x)` is the density
    of the k-th component distribution.
    """

    categorical: Categorical
    """The categorical distribution of this mixture."""

    components: List[Distribution]
    """The mixture components of this distribution."""

    _categorical_event_shape_pad: List[int]

    def __init__(self, 
                 categorical: Union[Categorical, OneHotCategorical],
                 components: Sequence[Distribution], 
                 reparameterized: bool = False,
                 event_ndims: Optional[int] = None,
                 validate_tensors: Optional[bool] = None):
        """
        Construct a new :class:`Mixture`.

        Args:
            categorical: The categorical distribution, indicating the
                probabilities of picking each mixture components.
            components: The component distributions of the mixture.
            reparameterized: Whether or not this mixture distribution
                is re-parameterized?  If :obj:`True`, the `components` must
                all be re-parameterized.  The `categorical` will be treated
                as constant, and the mixture samples will be composed by
                `one_hot(categorical samples) * stack([component samples])`,
                such that the gradients can be propagated back directly
                through these samples.  If :obj:`False`, `tf.stop_gradient`
                will be applied on the mixture samples, such that no gradient
                will be propagated back through these samples.
            event_ndims: If specified, override the `event_ndims` of the
                `components`.  Otherwise use the same as the components.
            validate_tensors: Whether or not to check the numerical issues?
                If `validate_tensors == True` for the `categorical` or any of
                the `components`, then this argument defaults to True.
                Otherwise defaults to `settings.validate_tensors`.
        """
        # check the categorical
        if not isinstance(categorical, (Categorical, OneHotCategorical)):
            raise TypeError(
                f'`categorical` is not a categorical distribution: '
                f'got {categorical!r}'
            )
        if categorical.event_ndims != categorical.min_event_ndims:
            raise ValueError(
                f'`categorical.event_ndims` does not equal to '
                f'`categorical.min_event_ndims`: '
                f'got {categorical.__class__.__qualname__} instance, '
                f'with `event_ndims` {categorical.event_ndims}.'
            )
        n_classes = categorical.n_classes

        # `components`
        components = list(components)
        if not components:
            raise ValueError('`components` must not be empty.')
        if len(components) != n_classes:
            raise ValueError(f'`len(components)` != `categorical.n_classes`: '
                             f'{len(components)} vs {categorical.n_classes}.')

        for i, c in enumerate(components):
            if not isinstance(c, Distribution):
                raise TypeError(f'`components[{i}]` is not an instance of '
                                f'`Distribution`: got {c!r}.')
            if reparameterized and not c.reparameterized:
                raise ValueError(
                    f'`reparameterized` is True, but `components[{i}]` '
                    f'is not re-parameterizable: {c!r}.'
                )
            if validate_tensors is None and c.validate_tensors:
                validate_tensors = True

        # attributes of `components`
        for attr in ('dtype', 'continuous', 'event_ndims', 'batch_shape', 'device'):
            c0_val = getattr(components[0], attr)
            for i, c in enumerate(components[1:], 1):
                c_val = getattr(c, attr)
                if c_val != c0_val:
                    raise ValueError(
                        f'`components[{i}].{attr}` != `components[0].{attr}`: '
                        f'{c_val} vs {c0_val}.'
                    )
        dtype = components[0].dtype
        device = components[0].device
        continuous = components[0].continuous
        batch_shape = components[0].batch_shape

        # categorical `batch_shape` and `device` must match the components
        for attr in ('batch_shape', 'device'):
            if getattr(categorical, attr) != getattr(components[0], attr):
                raise ValueError(
                    f'`categorical.{attr}` != the `{attr}` of '
                    f'`components`: {getattr(categorical, attr)} vs '
                    f'{getattr(components[0], attr)}.'
                )

        # infer the `min_event_shape` and `min_event_ndims`
        min_event_shape = components[0].event_shape
        for i, c in enumerate(components[1:], 1):
            if min_event_shape is None:
                min_event_shape = c.event_shape
            elif c.event_shape is not None:
                if c.event_shape != min_event_shape:
                    raise ValueError(
                        f'`components[{i}].event_shape` does not agree with '
                        f'others: {c.event_shape} vs {min_event_shape}.'
                    )

        min_event_ndims = components[0].event_ndims
        max_event_ndims = min_event_ndims + len(batch_shape)

        if min_event_shape is not None:
            value_shape = batch_shape + min_event_shape
        else:
            value_shape = None

        # `event_ndims`
        if event_ndims is None:
            event_ndims = min_event_ndims
        else:
            if event_ndims < min_event_ndims or event_ndims > max_event_ndims:
                raise ValueError(
                    f'`event_ndims` out of range: got {event_ndims}, but '
                    f'the minimum allowed value is {min_event_ndims}, '
                    f'and the maximum allowed value is {max_event_ndims}.'
                )
            batch_shape = batch_shape[:(len(batch_shape) -
                                        (event_ndims - min_event_ndims))]

        super(Mixture, self).__init__(
            dtype=dtype,
            value_shape=value_shape,
            batch_shape=batch_shape,
            continuous=continuous,
            reparameterized=reparameterized,
            event_ndims=event_ndims,
            min_event_ndims=min_event_ndims,
            device=device,
            validate_tensors=validate_tensors,
        )
        self.categorical = categorical.to_indexed()
        self.components = components
        self._categorical_event_shape_pad = [1] * self.min_event_ndims

    @property
    def n_components(self):
        """Get the number of mixture components."""
        return len(self.components)

    def _sample(self,
                n_samples: Optional[int],
                group_ndims: int,
                reduce_ndims: int,
                reparameterized: bool) -> StochasticTensor:
        # slow routine: generate the mixture sample by one_hot * stack([c.sample()])
        # TODO: add fast routine, which uses embedding_lookup to generate the mixture sample
        #       when reparameterized = False
        cat_samples = self.categorical.sample(n_samples).tensor
        cat_samples = T.reshape(
            cat_samples,
            T.shape(cat_samples) + self._categorical_event_shape_pad
        )
        mask = T.one_hot(cat_samples, len(self.components), dtype=self.dtype)
        mask = T.stop_grad(mask)

        # derive the mixture samples
        # TODO: make use of the cached `log_prob` in sampled tensor of each component
        c_samples = [c.sample(n_samples).tensor for c in self.components]
        c_samples = T.stack(c_samples, axis=-1)

        samples = T.reduce_sum_axis(mask * c_samples, axis=-1)

        if not reparameterized:
            samples = T.stop_grad(samples)

        return StochasticTensor(
            tensor=samples,
            distribution=self,
            n_samples=n_samples,
            group_ndims=group_ndims,
            reparameterized=reparameterized
        )

    def _log_prob(self,
                  given: T.Tensor,
                  group_ndims: int,
                  reduce_ndims: int) -> T.Tensor:
        cat_log_prob = T.nn.log_softmax(self.categorical.logits, axis=-1)
        c_log_prob = T.stack(
            [c.log_prob(given) for c in self.components],
            axis=-1
        )
        log_prob = T.log_sum_exp_axis(cat_log_prob + c_log_prob, axis=-1)
        if reduce_ndims > 0:
            log_prob = T.reduce_sum(log_prob, axis=T.int_range(-reduce_ndims, 0))
        return log_prob

    def copy(self, **overrided_params):
        return copy_distribution(
            cls=Mixture,
            base=self,
            attrs=('categorical', 'components', 'reparameterized',
                   'event_ndims', 'validate_tensors'),
            overrided_params=overrided_params,
        )
