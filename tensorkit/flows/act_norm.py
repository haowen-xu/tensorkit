from functools import partial
from typing import *

from .. import init, tensor as T
from ..tensor import (Tensor, Module, reshape, shape, int_range,
                      calculate_mean_and_var, assert_finite,
                      float_scalar_like, maximum, log, sqrt)
from ..layers import *
from ..typing_ import *
from .core import *

__all__ = [
    'ActNorm', 'ActNorm1d', 'ActNorm2d', 'ActNorm3d',
]


class ActNorm(FeatureMappingFlow):
    """
    ActNorm proposed by (Kingma & Dhariwal, 2018).

    `y = (x + bias) * scale; log_det = y / scale - bias`

    `bias` and `scale` are initialized such that `y` will have zero mean and
    unit variance for the initial mini-batch of `x`.
    It can be initialized only through the forward pass.  You may need to use
    :meth:`BaseFlow.invert()` to get a inverted flow if you need to initialize
    the parameters via the opposite direction.
    """

    __constants__ = FeatureMappingFlow.__constants__ + (
        'num_features', 'scale_type', 'epsilon',
    )

    num_features: int
    scale: Module
    scale_type: str
    epsilon: float
    initialized: bool

    def __init__(self,
                 num_features: int,
                 axis: int = -1,
                 event_ndims: int = 1,
                 scale: Union[str, ActNormScaleType] = 'exp',
                 initialized: bool = False,
                 dtype: str = T.float_x(),
                 device: Optional[str] = None,
                 epsilon: float = T.EPSILON):
        """
        Construct a new :class:`ActNorm` instance.

        Args:
            num_features: The size of the feature axis.
            scale: One of {"exp", "linear"}.
                If "exp", ``y = (x + bias) * tf.exp(log_scale)``.
                If "linear", ``y = (x + bias) * scale``.
                Defaults to "exp".
            axis: The axis to apply ActNorm.
                Dimensions not in `axis` will be averaged out when computing
                the mean of activations. Default `-1`, the last dimension.
                All items of the `axis` should be covered by `event_ndims`.
            event_ndims: Number of value dimensions in both `x` and `y`.
                `x.ndims - event_ndims == log_det.ndims` and
                `y.ndims - event_ndims == log_det.ndims`.
            initialized: Whether or not the variables have been
                initialized?  Defaults to :obj:`False`, where the first input
                `x` in the forward pass will be used to initialize the variables.
            dtype: Dtype of the parameters.
            device: The device where to place new tensors and variables.
            epsilon: The infinitesimal constant to avoid dividing by zero or
                taking logarithm of zero.
        """
        # validate the arguments
        scale_type = ActNormScaleType(scale)
        epsilon = float(epsilon)

        if scale_type == ActNormScaleType.EXP:
            scale = ExpScale()
            pre_scale_init = partial(init.fill, fill_value=0.)
        elif scale_type == ActNormScaleType.LINEAR:
            scale = LinearScale(epsilon=epsilon)
            pre_scale_init = partial(init.fill, fill_value=1.)
        else:  # pragma: no cover
            raise ValueError(f'Unsupported `scale_type`: {scale_type}')

        device = device or T.current_device()

        # construct the layer
        super().__init__(axis=axis,
                         event_ndims=event_ndims,
                         explicitly_invertible=True)

        self.num_features = num_features
        self.scale = scale
        self.scale_type = scale_type.value
        self.epsilon = epsilon
        self.initialized = initialized

        add_parameter(
            self, 'pre_scale',
            T.variable([num_features], dtype=dtype, initializer=pre_scale_init,
                       device=device),
        )
        add_parameter(
            self, 'bias',
            T.variable([num_features], dtype=dtype, initializer=init.zeros,
                       device=device),
        )

    @T.jit_method
    def set_initialized(self, initialized: bool = True) -> None:
        self.initialized = initialized

    @T.jit_method
    def calculate_bias_and_pre_scale_for_init(self, input: Tensor) -> Tuple[Tensor, Tensor]:
        # PyTorch 1.3.1 bug: cannot mark this method as returning `None`.
        input_rank = T.rank(input)

        if not isinstance(input, Tensor) or input_rank < self.x_event_ndims + 1:
            raise ValueError(
                '`input` is required to be a tensor with at least {} dimensions: '
                'got input shape {}.'.
                format(self.x_event_ndims, shape(input))
            )

        # calculate the axis to reduce
        feature_axis = input_rank + self.axis
        reduce_axis = (
            int_range(0, feature_axis) +
            int_range(feature_axis + 1, input_rank)
        )

        # calculate sample mean and variance
        input_mean, input_var = calculate_mean_and_var(
            input, axis=reduce_axis, unbiased=True)
        input_var = assert_finite(input_var, 'input_var')

        # calculate the initial_value for `bias`
        bias = -input_mean

        # calculate the initial value for `pre_scale`
        epsilon = float_scalar_like(self.epsilon, input_var)
        if self.scale_type == 'exp':
            pre_scale = -0.5 * log(maximum(input_var, epsilon))
        else:
            pre_scale = 1. / sqrt(maximum(input_var, epsilon))

        return bias, pre_scale

    @T.jit_ignore
    def _initialize_act_norm(self, input: Tensor) -> bool:
        bias, pre_scale = self.calculate_bias_and_pre_scale_for_init(input)
        with T.no_grad():
            T.assign(self.bias, bias)
            T.assign(self.pre_scale, pre_scale)
        self.set_initialized(True)
        return False

    @T.jit_method
    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        # initialize the parameters
        if not self.initialized:
            if inverse:
                raise RuntimeError(
                    '`ActNorm` must be initialized with `inverse = False`.')
            # self.initialize_with_input(input)
            self._initialize_act_norm(input)
            self.set_initialized(True)

        # do transformation
        shape_aligned = [self.num_features] + [1] * (-self.axis - 1)
        shift = reshape(self.bias, shape_aligned)
        pre_scale = reshape(self.pre_scale, shape_aligned)

        input = input if inverse else input + shift
        event_ndims = self.x_event_ndims
        input, input_log_det = self.scale(
            input=input,
            pre_scale=pre_scale,
            event_ndims=event_ndims,
            input_log_det=input_log_det,
            compute_log_det=compute_log_det,
            inverse=inverse,
        )
        if inverse:
            input = input - shift

        return input, input_log_det


class ActNormNd(ActNorm):

    def __init__(self,
                 num_features: int,
                 scale: Union[str, ActNormScaleType] = 'exp',
                 initialized: bool = False,
                 dtype: str = T.float_x(),
                 device: Optional[str] = None,
                 epsilon: float = T.EPSILON):
        """
        Construct a new convolutional :class:`ActNorm` instance.

        Args:
            num_features: The size of the feature axis.
            scale: One of {"exp", "linear"}.
                If "exp", ``y = (x + bias) * tf.exp(log_scale)``.
                If "linear", ``y = (x + bias) * scale``.
                Defaults to "exp".
            initialized: Whether or not the variables have been
                initialized?  Defaults to :obj:`False`, where the first input
                `x` in the forward pass will be used to initialize the variables.
            dtype: Dtype of the parameters.
            device: The device where to place new tensors and variables.
            epsilon: The infinitesimal constant to avoid dividing by zero or
                taking logarithm of zero.
        """
        spatial_ndims = self._get_spatial_ndims()
        feature_axis = -1 if T.IS_CHANNEL_LAST else -(spatial_ndims + 1)

        super().__init__(
            num_features=num_features,
            axis=feature_axis,
            event_ndims=spatial_ndims + 1,
            scale=scale,
            initialized=initialized,
            dtype=dtype,
            device=device,
            epsilon=epsilon,
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class ActNorm1d(ActNormNd):
    """1D convolutional ActNorm flow."""

    def _get_spatial_ndims(self) -> int:
        return 1


class ActNorm2d(ActNormNd):
    """2D convolutional ActNorm flow."""

    def _get_spatial_ndims(self) -> int:
        return 2


class ActNorm3d(ActNormNd):
    """3D convolutional ActNorm flow."""

    def _get_spatial_ndims(self) -> int:
        return 3
