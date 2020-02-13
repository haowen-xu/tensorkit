from typing import *

import numpy as np
import torch
from scipy import linalg as la

from ...settings_ import settings
from ...typing_ import TensorInitArgType
from . import init
from .core import *
from .layers import *
from .linalg import *
from .nn import *

__all__ = [
    'BaseFlow', 'FeatureMappingFlow',
    'InverseFlow', 'SequentialFlow',
    'LooseInvertibleMatrix', 'StrictInvertibleMatrix',
    'InvertibleDense', 'InvertibleConv1d', 'InvertibleConv2d',
    'InvertibleConv3d',
    'BaseScale', 'SigmoidScale', 'ExpScale', 'LinearScale',
]


# ---- base flow classes ----
class BaseFlow(BaseLayer):
    """
    Base class for normalizing flows.

    A normalizing flow transforms a random variable `x` into `y` by an
    (implicitly) invertible mapping :math:`y = f(x)`, whose Jaccobian matrix
    determinant :math:`\\det \\frac{\\partial f(x)}{\\partial x} \\neq 0`, thus
    can derive :math:`\\log p(y)` from given :math:`\\log p(x)`.
    """

    __constants__ = ('x_event_ndims', 'y_event_ndims', 'explicitly_invertible')

    x_event_ndims: int
    """Number of event dimensions in `x`."""

    y_event_ndims: int
    """Number of event dimensions in `y`."""

    explicitly_invertible: bool
    """
    Whether or not this flow is explicitly invertible?

    If a flow is not explicitly invertible, then it only supports to
    transform `x` into `y`, and corresponding :math:`\\log p(x)` into
    :math:`\\log p(y)`.  It cannot compute :math:`\\log p(y)` directly
    without knowing `x`, nor can it transform `x` back into `y`.
    """

    def __init__(self,
                 x_event_ndims: int,
                 y_event_ndims: int,
                 explicitly_invertible: bool):
        super().__init__()

        self.x_event_ndims = int(x_event_ndims)
        self.y_event_ndims = int(y_event_ndims)
        self.explicitly_invertible = bool(explicitly_invertible)

    def invert(self) -> 'BaseFlow':
        """
        Get the inverse flow from this flow.

        Specifying `inverse = True` when calling the inverse flow will be
        interpreted as having `inverse = False` in the original flow, and
        vise versa.

        If the current flow requires to be initialized by calling it
        with `inverse = False`, then the inversed flow will require to be
        initialized by calling it with `inverse = True`, and vise versa.

        Returns:
            The inverse flow.
        """
        return InverseFlow(self)

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError()

    def forward(self,
                input: Tensor,
                input_log_det: Optional[Tensor] = None,
                inverse: bool = False,
                compute_log_det: bool = True
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Transform `x` into `y` and compute the log-determinant of `f` at `x`
        (if `inverse` is False); or transform `y` into `x` and compute the
        log-determinant of `f^{-1}` at `y` (if `inverse` is True).

        Args:
            input: `x` (if `inverse` is False) or `y` (if `inverse` is True).
            input_log_det: The log-determinant of the previous layer.
                Will add the log-determinant of this layer to `input_log_det`,
                to obtain the output log-determinant.  If no previous layer,
                will start from zero log-det.
            inverse: See above.
            compute_log_det: Whether or not to compute the log-determinant?

        Returns:
            The transformed tensor, and the summed log-determinant of
            the previous flow layer and this layer.
        """
        if inverse:
            event_ndims = self.y_event_ndims
        else:
            event_ndims = self.x_event_ndims

        if input.dim() < event_ndims:
            raise ValueError(
                '`input` is required to be at least {}d, but the input shape '
                'is {}.'.format(event_ndims, shape(input))
            )

        input_shape = shape(input)
        log_det_shape = input_shape[: len(input_shape) - event_ndims]

        if input_log_det is not None:
            if shape(input_log_det) != log_det_shape:
                raise ValueError(
                    'The shape of `input_log_det` is not expected: '
                    'expected to be {}, but got {}.'.
                    format(log_det_shape, shape(input_log_det))
                )

        # compute the transformed output and log-det
        output, output_log_det = self._forward(
            input, input_log_det, inverse, compute_log_det)

        if output_log_det is not None:
            if output_log_det.dim() < len(log_det_shape):
                output_log_det = broadcast_to(output_log_det, log_det_shape)

            if shape(output_log_det) != log_det_shape:
                raise ValueError(
                    'The shape of `output_log_det` is not expected: '
                    'expected to be {}, but got {}.'.
                    format(log_det_shape, shape(output_log_det))
                )

        return output, output_log_det


class FeatureMappingFlow(BaseFlow):
    """Base class for flows mapping input features to output features."""

    __constants__ = BaseFlow.__constants__ + ('axis',)

    axis: int
    """The feature axis (negative index)."""

    def __init__(self,
                 axis: int,
                 event_ndims: int,
                 explicitly_invertible: bool):
        """
        Construct a new :class:`FeatureMappingFlow`.

        Args:
            axis: The feature axis, on which to apply the transformation.
                It must be a negative integer, and included in the
                event dimensions.
            event_ndims: Number of event dimensions in both `x` and `y`.
                `x.ndims - event_ndims == log_det.ndims` and
                `y.ndims - event_ndims == log_det.ndims`.
            explicitly_invertible: Whether or not this flow is explicitly
                invertible?
        """
        # check the arguments
        axis = int(axis)
        event_ndims = int(event_ndims)

        if event_ndims < 1:
            raise ValueError(f'`event_ndims` must be at least 1: '
                             f'got {event_ndims}')

        if axis >= 0 or axis < -event_ndims:
            raise ValueError(
                f'`-event_ndims <= axis < 0` does not hold: '
                f'`axis` is {axis}, while `event_ndims` is {event_ndims}.')

        # construct the layer
        super().__init__(x_event_ndims=event_ndims,
                         y_event_ndims=event_ndims,
                         explicitly_invertible=explicitly_invertible)
        self.axis = axis

    @property
    def event_ndims(self) -> int:
        """Get the number of event dimensions in both `x` and `y`."""
        return self.x_event_ndims


# ---- composite flows ----
class InverseFlow(BaseFlow):
    """A flow that inverts another given flow."""

    __constants__ = BaseFlow.__constants__ + ('original_flow',)

    original_flow: Module
    """The original flow, which is inverted by this :class:`InverseFlow`."""

    def __init__(self, flow: Module):
        if (not isinstance(flow, BaseFlow) and not is_jit_layer(flow)) or \
                not flow.explicitly_invertible:
            raise TypeError(
                f'`flow` must be an explicitly invertible flow: '
                f'got {flow!r}'
            )

        super().__init__(
            x_event_ndims=flow.y_event_ndims,
            y_event_ndims=flow.x_event_ndims,
            explicitly_invertible=flow.explicitly_invertible,
        )
        self.original_flow = flow

    def invert(self) -> BaseFlow:
        return self.original_flow

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        return self.original_flow(
            input, input_log_det, not inverse, compute_log_det)


class _NotInvertibleFlow(Module):

    def forward(self,
                input: Tensor,
                input_log_det: Optional[Tensor],
                inverse: bool,
                compute_log_det: bool
                ) -> Tuple[Tensor, Optional[Tensor]]:
        raise RuntimeError('Not an explicitly invertible flow.')


class SequentialFlow(BaseFlow):

    __constants__ = BaseFlow.__constants__ + ('_chain', '_inverse_chain')

    _chain: ModuleList

    # The inverse chain is provided, such that JIT support is still okay.
    # TODO: This separated inverse chain will cause `state_dict()` to have
    #       duplicated weights.  Deal with this issue.
    _inverse_chain: ModuleList

    def __init__(self, *flows: Union[Module, Sequence[Module]]):
        from tensorkit.layers import flatten_nested_layers

        # validate the arguments
        flows = flatten_nested_layers(flows)
        if not flows:
            raise ValueError('`flows` must not be empty.')

        for i, flow in enumerate(flows):
            if not isinstance(flow, BaseFlow) and not is_jit_layer(flow):
                raise TypeError(f'`flows[{i}]` is not a flow: got {flow!r}')

        for i, (flow1, flow2) in enumerate(zip(flows[:-1], flows[1:])):
            if flow2.x_event_ndims != flow1.y_event_ndims:
                raise ValueError(
                    f'`x_event_ndims` of `flows[{i + 1}]` != '
                    f'`y_event_ndims` of `flows[{i}]`: '
                    f'{flow2.x_event_ndims} vs {flow1.y_event_ndims}.'
                )

        super().__init__(
            x_event_ndims=flows[0].x_event_ndims,
            y_event_ndims=flows[-1].y_event_ndims,
            explicitly_invertible=all(
                flow.explicitly_invertible for flow in flows)
        )

        self._chain = ModuleList(flows)
        if self.explicitly_invertible:
            self._inverse_chain = ModuleList(reversed(flows))
        else:
            self._inverse_chain = ModuleList([_NotInvertibleFlow()])

    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        output, output_log_det = input, input_log_det

        if inverse:
            for flow in self._inverse_chain:
                output, output_log_det = flow(
                    output, output_log_det, True, compute_log_det)
        else:
            for flow in self._chain:
                output, output_log_det = flow(
                    output, output_log_det, False, compute_log_det)

        return output, output_log_det


# ---- invertible linear flows ----
class InvertibleMatrix(Module):

    __constants__ = ('size',)

    size: int

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __repr__(self):
        return f'{self.__class__.__qualname__}(size={self.size})'


class LooseInvertibleMatrix(InvertibleMatrix):
    """
    A matrix initialized to be an invertible, orthogonal matrix.

    There is no guarantee that the matrix will keep invertible during training.
    But according to the measure theory, the non-invertible n by n real matrices
    are of measure 0.  Thus this class is generally enough for use.
    """

    def __init__(self,
                 seed_matrix: np.ndarray,
                 dtype: str = settings.float_x):
        """
        Construct a new :class:`LooseInvertibleMatrix`.

        Args:
            seed_matrix: A matrix that is used as a seed to obtain the
                initial invertible and orthogonal matrix.
            dtype: The dtype of the matrix.
        """
        initial_matrix = la.qr(seed_matrix)[0]

        super().__init__(initial_matrix.shape[0])
        add_parameter(self, 'matrix', from_numpy(initial_matrix, dtype=dtype))

    def forward(self,
                inverse: bool,
                compute_log_det: bool
                ) -> Tuple[Tensor, Optional[Tensor]]:
        log_det: Optional[Tensor] = None
        if inverse:
            matrix = matrix_inverse(self.matrix)
            if compute_log_det:
                log_det = -slogdet(self.matrix)[1]
        else:
            matrix = self.matrix
            if compute_log_det:
                log_det = slogdet(self.matrix)[1]
        return matrix, log_det


class StrictInvertibleMatrix(InvertibleMatrix):
    """
    A matrix initialized to be an invertible, orthogonal matrix, and is
    guarnteed to keep invertible during training.
    """

    def __init__(self,
                 seed_matrix: np.ndarray,
                 dtype: str = settings.float_x,
                 epsilon: float = EPSILON):
        """
        Construct a new :class:`StrictInvertibleMatrix`.

        Args:
            seed_matrix: A matrix that is used as a seed to obtain the
                initial invertible and orthogonal matrix.
            dtype: The dtype of the matrix.
        """
        initial_matrix = la.qr(seed_matrix)[0]

        super().__init__(initial_matrix.shape[0])
        matrix_shape = list(initial_matrix.shape)
        self.size = matrix_shape[0]

        initial_P, initial_L, initial_U = la.lu(initial_matrix)
        initial_s = np.diag(initial_U)
        initial_sign = np.sign(initial_s)
        initial_log_s = np.log(np.maximum(np.abs(initial_s), epsilon))
        initial_U = np.triu(initial_U, k=1)

        add_buffer(self, 'P', from_numpy(initial_P, dtype=dtype))
        assert_finite(
            add_parameter(self, 'pre_L', from_numpy(initial_L, dtype=dtype)),
            'pre_L',
        )
        add_buffer(
            self, 'L_mask',
            from_numpy(np.tril(np.ones(matrix_shape), k=-1), dtype=dtype))
        assert_finite(
            add_parameter(self, 'pre_U', from_numpy(initial_U, dtype=dtype)),
            'pre_U',
        )
        add_buffer(
            self, 'U_mask',
            from_numpy(np.triu(np.ones(matrix_shape), k=1), dtype=dtype))
        add_buffer(
            self, 'sign', from_numpy(initial_sign, dtype=dtype))
        assert_finite(
            add_parameter(self, 'log_s', from_numpy(initial_log_s, dtype=dtype)),
            'log_s',
        )

    def forward(self,
                inverse: bool,
                compute_log_det: bool
                ) -> Tuple[Tensor, Optional[Tensor]]:
        P = self.P
        L = (self.L_mask * self.pre_L +
             torch.eye(self.size, dtype=P.dtype))
        U = self.U_mask * self.pre_U + torch.diag(self.sign * exp(self.log_s))

        log_det: Optional[Tensor] = None
        if inverse:
            matrix = matmul(
                matrix_inverse(U),
                matmul(matrix_inverse(L), matrix_inverse(P))
            )
            if compute_log_det:
                log_det = -reduce_sum(self.log_s)
        else:
            matrix = matmul(P, matmul(L, U))
            if compute_log_det:
                log_det = reduce_sum(self.log_s)
        return matrix, log_det


class InvertibleLinearNd(FeatureMappingFlow):
    """Base class for invertible linear transformation flows."""

    __constants__ = FeatureMappingFlow.__constants__ + (
        'invertible_matrix', 'num_features', 'strict', 'epsilon',
    )

    invertible_matrix: Module
    num_features: int
    strict: bool
    epsilon: float

    def __init__(self,
                 num_features: int,
                 strict: bool = False,
                 weight_init: TensorInitArgType = init.kaming_uniform,
                 dtype: str = settings.float_x,
                 epsilon: float = EPSILON):
        """
        Construct a new linear transformation flow.

        Args:
            num_features: The number of features to be transformed.
                The invertible transformation matrix will have the shape
                ``[num_features, num_features]``.
            strict: Whether or not to use the strict invertible matrix?
                Defaults to :obj:`False`.  See :class:`LooseInvertibleMatrix`
                and :class:`StrictInvertibleMatrix`.
            weight_init: The weight initializer for the seed matrix.
            dtype: The dtype of the invertible matrix.
            epsilon: The infinitesimal constant to avoid having numerical issues.
        """
        spatial_ndims = self._get_spatial_ndims()
        super().__init__(
            axis=-(spatial_ndims + 1),
            event_ndims=(spatial_ndims + 1),
            explicitly_invertible=True,
        )

        self.num_features = int(num_features)
        self.strict = bool(strict)
        self.epsilon = float(epsilon)

        # Using the backend random generator instead of numpy generator
        # will allow the backend random seed to have effect on the initialization
        # step of the invertible matrix.
        seed_matrix = variable(
            shape=[num_features, num_features], dtype=dtype,
            initializer=weight_init, requires_grad=False,
        )

        if strict:
            self.invertible_matrix = StrictInvertibleMatrix(
                to_numpy(seed_matrix), dtype=dtype, epsilon=epsilon)
        else:
            self.invertible_matrix = LooseInvertibleMatrix(
                to_numpy(seed_matrix), dtype=dtype)

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        raise NotImplementedError()

    @jit_method
    def _forward(self,
                 input: Tensor,
                 input_log_det: Optional[Tensor],
                 inverse: bool,
                 compute_log_det: bool
                 ) -> Tuple[Tensor, Optional[Tensor]]:
        # obtain the weight
        weight, log_det = self.invertible_matrix(
            inverse=inverse, compute_log_det=compute_log_det)
        spatial_ndims = self.x_event_ndims - 1
        weight = reshape(weight, shape(weight) + [1] * spatial_ndims)

        # compute the output
        output, front_shape = flatten_to_ndims(input, spatial_ndims + 2)
        output = self._linear_transform(output, weight)
        output = unflatten_from_ndims(output, front_shape)

        # compute the log_det
        output_log_det = input_log_det
        if log_det is not None:
            for axis in int_range(-spatial_ndims, 0):
                log_det = log_det * float(input.shape[axis])
            if input_log_det is not None:
                output_log_det = input_log_det + log_det
            else:
                output_log_det = log_det

        return output, output_log_det


class InvertibleDense(InvertibleLinearNd):
    """An invertible linear transformation."""

    def _get_spatial_ndims(self) -> int:
        return 0

    @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.linear(input, weight)


class InvertibleConv1d(InvertibleLinearNd):
    """An invertible 1d 1x1 convolutional transformation."""

    def _get_spatial_ndims(self) -> int:
        return 1

    @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.conv1d(input, weight)


class InvertibleConv2d(InvertibleLinearNd):
    """An invertible 2d 1x1 convolutional transformation."""

    def _get_spatial_ndims(self) -> int:
        return 2

    @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.conv2d(input, weight)


class InvertibleConv3d(InvertibleLinearNd):
    """An invertible 3d 1x1 convolutional transformation."""

    def _get_spatial_ndims(self) -> int:
        return 3

    @jit_method
    def _linear_transform(self, input: Tensor, weight: Tensor) -> Tensor:
        return torch.nn.functional.conv3d(input, weight)


# ---- scale modules, for transforming input to output by a scale ----
class BaseScale(BaseLayer):
    """Base class for scaling `input`."""

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        raise NotImplementedError()

    def forward(self,
                input: Tensor,
                pre_scale: Tensor,
                event_ndims: int,
                input_log_det: Optional[Tensor] = None,
                compute_log_det: bool = True,
                inverse: bool = False
                ) -> Tuple[Tensor, Optional[Tensor]]:
        # validate the argument
        if input.dim() < event_ndims:
            raise ValueError(
                '`rank(input) >= event_ndims` does not hold: the `input` shape '
                'is {}, while `event_ndims` is {}.'.
                format(shape(input), event_ndims)
            )
        if pre_scale.dim() > input.dim():
            raise ValueError(
                '`rank(input) >= rank(pre_scale)` does not hold: the `input` '
                'shape is {}, while the shape of `pre_scale` is {}.'.
                format(shape(input), shape(pre_scale))
            )

        input_shape = shape(input)
        event_ndims_start = len(input_shape) - event_ndims
        event_shape = input_shape[event_ndims_start:]
        log_det_shape = input_shape[: event_ndims_start]

        if input_log_det is not None:
            if shape(input_log_det) != log_det_shape:
                raise ValueError(
                    'The shape of `input_log_det` is not expected: '
                    'expected to be {}, but got {}'.
                    format(log_det_shape, shape(input_log_det))
                )

        scale, log_scale = self._scale_and_log_scale(
            pre_scale, inverse, compute_log_det)
        output = input * scale

        if log_scale is not None:
            log_scale = broadcast_to(
                log_scale,
                broadcast_shape(shape(log_scale), event_shape)
            )

            # the last `event_ndims` dimensions must match the `event_shape`
            log_scale_shape = shape(log_scale)
            log_scale_event_shape = \
                log_scale_shape[len(log_scale_shape) - event_ndims:]
            if log_scale_event_shape != event_shape:
                raise ValueError(
                    'The shape of the final {}d of `log_scale` is not expected: '
                    'expected to be {}, but got {}.'.
                    format(event_ndims, event_shape, log_scale_event_shape)
                )

            # reduce the last `event_ndims` of log_scale
            log_scale = reduce_sum(log_scale, axis=int_range(-event_ndims, 0))

            # now add to input_log_det, or broadcast `log_scale` to `log_det_shape`
            if input_log_det is not None:
                output_log_det = input_log_det + log_scale
                if shape(output_log_det) != log_det_shape:
                    raise ValueError(
                        'The shape of the computed `output_log_det` is not expected: '
                        'expected to be {}, but got {}.'.
                        format(shape(output_log_det), log_det_shape)
                    )
            else:
                output_log_det = broadcast_to(log_scale, log_det_shape)
        else:
            output_log_det = None

        return output, output_log_det


class ExpScale(BaseScale):
    """
    Scaling `input` with `exp` activation.

    ::

        if inverse:
            output = input / exp(pre_scale)
            output_log_det = -pre_scale
        else:
            output = input * exp(pre_scale)
            output_log_det = pre_scale
    """

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        log_scale: Optional[Tensor] = None
        if inverse:
            neg_pre_scale = -pre_scale
            scale = exp(neg_pre_scale)
            if compute_log_scale:
                log_scale = neg_pre_scale
        else:
            scale = exp(pre_scale)
            if compute_log_scale:
                log_scale = pre_scale
        return scale, log_scale


class SigmoidScale(BaseScale):
    """
    Scaling `input` with `sigmoid` activation.

    ::

        if inverse:
            output = input / sigmoid(pre_scale)
            output_log_det = -log(sigmoid(pre_scale))
        else:
            output = input * sigmoid(pre_scale)
            output_log_det = log(sigmoid(pre_scale))
    """

    __constants__ = ('pre_scale_bias',)

    pre_scale_bias: float

    def __init__(self, pre_scale_bias: float = 0.):
        super().__init__()
        self.pre_scale_bias = pre_scale_bias

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        if self.pre_scale_bias != 0.:
            pre_scale = pre_scale + self.pre_scale_bias

        log_scale: Optional[Tensor] = None
        if inverse:
            neg_pre_scale = -pre_scale
            scale = exp(neg_pre_scale) + 1.
            if compute_log_scale:
                log_scale = softplus(neg_pre_scale)
        else:
            scale = sigmoid(pre_scale)
            if compute_log_scale:
                log_scale = -softplus(-pre_scale)

        return scale, log_scale


class LinearScale(BaseScale):
    """
    Scaling `input` with `linear` activation.

    ::

        if inverse:
            output = input / pre_scale
            output_log_det = -log(abs(pre_scale))
        else:
            output = input * pre_scale
            output_log_det = log(abs(pre_scale))
    """

    __constants__ = ('epsilon',)

    epsilon: float

    def __init__(self, epsilon: float = EPSILON):
        super().__init__()
        self.epsilon = epsilon

    def _scale_and_log_scale(self,
                             pre_scale: Tensor,
                             inverse: bool,
                             compute_log_scale: bool
                             ) -> Tuple[Tensor, Optional[Tensor]]:
        log_scale: Optional[Tensor] = None
        if inverse:
            scale = 1. / pre_scale
            if compute_log_scale:
                epsilon = as_tensor_backend(self.epsilon, dtype=pre_scale.dtype)
                log_scale = -log(maximum(abs(pre_scale), epsilon))
        else:
            scale = pre_scale
            if compute_log_scale:
                epsilon = as_tensor_backend(self.epsilon, dtype=pre_scale.dtype)
                log_scale = log(maximum(abs(pre_scale), epsilon))

        return scale, log_scale
