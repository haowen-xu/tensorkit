from typing import *

from ..tensor import Tensor, reshape_tail, as_tensor_backend, jit_method
from ..tensor.nn import *
from .core import *

__all__ = [
    'ReshapeFlow',
    'DepthToSpace1d', 'DepthToSpace2d', 'DepthToSpace3d',
    'SpaceToDepth1d', 'SpaceToDepth2d', 'SpaceToDepth3d',
]


class ReshapeFlow(Flow):
    """
    A flow which reshapes the last `x_event_ndims` of `x` into `y_event_shape`.

    Usage::

        # to reshape a conv2d output into dense input
        flow = ReshapeFlow(x_event_ndims=3, y_event_shape=[-1])
        x = T.random.randn([2, 3, 4, 5])
        y, log_det = flow.transform(x)

        # y == T.reshape(x, [2, -1])
        # log_det == tf.zeros([2])
    """

    __constants__ = Flow.__constants__ + ('x_event_shape', 'y_event_shape')

    x_event_shape: List[int]
    y_event_shape: List[int]

    def __init__(self,
                 x_event_shape: Sequence[int],
                 y_event_shape: Sequence[int]):
        """
        Construct a new :class:`ReshapeFlow`.

        Args:
            x_event_shape: The event shape of `x`.
            y_event_shape: The event shape of `y`.
        """
        def check_shape(name, event_shape):
            event_shape = list(map(int, event_shape))
            neg_one_count = 0
            for s in event_shape:
                if s == -1:
                    if neg_one_count == 0:
                        neg_one_count += 1
                    else:
                        raise ValueError(
                            f'Too many `-1` specified in `{name}`: '
                            f'got {event_shape!r}.'
                        )
                elif s < 0:
                    raise ValueError(
                        f'All elements of `{name}` must be positive '
                        f'integers or `-1`: got {event_shape!r}.'
                    )
            return event_shape

        x_event_shape = check_shape('x_event_shape', x_event_shape)
        y_event_shape = check_shape('y_event_shape', y_event_shape)

        super(ReshapeFlow, self).__init__(
            x_event_ndims=len(x_event_shape),
            y_event_ndims=len(y_event_shape),
            explicitly_invertible=True
        )
        self.x_event_shape = x_event_shape
        self.y_event_shape = y_event_shape

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = reshape_tail(input, self.y_event_ndims, self.x_event_shape)
        else:
            output = reshape_tail(input, self.x_event_ndims, self.y_event_shape)

        output_log_det = input_log_det
        if compute_log_det and output_log_det is None:
            output_log_det = as_tensor_backend(0., dtype=input.dtype)
        return output, output_log_det


class SpaceDepthTransformFlow(Flow):

    __constants__ = ('block_size',)

    block_size: int

    def __init__(self, block_size: int):
        """
        Construct a new instance.

        Args:
            block_size: The block size for space-depth transformation.
        """
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError('`block_size` must be at least 1.')

        self.block_size = block_size
        super().__init__(
            x_event_ndims=self._get_spatial_ndim() + 1,
            y_event_ndims=self._get_spatial_ndim() + 1,
            explicitly_invertible=True,
        )

    def _get_spatial_ndim(self) -> int:
        raise NotImplementedError()

    def _transform_forward(self, input: Tensor) -> Tensor:
        raise NotImplementedError()

    def _transform_inverse(self, input: Tensor) -> Tensor:
        raise NotImplementedError()

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = self._transform_inverse(input)
        else:
            output = self._transform_forward(input)

        output_log_det = input_log_det
        if compute_log_det and output_log_det is None:
            output_log_det = as_tensor_backend(0., dtype=input.dtype)

        return output, output_log_det


class SpaceToDepth1d(SpaceDepthTransformFlow):
    """A flow that transforms the input by :func:`T.nn.space_to_depth1d`."""

    def _get_spatial_ndim(self) -> int:
        return 1

    @jit_method
    def _transform_forward(self, input: Tensor) -> Tensor:
        return space_to_depth1d(input, self.block_size)

    @jit_method
    def _transform_inverse(self, input: Tensor) -> Tensor:
        return depth_to_space1d(input, self.block_size)

    def invert(self) -> Flow:
        return DepthToSpace1d(self.block_size)


class SpaceToDepth2d(SpaceDepthTransformFlow):
    """A flow that transforms the input by :func:`T.nn.space_to_depth2d`."""

    def _get_spatial_ndim(self) -> int:
        return 2

    @jit_method
    def _transform_forward(self, input: Tensor) -> Tensor:
        return space_to_depth2d(input, self.block_size)

    @jit_method
    def _transform_inverse(self, input: Tensor) -> Tensor:
        return depth_to_space2d(input, self.block_size)

    def invert(self) -> Flow:
        return DepthToSpace2d(self.block_size)


class SpaceToDepth3d(SpaceDepthTransformFlow):
    """A flow that transforms the input by :func:`T.nn.space_to_depth3d`."""

    def _get_spatial_ndim(self) -> int:
        return 3

    @jit_method
    def _transform_forward(self, input: Tensor) -> Tensor:
        return space_to_depth3d(input, self.block_size)

    @jit_method
    def _transform_inverse(self, input: Tensor) -> Tensor:
        return depth_to_space3d(input, self.block_size)

    def invert(self) -> Flow:
        return DepthToSpace3d(self.block_size)


class DepthToSpace1d(SpaceDepthTransformFlow):
    """A flow that transforms the input by :func:`T.nn.depth_to_space1d`."""

    def _get_spatial_ndim(self) -> int:
        return 1

    @jit_method
    def _transform_forward(self, input: Tensor) -> Tensor:
        return depth_to_space1d(input, self.block_size)

    @jit_method
    def _transform_inverse(self, input: Tensor) -> Tensor:
        return space_to_depth1d(input, self.block_size)

    def invert(self) -> Flow:
        return SpaceToDepth1d(self.block_size)


class DepthToSpace2d(SpaceDepthTransformFlow):
    """A flow that transforms the input by :func:`T.nn.depth_to_space2d`."""

    def _get_spatial_ndim(self) -> int:
        return 2

    @jit_method
    def _transform_forward(self, input: Tensor) -> Tensor:
        return depth_to_space2d(input, self.block_size)

    @jit_method
    def _transform_inverse(self, input: Tensor) -> Tensor:
        return space_to_depth2d(input, self.block_size)

    def invert(self) -> Flow:
        return SpaceToDepth2d(self.block_size)


class DepthToSpace3d(SpaceDepthTransformFlow):
    """A flow that transforms the input by :func:`T.nn.depth_to_space3d`."""

    def _get_spatial_ndim(self) -> int:
        return 3

    @jit_method
    def _transform_forward(self, input: Tensor) -> Tensor:
        return depth_to_space3d(input, self.block_size)

    @jit_method
    def _transform_inverse(self, input: Tensor) -> Tensor:
        return space_to_depth3d(input, self.block_size)

    def invert(self) -> Flow:
        return SpaceToDepth3d(self.block_size)
