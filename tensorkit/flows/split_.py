from typing import *

from .. import tensor as T
from ..layers import is_jit_layer
from ..tensor import Tensor, Module, split, concat
from .core import *

__all__ = [
    'SplitFlow', 'SplitFlow1d', 'SplitFlow2d', 'SplitFlow3d',
]


class SplitFlow(Flow):
    """
    A flow which splits input `x` into halves, apply different flows on each
    half, then concat the output together.

    Basically, a :class:`SplitFlow` performs the following transformation::

        x1, x2 = split(x, axis=axis, section=sections)
        y1, log_det1 = left(x1)
        if right is not None:
            y2, log_det2 = right(x2)
        else:
            y2, log_det2 = x2, 0.
        y = concat([y1, y2], axis=axis)
        log_det = log_det1 + log_det2
    """

    __constants__ = Flow.__constants__ + (
        'left', 'right', 'x_sections', 'x_axis', 'y_sections', 'y_axis',
    )

    left: Module
    right: Module
    x_sections: List[int]
    x_axis: int
    y_sections: List[int]
    y_axis: int

    def __init__(self,
                 x_sections: Sequence[int],
                 left: Flow,
                 right: Optional[Flow] = None,
                 y_sections: Optional[Sequence[int]] = None,
                 x_axis: int = -1,
                 y_axis: Optional[int] = None):
        """
        Construct a new :class:`SplitFlow`.

        Args:
            x_sections: The size of each sections of the splitted `x` along
                `x_axis`.
            left : The `left` flow (see above).
            right: The `right` flow (see above).
                `right.x_event_ndims` must equal to `left.x_event_ndims`, and
                `right.y_event_ndims` must equal to `left.y_event_ndims`.
                If not specified, the right flow will be identity.
                Must be specified if `left.x_event_ndims != left.y_event_ndims`.
            y_sections: The size of each sections of the splitted `y` along
                `y_axis`.  Defaults to be the same as `x_sections`.
            x_axis: Along which axis to split or join `x`.
            y_axis: Along which axis to split or join `y`.
                If not specified, use `x_axis`.
                Must be specified if `left.x_event_ndims != left.y_event_ndims`.
        """
        # validate the arguments
        if len(x_sections) != 2 or any(s < 1 for s in x_sections):
            raise ValueError(f'`x_sections` must be a sequence of '
                             f'two positive integers: got {y_sections!r}.')
        x_sections = list(map(int, x_sections))

        x_axis = int(x_axis)

        if y_sections is None:
            y_sections = x_sections
        else:
            if len(y_sections) != 2 or any(s < 1 for s in y_sections):
                raise ValueError(f'`y_sections` must be None or a sequence of '
                                 f'two positive integers: got {y_sections!r}.')
            y_sections = list(map(int, y_sections))

        if not isinstance(left, Flow) and not is_jit_layer(left):
            raise TypeError(f'`left` is not a flow: got {left!r}.')
        x_event_ndims = left.get_x_event_ndims()
        y_event_ndims = left.get_y_event_ndims()

        if right is not None:
            if not isinstance(right, Flow) and not is_jit_layer(right):
                raise TypeError(f'`right` is not a flow: got {right!r}.')
            if right.get_x_event_ndims() != x_event_ndims or \
                    right.get_y_event_ndims() != y_event_ndims:
                raise ValueError(
                    f'`left` and `right` flows must have same `x_event_ndims` '
                    f'and `y_event_ndims`: '
                    f'got `left.x_event_ndims` == {left.get_x_event_ndims()!r}, '
                    f'`left.y_event_ndims` == {left.get_y_event_ndims()}, '
                    f'`right.x_event_ndims` == {right.get_x_event_ndims()}, '
                    f'and `right.y_event_ndims` == {right.get_y_event_ndims()}.'
                )

        if x_event_ndims != y_event_ndims:
            if y_axis is None:
                raise ValueError('`x_event_ndims` != `y_event_ndims`, thus '
                                 '`y_axis` must be specified.')
            if right is None:
                raise ValueError('`x_event_ndims` != `y_event_ndims`, thus '
                                 '`right` must be specified.')

        else:
            if y_axis is None:
                y_axis = x_axis
            else:
                y_axis = int(y_axis)

        super(SplitFlow, self).__init__(
            x_event_ndims=x_event_ndims,
            y_event_ndims=y_event_ndims,
            explicitly_invertible=True,
        )
        self.left = left
        self.right = right
        self.x_sections = x_sections
        self.x_axis = x_axis
        self.y_sections = y_sections
        self.y_axis = y_axis

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        sections: List[int] = []
        if inverse:
            sections.extend(self.y_sections)
            axis = self.y_axis
            join_axis = self.x_axis
        else:
            sections.extend(self.x_sections)
            axis = self.x_axis
            join_axis = self.y_axis

        out_left, out_right = split(input, sections=sections, axis=axis)
        out_left, output_log_det = self.left(
            input=out_left, input_log_det=input_log_det, inverse=inverse,
            compute_log_det=compute_log_det,
        )
        if self.right is not None:
            out_right, output_log_det = self.right(
                input=out_right, input_log_det=output_log_det, inverse=inverse,
                compute_log_det=compute_log_det,
            )

        output = concat([out_left, out_right], axis=join_axis)
        return output, output_log_det


class SplitFlowNd(SplitFlow):

    def __init__(self,
                 x_sections: Sequence[int],
                 left: Flow,
                 right: Optional[Flow] = None,
                 y_sections: Optional[Sequence[int]] = None):
        """
        Construct a new convolutional split flow.

        Args:
            x_sections: The size of each sections of the splitted `x` along
                the channel axis.
            left : The `left` flow.
            right: The `right` flow.
            y_sections: The size of each sections of the splitted `y` along
                the channel axis.  Defaults to be the same as `x_sections`.
        """
        spatial_ndims = self._get_spatial_ndims()
        feature_axis = -1 if T.IS_CHANNEL_LAST else -(spatial_ndims + 1)
        event_ndims = spatial_ndims + 1

        for arg_name, arg in [('left', left), ('right', right)]:
            # type error deferred to the base class, thus we only check
            # the event ndims if `arg` looks like a flow.
            if arg is not None and hasattr(arg, 'x_event_ndims'):
                if arg.get_x_event_ndims() != event_ndims or \
                        arg.get_y_event_ndims() != event_ndims:
                    raise ValueError(
                        f'The `x_event_ndims` and `y_event_ndims` of '
                        f'`{arg_name}` are required to be {event_ndims}: '
                        f'got `x_event_ndims` == {arg.get_x_event_ndims()}, '
                        f'and `y_event_ndims` == {arg.get_y_event_ndims()}.'
                    )

        super().__init__(
            x_sections=x_sections,
            left=left,
            right=right,
            y_sections=y_sections,
            x_axis=feature_axis,
            y_axis=feature_axis,
        )

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class SplitFlow1d(SplitFlowNd):
    """
    A flow which splits the channel axis of 1D convolutional `x` into halves,
    apply different 1D convolutional flows on each half, then concat the
    output together.
    """

    def _get_spatial_ndims(self) -> int:
        return 1


class SplitFlow2d(SplitFlowNd):
    """
    A flow which splits the channel axis of 2D convolutional `x` into halves,
    apply different 2D convolutional flows on each half, then concat the
    output together.
    """

    def _get_spatial_ndims(self) -> int:
        return 2


class SplitFlow3d(SplitFlowNd):
    """
    A flow which splits the channel axis of 3D convolutional `x` into halves,
    apply different 3D convolutional flows on each half, then concat the
    output together.
    """

    def _get_spatial_ndims(self) -> int:
        return 3
