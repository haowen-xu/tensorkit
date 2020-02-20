import unittest

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.flows import *
from tests.helper import *
from tests.ops import *


def check_split_flow(ctx,
                     spatial_ndims: int,
                     num_features: int,
                     cls,
                     x_sections,
                     left,
                     right,
                     **kwargs):
    y_sections = kwargs.get("y_sections", x_sections)
    x_axis = kwargs.get("x_axis", get_channel_axis(spatial_ndims))
    y_axis = kwargs.get("y_axis", x_axis)

    for batch_shape in ([2], [2, 3]):
        x = T.random.randn(make_conv_shape(
            batch_shape, num_features, [6, 7, 8][:spatial_ndims]))
        input_log_det = T.random.randn(batch_shape)

        # without right
        if y_axis == x_axis:
            flow = cls(x_sections, left, None, **kwargs)
            ctx.assertIn(f'x_sections={x_sections}', repr(flow))
            ctx.assertIn(f'y_sections={y_sections}', repr(flow))
            ctx.assertIn(f'x_axis={x_axis}', repr(flow))
            ctx.assertIn(f'y_axis={y_axis}', repr(flow))
            flow = tk.layers.jit_compile(flow)

            x1, x2 = T.split(x, x_sections, axis=x_axis)
            y1, expected_log_det = left(x1, compute_log_det=True)
            y2 = x2
            expected_y = T.concat([y1, y2], axis=y_axis)

            flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                                input_log_det)

        # with right
        flow = cls(x_sections, left, right, **kwargs)
        flow = tk.layers.jit_compile(flow)

        x1, x2 = T.split(x, x_sections, axis=x_axis)
        y1, expected_log_det = left(x1, compute_log_det=True)
        y2, expected_log_det = right(
            x2, input_log_det=expected_log_det, compute_log_det=True)
        expected_y = T.concat([y1, y2], axis=y_axis)

        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            input_log_det)

    # test argument error
    with pytest.raises(ValueError,
                       match='`x_sections` must be a sequence of '
                             'two positive integers'):
        _ = cls([1, 2, 3], left)

    with pytest.raises(ValueError,
                       match='`x_sections` must be a sequence of '
                             'two positive integers'):
        _ = cls([-1, 2], left)

    with pytest.raises(ValueError,
                       match='`y_sections` must be None or a sequence of '
                             'two positive integers'):
        _ = cls([2, 3], left, right, y_sections=[1, 2, 3])

    with pytest.raises(ValueError,
                       match='`y_sections` must be None or a sequence of '
                             'two positive integers'):
        _ = cls([2, 3], left, right, y_sections=[-1, 2])

    with pytest.raises(TypeError, match='`left` is not a flow'):
        _ = cls([2, 3], tk.layers.Linear(2, 3))

    with pytest.raises(TypeError, match='`right` is not a flow'):
        _ = cls([2, 3], left, tk.layers.Linear(2, 3))


class SplitFlowTestCase(TestCase):

    @slow_test
    def test_SplitFlow(self):
        # x and y with the same event ndims
        left = tk.layers.jit_compile(InvertibleDense(2))
        right = tk.layers.jit_compile(InvertibleDense(3))

        check_split_flow(
            ctx=self,
            spatial_ndims=0,
            num_features=5,
            cls=SplitFlow,
            x_sections=[2, 3],
            left=left,
            right=right,
        )

        # test argument error
        with pytest.raises(ValueError,
                           match=f'`left` and `right` flows must have same '
                                 f'`x_event_ndims` and `y_event_ndims`: '
                                 f'got `left.x_event_ndims` == {left.get_x_event_ndims()}, '
                                 f'`left.y_event_ndims` == {left.get_y_event_ndims()}, '
                                 f'`right.x_event_ndims` == {left.get_x_event_ndims()}, '
                                 f'and `right.y_event_ndims` == 6'):
            _ = SplitFlow([2, 3], left, ReshapeFlow([1] * left.get_x_event_ndims(), [1] * 6))

        with pytest.raises(ValueError,
                           match=f'`left` and `right` flows must have same '
                                 f'`x_event_ndims` and `y_event_ndims`: '
                                 f'got `left.x_event_ndims` == {left.get_x_event_ndims()}, '
                                 f'`left.y_event_ndims` == {left.get_y_event_ndims()}, '
                                 f'`right.x_event_ndims` == 6, '
                                 f'and `right.y_event_ndims` == {left.get_y_event_ndims()}'):
            _ = SplitFlow([2, 3], left, ReshapeFlow([1] * 6, [1] * left.get_y_event_ndims()))

        # x and y with different event ndims
        left = ReshapeFlow([-1], [-1, 2])
        right = ReshapeFlow([-1], [-1, 2])

        check_split_flow(
            ctx=self,
            spatial_ndims=0,
            num_features=10,
            cls=SplitFlow,
            x_sections=[4, 6],
            y_sections=[2, 3],
            left=left,
            right=right,
            x_axis=-1,
            y_axis=-2,
        )

        # test argument error
        with pytest.raises(ValueError,
                           match=f'`x_event_ndims` != `y_event_ndims`, thus '
                                 f'`y_axis` must be specified'):
            _ = SplitFlow([2, 3], left)

        with pytest.raises(ValueError,
                           match=f'`x_event_ndims` != `y_event_ndims`, thus '
                                 '`right` must be specified'):
            _ = SplitFlow([2, 3], left, y_axis=-2)

    @slow_test
    def test_SplitFlowNd(self):
        for spatial_ndims in (1, 2, 3):
            cls = getattr(tk.flows, f'SplitFlow{spatial_ndims}d')
            sub_cls = getattr(tk.flows, f'InvertibleConv{spatial_ndims}d')

            left = tk.layers.jit_compile(sub_cls(2))
            right = tk.layers.jit_compile(sub_cls(3))

            check_split_flow(
                ctx=self,
                spatial_ndims=spatial_ndims,
                num_features=5,
                cls=cls,
                x_sections=[2, 3],
                left=left,
                right=right,
            )

            # this class should validate the event_ndims of left and right flow
            for arg_name in ('left', 'right'):
                kwargs = {'left': left, 'right': right}
                kwargs[arg_name] = sub_cls({'left': 2, 'right': 3}[arg_name])

                for attr_name in ('x_event_ndims', 'y_event_ndims'):
                    setattr(kwargs[arg_name], attr_name, 1)
                    with pytest.raises(ValueError,
                                       match=f'The `x_event_ndims` and '
                                             f'`y_event_ndims` of `{arg_name}` '
                                             f'are required to be {spatial_ndims + 1}'):
                        _ = cls([2, 3], **kwargs)
                    setattr(kwargs[arg_name], attr_name, spatial_ndims + 1)
