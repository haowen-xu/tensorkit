import unittest

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.flows import *
from tests.helper import *
from tests.ops import *


def check_coupling_layer(ctx,
                         spatial_ndims: int,
                         num_features: int,
                         cls,
                         shift_and_pre_scale_factory):
    batch_shape = [11]
    sigmoid_scale_bias = 1.5

    n1, n2 = (num_features // 2), (num_features - num_features // 2)
    shift_and_pre_scale_1 = tk.layers.jit_compile(shift_and_pre_scale_factory(n1, n2))
    shift_and_pre_scale_2 = tk.layers.jit_compile(shift_and_pre_scale_factory(n2, n1))

    def do_check(secondary, scale_type):
        x = T.random.randn(make_conv_shape(
            batch_shape, num_features, [6, 7, 8][:spatial_ndims]))
        n1, n2 = (num_features // 2), (num_features - num_features // 2)

        # construct the instance
        shift_and_pre_scale = (shift_and_pre_scale_2
                               if secondary else shift_and_pre_scale_1)
        flow = cls(
            shift_and_pre_scale, scale=scale_type, secondary=secondary,
            sigmoid_scale_bias=sigmoid_scale_bias
        )
        ctx.assertIn(f'secondary={secondary}', repr(flow))
        flow = tk.layers.jit_compile(flow)

        # obtain the expected output
        channel_axis = get_channel_axis(spatial_ndims)
        x1, x2 = T.split(x, [n1, n2], axis=channel_axis)
        if secondary:
            x1, x2 = x2, x1

        y1 = x1
        shift, pre_scale = shift_and_pre_scale(x1)
        if scale_type == 'exp' or scale_type is ExpScale:
            scale = ExpScale()
        elif scale_type == 'sigmoid' or scale_type is SigmoidScale:
            scale = SigmoidScale(pre_scale_bias=sigmoid_scale_bias)
        elif scale_type == 'linear' or scale_type is LinearScale:
            scale = LinearScale()
        elif isinstance(scale_type, Scale) or tk.layers.is_jit_layer(scale_type):
            scale = scale_type
        else:
            raise ValueError(f'Invalid value for `scale`: {scale_type}')
        y2, log_det = scale(x2 + shift, pre_scale,
                            event_ndims=spatial_ndims + 1,
                            compute_log_det=True)

        if secondary:
            y1, y2 = y2, y1
        expected_y = T.concat([y1, y2], axis=channel_axis)
        expected_log_det = log_det

        # now check the flow
        flow_standard_check(ctx, flow, x, expected_y, expected_log_det,
                            T.random.randn(batch_shape))

    for secondary in (False, True):
        do_check(secondary, 'exp')

    for scale_type in ('exp', 'sigmoid', 'linear',
                       SigmoidScale, tk.layers.jit_compile(LinearScale())):
        do_check(False, scale_type)

    # test error constructors
    shift_and_pre_scale = shift_and_pre_scale_factory(2, 3)
    for scale in ('invalid', object(), tk.layers.Linear(2, 3)):
        with pytest.raises(ValueError,
                           match=r'`scale` must be a `BaseScale` class, '
                                 r'an instance of `BaseScale`, a factory to '
                                 r'construct a `BaseScale` instance, or one of '
                                 r'\{"exp", "sigmoid", "linear"\}'):
            _ = cls(shift_and_pre_scale, scale=scale)


class CouplingLayerTestCase(TestCase):

    @slow_test
    def test_CouplingLayer(self):
        def shift_and_pre_scale_factory(n1, n2):
            return tk.layers.Branch(
                [
                    tk.layers.Linear(10, n2),
                    tk.layers.Linear(10, n2),
                ],
                shared=tk.layers.Linear(n1, 10),
            )

        check_coupling_layer(
            self,
            spatial_ndims=0,
            num_features=5,
            cls=CouplingLayer,
            shift_and_pre_scale_factory=shift_and_pre_scale_factory,
        )

    @slow_test
    def test_CouplingLayerNd(self):
        for spatial_ndims in (1, 2, 3):
            conv_cls = getattr(tk.layers, f'LinearConv{spatial_ndims}d')

            def shift_and_pre_scale_factory(n1, n2):
                return tk.layers.Branch(
                    [
                        conv_cls(10, n2, kernel_size=1),
                        conv_cls(10, n2, kernel_size=1),
                    ],
                    shared=conv_cls(n1, 10, kernel_size=1),
                )

            check_coupling_layer(
                self,
                spatial_ndims=spatial_ndims,
                num_features=5,
                cls=getattr(tk.flows, f'CouplingLayer{spatial_ndims}d'),
                shift_and_pre_scale_factory=shift_and_pre_scale_factory,
            )
