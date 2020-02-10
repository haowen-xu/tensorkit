import unittest
from itertools import product
from typing import *

import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.tensor import Tensor
from tests.helper import *
from tests.ops import *


def make_causal_test_input(size: List[int],
                           pos: List[int],
                           single_point: bool = True,
                           ) -> np.ndarray:
    ret = np.zeros(size, dtype=np.float32)
    if single_point:
        tmp = ret
        for p in pos[:-1]:
            tmp = tmp[p]
        tmp[pos[-1]] = 1.
    else:
        tmp = ret
        for p in pos[:-1]:
            tmp[(p+1):] = 1.
            tmp = tmp[p]
        tmp[pos[-1]:] = 1.
    return np.reshape(ret, make_conv_shape([1], 1, size))


def make_causal_mask(size: List[int], pos: List[int]) -> np.ndarray:
    ret = make_causal_test_input(size, pos, single_point=False)
    r_shape = ret.shape
    ret = ret.reshape(size)
    tmp = ret
    for p in pos[:-1]:
        tmp = tmp[p]
    tmp[pos[-1]] = 0.
    return ret.reshape(r_shape)


def iter_causal_test_pos(size: List[int]):
    return list(product(*([0, s // 2, s - 1] for s in size)))


def ensure_stacks_causality(ctx,
                            outputs,
                            size: List[int],
                            pos: List[int]):
    ctx.assertEqual(len(outputs), len(size))
    spatial_ndims = len(outputs)
    for i in range(spatial_ndims):
        output = outputs[i]
        if isinstance(output, T.Tensor):
            output = T.to_numpy(output)
        output = output.reshape(size)
        this_pos = list(pos)
        this_pos[i] += 1
        k = i
        while k > 0 and this_pos[k] >= size[k]:
            this_pos[k - 1] += 1
            this_pos[k] = 0
            k -= 1
        for j in range(i + 1, spatial_ndims):
            this_pos[j] = 0
        if this_pos[0] >= size[0]:
            mask = np.zeros(size, dtype=np.float32)
        else:
            mask = make_causal_test_input(size, this_pos, single_point=False)
        is_wrong = np.any(
            np.logical_and(
                np.abs(output) > 1e-6,
                np.logical_not(mask.astype(np.bool))
            )
        )
        ctx.assertFalse(
            is_wrong,
            msg=f'stack.id={i}, pos={pos}, output={output}, mask={mask}'
        )


def ensure_full_receptive_field(ctx,
                                output,
                                size: List[int],
                                pos: List[int]):
    if isinstance(output, T.Tensor):
        output = T.to_numpy(output)
    output_true = (np.abs(output.reshape(size)) >= 1e-6).astype(np.int32)
    mask = make_causal_mask(size, pos).astype(np.int32)
    ctx.assertTrue(
        np.all(
            np.logical_not(
                np.logical_xor(
                    mask.astype(np.bool),
                    output_true.astype(np.bool)
                )
            )
        ),
        msg=f'pos={pos}, output_true={output_true}, mask={mask}'
    )


class _MyAddContext(tk.layers.BaseContextualLayer):

    def _call(self, input: Tensor, context: List[Tensor]) -> Tensor:
        if len(context) == 0:
            return input
        elif len(context) == 1:
            return input + context[0]
        else:
            raise ValueError('Expected context to have 0 or 1 element.')


class PixelCNNTestCase(unittest.TestCase):

    def test_causality_and_receptive_field(self):
        for size in [[12], [12, 11], [12, 11, 10]]:
            spatial_ndims = len(size)

            for kernel_size in [3, 5, [5, 3, 5][:spatial_ndims]]:
                # ---- construct the layers ----
                # the input layer
                input_layer_cls = getattr(
                    tk.layers, f'PixelCNNInput{spatial_ndims}d')
                input_layer = input_layer_cls(
                    1, 1, kernel_size=kernel_size, add_ones_channel=False,
                    weight_init=tk.init.ones,
                )
                input_layer = T.jit_compile(input_layer)

                with pytest.raises(Exception,
                                   match='`input` is expected to be .*d'):
                    _ = input_layer(T.zeros([1] * (spatial_ndims + 1)))
                with pytest.raises(Exception,
                                   match='`input` is expected to be .*d'):
                    _ = input_layer(T.zeros([1] * (spatial_ndims + 3)))

                # `add_ones_channnel = True`
                input_layer2 = input_layer_cls(
                    1, 1, kernel_size=kernel_size, weight_init=tk.init.ones)

                # the pixelcnn resblock
                resblock_layer_cls = getattr(
                    tk.layers, f'PixelCNNResBlock{spatial_ndims}d')

                with pytest.raises(ValueError,
                                   match=r'`kernel_size` is required to be at '
                                         r'least 3'):
                    _ = resblock_layer_cls(1, 1, kernel_size=1)
                with pytest.raises(ValueError,
                                   match=r'`kernel_size` is required to be odd'):
                    _ = resblock_layer_cls(1, 1, kernel_size=[4, 3, 5][:spatial_ndims])

                resblock_layer = resblock_layer_cls(
                    1, 1, kernel_size=kernel_size, weight_init=tk.init.ones
                )
                resblock_layer = T.jit_compile(resblock_layer)

                with pytest.raises(Exception):
                    _ = resblock_layer([T.zeros([])] * (spatial_ndims - 1))
                with pytest.raises(Exception):
                    _ = resblock_layer([T.zeros([])] * (spatial_ndims + 1))

                # the down-sampling and up-sampling layer
                down_sample_cls = getattr(tk.layers, f'PixelCNNConv{spatial_ndims}d')
                down_sample_layer = down_sample_cls(1, 1, kernel_size, stride=2)
                down_sample_layer = T.jit_compile(down_sample_layer)

                down_sample_output_size = T.shape(down_sample_layer(
                    [T.zeros(make_conv_shape([1], 1, size))] * spatial_ndims)[0])
                up_sample_cls = getattr(tk.layers, f'PixelCNNConvTranspose{spatial_ndims}d')
                up_sample_layer = up_sample_cls(
                    1, 1, kernel_size, stride=2,
                    output_padding=tk.layers.get_deconv_output_padding(
                        input_size=[down_sample_output_size[a]
                                    for a in get_spatial_axis(spatial_ndims)],
                        output_size=size,
                        kernel_size=kernel_size,
                        stride=2,
                        padding='half',  # sum of the both sides == (kernel_size - 1) * dilation
                    )
                )
                up_sample_layer = T.jit_compile(up_sample_layer)

                # the output layer
                output_layer_cls = getattr(
                    tk.layers, f'PixelCNNOutput{spatial_ndims}d')
                output_layer = output_layer_cls()
                output_layer = T.jit_compile(output_layer)

                with pytest.raises(Exception,
                                   match=r'`len\(inputs\)` is expected to be .*'):
                    _ = output_layer([T.zeros([])] * (spatial_ndims - 1))
                with pytest.raises(Exception,
                                   match=r'`len\(inputs\)` is expected to be .*'):
                    _ = output_layer([T.zeros([])] * (spatial_ndims + 1))

                # ---- test the causality ----
                for pos, single_point in product(
                            iter_causal_test_pos(size),
                            (True, False)
                        ):
                    x = make_causal_test_input(
                        size, pos, single_point=single_point)
                    x_t = T.as_tensor(x)

                    # check the input layer output
                    outputs = input_layer(x_t)
                    ensure_stacks_causality(self, outputs, size, pos)

                    # check the final output
                    assert_allclose(output_layer(outputs), outputs[-1])

                    # check the resblock output
                    resblock_outputs = resblock_layer(outputs)
                    ensure_stacks_causality(self, resblock_outputs, size, pos)

                    outputs2 = resblock_outputs
                    for i in range(4):
                        outputs2 = resblock_layer(outputs2)
                    ensure_full_receptive_field(self, outputs2[-1], size, pos)

                    # check the down-sample and up-sample
                    down_sample_outputs = down_sample_layer(outputs)
                    up_sample_outputs = up_sample_layer(down_sample_outputs)
                    ensure_stacks_causality(self, up_sample_outputs, size, pos)

                # ---- test zero input on different input layers ----
                x_t = T.zeros(make_conv_shape([1], 1, size), dtype=T.float32)
                outputs = input_layer(x_t)
                assert_equal(
                    (np.abs(T.to_numpy(outputs[-1])) >= 1e-6).astype(np.int32),
                    x_t
                )
                outputs = input_layer2(x_t)
                assert_equal(
                    (np.abs(T.to_numpy(outputs[-1])) >= 1e-6).astype(np.int32),
                    make_causal_mask(size, [0] * spatial_ndims).astype(np.int32)
                )

    def test_pixelcnn_network(self):
        T.random.seed(1234)
        in_channels = 3
        out_channels = 5

        for size in [[15], [15, 13], [15, 13, 11]]:
            spatial_ndims = len(size)

            for kernel_size in [3, 5, [5, 3, 5][:spatial_ndims]]:
                # ---- construct the layers ----
                # the input layer
                input_layer_cls = getattr(
                    tk.layers, f'PixelCNNInput{spatial_ndims}d')
                input_layer = input_layer_cls(
                    in_channels, out_channels, kernel_size=kernel_size)
                input_layer = T.jit_compile(input_layer)

                # the pixelcnn layers
                resblock_layer_cls = getattr(
                    tk.layers, f'PixelCNNResBlock{spatial_ndims}d')
                conv_layer_cls = getattr(
                    tk.layers, f'PixelCNNConv{spatial_ndims}d')
                deconv_layer_cls = getattr(
                    tk.layers, f'PixelCNNConvTranspose{spatial_ndims}d')
                normalizer_cls = getattr(
                    tk.layers, f'ActNorm{spatial_ndims}d')
                dropout_cls = getattr(
                    tk.layers, f'Dropout{spatial_ndims}d')

                pixelcnn_layers = [
                    resblock_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        activation=tk.layers.LeakyReLU, normalizer=normalizer_cls,
                        merge_context1=_MyAddContext,
                        data_init=tk.init.StdDataInit,
                    ),
                    conv_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        stride=2, activation=tk.layers.Tanh, normalizer=normalizer_cls,
                        data_init=tk.init.StdDataInit,
                    ),
                    deconv_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        stride=2, activation=tk.layers.Tanh, normalizer=normalizer_cls,
                        data_init=tk.init.StdDataInit,
                    ),
                    resblock_layer_cls(
                        out_channels, out_channels, kernel_size=kernel_size,
                        activation=tk.layers.Sigmoid, normalizer=normalizer_cls,
                        dropout=0.5, merge_context1=_MyAddContext,
                        data_init=tk.init.StdDataInit,
                    ),
                ]
                pixelcnn_layers = [T.jit_compile(l) for l in pixelcnn_layers]

                # the pixelcnn network
                network_cls = getattr(tk.layers, f'PixelCNN{spatial_ndims}d')

                with pytest.raises(TypeError,
                                   match='`input_layer` must be an instance of'):
                    _ = network_cls(*pixelcnn_layers)

                network1 = network_cls(input_layer)
                network2 = network_cls(input_layer, pixelcnn_layers[0], pixelcnn_layers[1:])

                # ---- test the network ----
                x_t = T.random.randn(make_conv_shape([3], in_channels, size))
                context = [T.random.randn(make_conv_shape([3], out_channels, size))]

                _ = network2(T.random.randn(T.shape(x_t)))  # run the initializers
                tk.layers.set_train_mode(network1, False)
                tk.layers.set_train_mode(network2, False)

                # without context
                expected_outputs2 = expected_outputs1 = input_layer(x_t)
                expected_output1 = expected_outputs1[-1]

                for l in pixelcnn_layers:
                    expected_outputs2 = l(expected_outputs2)
                expected_output2 = expected_outputs2[-1]

                assert_allclose(network1(x_t), expected_output1, atol=1e-6, rtol=1e-4)
                assert_allclose(network2(x_t), expected_output2, atol=1e-6, rtol=1e-4)

                # with context
                expected_outputs2 = expected_outputs1 = input_layer(x_t)
                expected_output1 = expected_outputs1[-1]

                for l in pixelcnn_layers:
                    expected_outputs2 = l(expected_outputs2, context)
                expected_output2 = expected_outputs2[-1]

                assert_allclose(network1(x_t, context), expected_output1, atol=1e-6, rtol=1e-4)
                assert_allclose(network2(x_t, context), expected_output2, atol=1e-6, rtol=1e-4)
