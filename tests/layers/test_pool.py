import unittest
from itertools import product

import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *
from tests.ops import *


class PoolTestCase(unittest.TestCase):

    def test_AvgPool_and_MaxPool(self):
        T.random.seed(1234)

        def is_valid_padding(padding, kernel_size):
            for p, k in zip(padding, kernel_size):
                if isinstance(p, int):
                    p = (p,)
                if all(t >= k / 2. for t in p):
                    return False
            return True

        for pool_type, spatial_ndims in product(('avg', 'max'), (1, 2, 3)):
            x = T.random.randn(make_conv_shape([3], 4, [15, 14, 13][: spatial_ndims]))
            cls_name = f'{pool_type.capitalize()}Pool{spatial_ndims}d'
            layer_factory = getattr(tk.layers, cls_name)

            with pytest.raises(ValueError,
                               match='Asymmetric padding is not supported'):
                _ = layer_factory(
                    kernel_size=1,
                    padding=[(3, 2), (2, 1), (1, 0)][:spatial_ndims])

            for kernel_size, stride, padding, count_padded_zeros in product(
                        (1, [3, 2, 1][:spatial_ndims]),
                        (None, 1, [3, 2, 1][:spatial_ndims]),
                        (0, 1, [(3, 3), 2, (1, 1)][:spatial_ndims], 'none'),
                        (True, False)
                    ):
                fn = getattr(T.nn, f'{pool_type}_pool{spatial_ndims}d')

                if pool_type == 'avg':
                    count_padded_zeros_arg = {'count_padded_zeros': count_padded_zeros}
                elif not count_padded_zeros:
                    continue
                else:
                    count_padded_zeros_arg = {}

                layer = layer_factory(
                    kernel_size=kernel_size, stride=stride, padding=padding,
                    **count_padded_zeros_arg
                )

                if isinstance(kernel_size, int):
                    kernel_size = [kernel_size] * spatial_ndims

                if isinstance(stride, int):
                    stride = [stride] * spatial_ndims
                elif stride is None:
                    stride = kernel_size

                if isinstance(padding, int):
                    padding = [padding] * spatial_ndims
                elif padding == 'none':
                    padding = [0] * spatial_ndims

                if not is_valid_padding(padding, kernel_size):
                    continue

                if pool_type == 'avg':
                    self.assertEqual(
                        repr(layer),
                        f'{cls_name}(kernel_size={kernel_size}, stride={stride}, '
                        f'padding={padding}, count_padded_zeros={count_padded_zeros})'
                    )
                else:
                    self.assertEqual(
                        repr(layer),
                        f'{cls_name}(kernel_size={kernel_size}, stride={stride}, '
                        f'padding={padding})'
                    )

                layer = T.jit_compile(layer)
                assert_allclose(
                    layer(x),
                    fn(x, kernel_size=kernel_size, stride=stride,
                       padding=padding, **count_padded_zeros_arg)
                )

    def test_GlobalAvgPool(self):
        for spatial_ndims, keepdims in product((1, 2, 3), (True, False)):
            if T.IS_CHANNEL_LAST:
                reduce_axis = tuple(range(-spatial_ndims - 1, -1))
            else:
                reduce_axis = tuple(range(-spatial_ndims, 0))

            def fn(arr):
                return np.mean(arr, axis=reduce_axis, keepdims=keepdims)

            layer_factory = getattr(tk.layers, f'GlobalAvgPool{spatial_ndims}d')

            layer = layer_factory(keepdims=keepdims)
            self.assertEqual(
                repr(layer),
                f'GlobalAvgPool{spatial_ndims}d(keepdims={keepdims})'
            )

            layer = T.jit_compile(layer)
            x = T.random.randn(make_conv_shape([4, 5], 6, [7, 8, 9][:spatial_ndims]))
            assert_allclose(layer(x), fn(T.to_numpy(x)), rtol=1e-4, atol=1e-6)

            with pytest.raises(Exception, match=r'`rank\(input\)` is too low'):
                _ = layer(T.random.randn([5]))
