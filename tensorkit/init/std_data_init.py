from typing import List

from .. import tensor as T
from ..layers import *
from .core import *

__all__ = ['StdDataInit']

from ..backend import Module, Tensor


class StdDataInit(DataDependentInitializer):
    """
    A data-dependent initializer that standardizes the feature (channel) axis
    of the output of a :class:`CoreLinear` layer.
    """

    epsilon: float

    def __init__(self, epsilon: float = T.EPSILON):
        super().__init__()
        self.epsilon = epsilon

    def _call(self, layer: Module, inputs: List[Tensor]) -> None:
        if T.is_jit_layer(layer):
            raise TypeError(f'JIT compiled layer is not supported: got {layer!r}')
        if not isinstance(layer, CoreLinear):
            raise TypeError(f'`layer` is not a core linear layer: got {layer!r}')
        if len(inputs) != 1:
            raise ValueError(f'`inputs` must have exactly one input tensor: got '
                             f'{inputs!r}')

        # get the weight and bias
        weight = layer.weight_store()
        bias = layer.bias_store() if layer.bias_store is not None else None
        is_conv_transpose = isinstance(layer, (LinearConvTranspose1d,
                                               LinearConvTranspose2d,
                                               LinearConvTranspose3d))

        # recognize the spatial ndims, the channel axis, and the receptive field
        weight_shape = T.shape(weight)
        weight_rank = len(weight_shape)

        if weight_rank == 2:
            spatial_ndims = 0
        elif weight_rank in (3, 4, 5):
            spatial_ndims = weight_rank - 2
        else:  # pragma: no cover
            raise TypeError(f'Unsupported layer: {layer!r}')

        channel_axis = -1 if T.IS_CHANNEL_LAST else -(spatial_ndims + 1)

        # get the output
        output = layer(inputs[0])

        # compute the current feature mean & var
        reduce_axis = [a for a in range(-T.rank(output), 0)
                       if a != channel_axis]
        out_mean, out_var = T.calculate_mean_and_var(
            output, axis=reduce_axis, keepdims=False)
        out_std = T.sqrt(
            T.maximum(
                out_var,
                T.as_tensor_backend(self.epsilon, dtype=out_var.dtype)
            )
        )
        weight_scale = out_std

        # standardize the output by scaling the weight and centering the bias
        with T.no_grad():
            # weight
            if T.backend_name == 'PyTorch':
                layer.weight_store.set(
                    weight / T.reshape(
                        weight_scale,
                        ([-1] + [1] * spatial_ndims if is_conv_transpose
                         else [-1] + [1] * (spatial_ndims + 1))
                    )
                )
            else:  # pragma: no cover
                raise RuntimeError(f'Backend `{T.backend_name}` not supported.')

            # bias
            if bias is not None:
                bias_shape = T.shape(bias)
                layer.bias_store.set(
                    bias + (
                        T.reshape(-out_mean, bias_shape) /
                        T.reshape(weight_scale, bias_shape)
                    )
                )
