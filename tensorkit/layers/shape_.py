from typing import *

from ..tensor import (Tensor, Module, shape, rank, flatten_to_ndims,
                      unflatten_from_ndims, pad)
from ..tensor.nn import *
from .core import *

__all__ = [
    'FlattenToNDims',
    'ConstantPad', 'ConstantPad1d', 'ConstantPad2d', 'ConstantPad3d',
    'ChannelFirstToLast1d', 'ChannelFirstToLast2d', 'ChannelFirstToLast3d',
    'ChannelLastToFirst1d', 'ChannelLastToFirst2d', 'ChannelLastToFirst3d',
]


# ---- FlattenToNDims ----
class FlattenToNDims(BaseSingleVariateLayer):

    __constants__ = ('layer', 'ndims')

    layer: Module
    ndims: int

    def __init__(self, layer: Module, ndims: int):
        super().__init__()
        self.layer = layer
        self.ndims = ndims

    def _forward(self, input: Tensor) -> Tensor:
        # validate the shape of input
        input_rank = rank(input)
        expected_rank = self.ndims
        if input_rank < expected_rank:
            raise ValueError(
                '`rank(input)` is too low: expected to be at least '
                '{}-dimensional, but the input shape is {}.'.
                format(expected_rank, shape(input))
            )

        # flatten, get output from the layer, and then unflatten
        output, front_shape = flatten_to_ndims(input, expected_rank)
        output = self.layer(output)
        return unflatten_from_ndims(output, front_shape)


# ---- pad ----
class ConstantPad(BaseSingleVariateLayer):

    __constants__ = ('padding', 'value')

    padding: List[Tuple[int, int]]
    value: float

    def __init__(self,
                 padding: Sequence[Union[int, Tuple[int, int]]],
                 value: float = 0.):
        # check the arguments
        def check_int_tuple(t):
            if not hasattr(t, '__iter__'):
                v = int(t)
                return v, v
            else:
                t = tuple(map(int, t))
                if len(t) != 2:
                    raise ValueError(f'`padding` must be a sequence of int '
                                     f'or tuple of (int, int): got {padding!r}')
                return t

        padding = list(map(check_int_tuple, padding))

        # construct the layer
        super().__init__()
        self.padding = padding
        self.value = value

    def _forward(self, input: Tensor) -> Tensor:
        return pad(input, self.padding, value=self.value)


class ConstantPadNd(ConstantPad):
    """
    `ConstantPad` specialized for padding the spatial dimensions of
    convolutional inputs.
    """

    def __init__(self,
                 *spatial_padding: Union[int, Tuple[int, int]],
                 value: float = 0.):
        spatial_ndims = self._get_spatial_ndims()

        if len(spatial_padding) == 1:
            padding = spatial_padding * spatial_ndims
        elif len(spatial_padding) != spatial_ndims:
            if spatial_ndims == 1:
                raise ValueError(
                    f'`{self.__class__.__qualname__}` requires 1 spatial padding '
                    f'to be specified, but got {len(spatial_padding)} paddings: '
                    f'{", ".join([str(t) for t in spatial_padding])}')
            else:
                raise ValueError(
                    f'`{self.__class__.__qualname__}` requires 1 or '
                    f'{spatial_ndims} spatial paddings to be specified, but '
                    f'got {len(spatial_padding)} paddings: '
                    f'{", ".join([str(t) for t in spatial_padding])}')
        else:
            padding = list(spatial_padding)

        super().__init__(padding, value)

    def _get_spatial_ndims(self):
        raise NotImplementedError()


class ConstantPad1d(ConstantPadNd):

    def _get_spatial_ndims(self):
        return 1


class ConstantPad2d(ConstantPadNd):

    def _get_spatial_ndims(self):
        return 2


class ConstantPad3d(ConstantPadNd):

    def _get_spatial_ndims(self):
        return 3


# ---- channel swap ----
class ChannelFirstToLast1d(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return channel_first_to_last1d(input)


class ChannelFirstToLast2d(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return channel_first_to_last2d(input)


class ChannelFirstToLast3d(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return channel_first_to_last3d(input)


class ChannelLastToFirst1d(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return channel_last_to_first1d(input)


class ChannelLastToFirst2d(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return channel_last_to_first2d(input)


class ChannelLastToFirst3d(BaseSingleVariateLayer):

    def _forward(self, input: Tensor) -> Tensor:
        return channel_last_to_first3d(input)
