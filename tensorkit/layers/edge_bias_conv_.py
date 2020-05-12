from .. import tensor as T
from ..tensor import Tensor, shape, concat, ones_like
from .core import *

__all__ = ['AddOnesChannel1d', 'AddOnesChannel2d', 'AddOnesChannel3d']


class AddOnesChannelNd(BaseLayer):

    __constants__ = ('_channel_axis', '_spatial_ndims')

    _channel_axis: int
    _spatial_ndims: int

    def __init__(self):
        super().__init__()
        spatial_ndims = self._get_spatial_ndims()
        self._spatial_ndims = spatial_ndims
        if T.IS_CHANNEL_LAST:
            self._channel_axis = -1
        else:
            self._channel_axis = -(spatial_ndims + 1)

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def forward(self, input: Tensor) -> Tensor:
        channel_shape = shape(input)
        channel_shape[self._channel_axis] = 1

        return concat([input, ones_like(input, shape=channel_shape)],
                      axis=self._channel_axis)


class AddOnesChannel1d(AddOnesChannelNd):

    def _get_spatial_ndims(self) -> int:
        return 1


class AddOnesChannel2d(AddOnesChannelNd):

    def _get_spatial_ndims(self) -> int:
        return 2


class AddOnesChannel3d(AddOnesChannelNd):

    def _get_spatial_ndims(self) -> int:
        return 3
