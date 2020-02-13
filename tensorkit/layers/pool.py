from typing import *

from .. import tensor as T
from ..arg_check import *
from ..tensor import Tensor, reduce_mean, shape
from ..tensor.nn import *
from ..typing_ import *
from .core import *

__all__ = [
    'AvgPool1d', 'AvgPool2d', 'AvgPool3d',
    'MaxPool1d', 'MaxPool2d', 'MaxPool3d',
    'GlobalAvgPool1d', 'GlobalAvgPool2d', 'GlobalAvgPool3d',
]


# ---- average pooling ----
class AvgPoolNd(BaseSingleVariateLayer):

    __constants__ = ('kernel_size', 'stride', 'padding', 'count_padded_zeros')

    kernel_size: List[int]
    stride: List[int]
    padding: List[int]
    count_padded_zeros: bool

    def __init__(self,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Optional[Union[int, Sequence[int]]] = None,
                 padding: PaddingArgType = PaddingMode.DEFAULT,
                 count_padded_zeros: bool = T.nn.AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS):
        """
        Construct the average pooling layer.

        Args:
            kernel_size: The kernel size of average pooling.
            stride: The stride of average pooling.  Defaults to `kernel_size`.
            padding: The padding mode or size at each border of the input.
            count_padded_zeros: Whether or not to count the padded zeros
                when taking average of the input blocks?  Defaults to `False`.
        """
        spatial_ndims = self._get_spatial_ndims()
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        padding = validate_padding(
            padding, kernel_size, [1] * spatial_ndims, spatial_ndims)
        _symmetric_padding = maybe_as_symmetric_padding(padding)
        if _symmetric_padding is None:
            raise ValueError('Asymmetric padding is not supported.')

        if stride is not None:
            stride = validate_conv_size('stride', stride, spatial_ndims)
        else:
            stride = kernel_size

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = _symmetric_padding
        self.count_padded_zeros = count_padded_zeros

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class AvgPool1d(AvgPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    def _forward(self, input: Tensor) -> Tensor:
        return avg_pool1d(
            input, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, count_padded_zeros=self.count_padded_zeros,
        )


class AvgPool2d(AvgPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    def _forward(self, input: Tensor) -> Tensor:
        return avg_pool2d(
            input, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, count_padded_zeros=self.count_padded_zeros,
        )


class AvgPool3d(AvgPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    def _forward(self, input: Tensor) -> Tensor:
        return avg_pool3d(
            input, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, count_padded_zeros=self.count_padded_zeros,
        )


# ---- max pooling ----
class MaxPoolNd(BaseSingleVariateLayer):

    __constants__ = ('kernel_size', 'stride', 'padding')

    kernel_size: List[int]
    stride: List[int]
    padding: List[int]

    def __init__(self,
                 kernel_size: Union[int, Sequence[int]],
                 stride: Optional[Union[int, Sequence[int]]] = None,
                 padding: PaddingArgType = PaddingMode.DEFAULT):
        """
        Construct the max pooling layer.

        Args:
            kernel_size: The kernel size of average pooling.
            stride: The stride of average pooling.  Defaults to `kernel_size`.
            padding: The padding mode or size at each border of the input.
        """
        spatial_ndims = self._get_spatial_ndims()
        kernel_size = validate_conv_size('kernel_size', kernel_size, spatial_ndims)
        padding = validate_padding(
            padding, kernel_size, [1] * spatial_ndims, spatial_ndims)
        _symmetric_padding = maybe_as_symmetric_padding(padding)
        if _symmetric_padding is None:
            raise ValueError('Asymmetric padding is not supported.')

        if stride is not None:
            stride = validate_conv_size('stride', stride, spatial_ndims)
        else:
            stride = kernel_size

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = _symmetric_padding

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()


class MaxPool1d(MaxPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 1

    def _forward(self, input: Tensor) -> Tensor:
        return max_pool1d(
            input, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding
        )


class MaxPool2d(MaxPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 2

    def _forward(self, input: Tensor) -> Tensor:
        return max_pool2d(
            input, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding
        )


class MaxPool3d(MaxPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 3

    def _forward(self, input: Tensor) -> Tensor:
        return max_pool3d(
            input, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding
        )


# ---- global average pooling ----
class GlobalAvgPoolNd(BaseSingleVariateLayer):

    __constants__ = ('spatial_ndims', 'reduce_axis', 'keepdims')

    spatial_ndims: int
    """The number of spatial dimensions."""

    reduce_axis: List[int]
    """The axis to be reduced for global average pooling."""

    keepdims: bool
    """Whether or not to keep the reduced spatial dimensions?"""

    def __init__(self, keepdims: bool = False):
        spatial_ndims = self._get_spatial_ndims()
        reduce_axis = (
            T.int_range(-spatial_ndims - 1, -1) if T.IS_CHANNEL_LAST
            else T.int_range(-spatial_ndims, 0)
        )
        super().__init__()
        self.spatial_ndims = spatial_ndims
        self.keepdims = keepdims
        self.reduce_axis = reduce_axis

    def _get_spatial_ndims(self) -> int:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f'{self.__class__.__qualname__}(keepdims={self.keepdims})'

    def _forward(self, input: Tensor) -> Tensor:
        if len(input.shape) < self.spatial_ndims + 1:
            raise ValueError(
                '`rank(input)` is too low: expected to be at least '
                '{}d, but the input shape is {}.'.
                format(self.spatial_ndims + 1, shape(input))
            )
        return reduce_mean(input, axis=self.reduce_axis, keepdims=self.keepdims)


class GlobalAvgPool1d(GlobalAvgPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 1


class GlobalAvgPool2d(GlobalAvgPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 2


class GlobalAvgPool3d(GlobalAvgPoolNd):

    def _get_spatial_ndims(self) -> int:
        return 3

