from typing import *

from .. import tensor as T
from ..tensor import Tensor, argsort, index_select, flip_axis, float_scalar_like
from ..tensor.random import random_permutation
from ..layers import *
from .core import *

__all__ = [
    'FeatureShufflingFlow', 'FeatureShufflingFlow1d', 'FeatureShufflingFlow2d',
    'FeatureShufflingFlow3d',
    'FeatureReversingFlow', 'FeatureReversingFlow1d', 'FeatureReversingFlow2d',
    'FeatureReversingFlow3d',
]


class FeatureShufflingFlow(FeatureMappingFlow):
    """
    An invertible flow which shuffles the order of input features.

    This type of flow is proposed in (Kingma & Dhariwal, 2018), as a possible
    replacement to the alternating pattern of coupling layers proposed in
    (Dinh et al., 2016).
    """

    __constants__ = FeatureMappingFlow.__constants__ + (
        'num_features',
    )

    num_features: int

    def __init__(self,
                 num_features: int,
                 axis: int = -1, 
                 event_ndims: int = 1,
                 device: Optional[str] = None):
        """
        Construct a new :class:`FeatureShufflingFlow`.

        Args:
            num_features: The size of the feature axis.
            axis: The feature axis, to apply the transformation.
            event_ndims: Number of dimensions to be considered as the
                event dimensions.  `x.ndims - event_ndims == log_det.ndims`.
            device: The device where to place new tensors and variables.
        """
        super().__init__(axis=int(axis), event_ndims=event_ndims,
                         explicitly_invertible=True)
        self.num_features = num_features

        # initialize the permutation variable, and the inverse permutation
        permutation = random_permutation(num_features, dtype=T.index_dtype,
                                         device=device)
        inv_permutation = argsort(permutation)

        # register the permutation as layer parameter, such that it could be
        # persisted by Model checkpoint.
        add_parameter(self, 'permutation', permutation, requires_grad=False)
        add_parameter(self, 'inv_permutation', inv_permutation,
                      requires_grad=False)

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        if inverse:
            output = index_select(input, self.inv_permutation, axis=self.axis)
        else:
            output = index_select(input, self.permutation, axis=self.axis)
        output_log_det = input_log_det
        if compute_log_det and output_log_det is None:
            output_log_det = float_scalar_like(0., input)
        return output, output_log_det


class FeatureShufflingFlow1d(FeatureShufflingFlow):
    """1D convolutional channel shuffling flow."""

    def __init__(self, num_features: int):
        axis = -1 if T.IS_CHANNEL_LAST else -2
        super().__init__(num_features, axis=axis, event_ndims=2)


class FeatureShufflingFlow2d(FeatureShufflingFlow):
    """2D convolutional channel shuffling flow."""

    def __init__(self, num_features: int):
        axis = -1 if T.IS_CHANNEL_LAST else -3
        super().__init__(num_features, axis=axis, event_ndims=3)


class FeatureShufflingFlow3d(FeatureShufflingFlow):
    """3D convolutional channel shuffling flow."""

    def __init__(self, num_features: int):
        axis = -1 if T.IS_CHANNEL_LAST else -4
        super().__init__(num_features, axis=axis, event_ndims=4)


class FeatureReversingFlow(FeatureMappingFlow):
    """An invertible flow that reverses the order of features."""

    def __init__(self,
                 axis: int = -1,
                 event_ndims: int = 1):
        """
        Construct a new :class:`FeatureShufflingFlow`.

        Args:
            axis: The feature axis, to apply the transformation.
            event_ndims: Number of dimensions to be considered as the
                event dimensions.  `x.ndims - event_ndims == log_det.ndims`.
        """
        super().__init__(axis=int(axis), event_ndims=event_ndims,
                         explicitly_invertible=True)

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[Tensor, Optional[Tensor]]:
        output = flip_axis(input, axis=self.axis)
        output_log_det = input_log_det
        if compute_log_det and output_log_det is None:
            output_log_det = float_scalar_like(0., input)
        return output, output_log_det


class FeatureReversingFlow1d(FeatureReversingFlow):
    """1D convolutional channel reversing flow."""

    def __init__(self):
        axis = -1 if T.IS_CHANNEL_LAST else -2
        super().__init__(axis=axis, event_ndims=2)


class FeatureReversingFlow2d(FeatureReversingFlow):
    """2D convolutional channel reversing flow."""

    def __init__(self):
        axis = -1 if T.IS_CHANNEL_LAST else -3
        super().__init__(axis=axis, event_ndims=3)


class FeatureReversingFlow3d(FeatureReversingFlow):
    """3D convolutional channel reversing flow."""

    def __init__(self):
        axis = -1 if T.IS_CHANNEL_LAST else -4
        super().__init__(axis=axis, event_ndims=4)
