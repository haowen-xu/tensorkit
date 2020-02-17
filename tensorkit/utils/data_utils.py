import numpy as np

from tensorkit import tensor as T

__all__ = [
    'numpy_channel_from_last_to_first1d',
    'numpy_channel_from_last_to_first2d',
    'numpy_channel_from_last_to_first3d',

    'numpy_channel_from_first_to_last1d',
    'numpy_channel_from_first_to_last2d',
    'numpy_channel_from_first_to_last3d',

    'numpy_channel_from_last_to_default1d',
    'numpy_channel_from_last_to_default2d',
    'numpy_channel_from_last_to_default3d',

    'numpy_channel_from_default_to_last1d',
    'numpy_channel_from_default_to_last2d',
    'numpy_channel_from_default_to_last3d',
]


def numpy_channel_from_last_to_first_nd(input: np.ndarray,
                                        spatial_ndims: int
                                        ) -> np.ndarray:
    if len(input.shape) < spatial_ndims + 2:
        raise ValueError(
            f'`input` is expected to be at least {spatial_ndims + 2}d: '
            f'got `input.shape` {input.shape}.'
        )
    axis = list(range(len(input.shape)))
    transpose_axis = (
        axis[: -(spatial_ndims + 1)] + [-1] +
        [i for i in range(-spatial_ndims - 1, -1)]
    )
    return np.transpose(input, transpose_axis)


def numpy_channel_from_last_to_first1d(input: np.ndarray) -> np.ndarray:
    return numpy_channel_from_last_to_first_nd(input, 1)


def numpy_channel_from_last_to_first2d(input: np.ndarray) -> np.ndarray:
    return numpy_channel_from_last_to_first_nd(input, 2)


def numpy_channel_from_last_to_first3d(input: np.ndarray) -> np.ndarray:
    return numpy_channel_from_last_to_first_nd(input, 3)


def numpy_channel_from_first_to_last_nd(input: np.ndarray,
                                        spatial_ndims: int
                                        ) -> np.ndarray:
    if len(input.shape) < spatial_ndims + 2:
        raise ValueError(
            f'`input` is expected to be at least {spatial_ndims + 2}d: '
            f'got `input.shape` {input.shape}.'
        )
    axis = list(range(len(input.shape)))
    transpose_axis = (
        axis[: -(spatial_ndims + 1)] +
        [i for i in range(-spatial_ndims, 0)] +
        [-(spatial_ndims + 1)]
    )
    return np.transpose(input, transpose_axis)


def numpy_channel_from_first_to_last1d(input: np.ndarray) -> np.ndarray:
    return numpy_channel_from_first_to_last_nd(input, 1)


def numpy_channel_from_first_to_last2d(input: np.ndarray) -> np.ndarray:
    return numpy_channel_from_first_to_last_nd(input, 2)


def numpy_channel_from_first_to_last3d(input: np.ndarray) -> np.ndarray:
    return numpy_channel_from_first_to_last_nd(input, 3)


if T.IS_CHANNEL_LAST:
    numpy_channel_from_last_to_default1d = \
        numpy_channel_from_last_to_default2d = \
        numpy_channel_from_last_to_default3d = \
        numpy_channel_from_default_to_last1d = \
        numpy_channel_from_default_to_last2d = \
        numpy_channel_from_default_to_last3d = \
        (lambda x: x)
else:
    numpy_channel_from_last_to_default1d = numpy_channel_from_last_to_first1d
    numpy_channel_from_last_to_default2d = numpy_channel_from_last_to_first2d
    numpy_channel_from_last_to_default3d = numpy_channel_from_last_to_first3d
    numpy_channel_from_default_to_last1d = numpy_channel_from_first_to_last1d
    numpy_channel_from_default_to_last2d = numpy_channel_from_first_to_last2d
    numpy_channel_from_default_to_last3d = numpy_channel_from_first_to_last3d
