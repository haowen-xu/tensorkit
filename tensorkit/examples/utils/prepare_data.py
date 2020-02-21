from typing import *

import mltk
import numpy as np

import tensorkit as tk
from .samplers import SamplerMapper

__all__ = [
    'get_mnist_streams'
]


def _scale_pixels_to_range(x, x_min, x_max):
    scale = (x_max - x_min) / 255.
    return np.minimum(np.maximum(x * scale + x_min, x_min), x_max)


def get_mnist_streams(batch_size: int,
                      test_batch_size: Optional[int] = None,
                      val_batch_size: Optional[int] = None,
                      val_portion: Optional[float] = None,
                      flatten: bool = False,
                      x_range: Optional[Tuple[float, float]] = None,
                      use_y: bool = True,
                      y_dtype: Union[str, np.dtype] = np.int32,
                      mapper: Optional[Callable[..., Tuple[np.ndarray, ...]]] = None,
                      fix_test_val_stream: bool = True,
                      as_tensor_stream: bool = True,
                      prefetch: Optional[int] = 5,
                      ) -> Tuple[mltk.DataStream, Optional[mltk.DataStream], mltk.DataStream]:
    # check the arguments
    if test_batch_size is None:
        test_batch_size = batch_size
    if val_batch_size is None:
        val_batch_size = batch_size

    # load data
    x_shape = [784] if flatten else [28, 28, 1]
    (train_x, train_y), (test_x, test_y) = mltk.data.load_mnist(
        x_shape=x_shape, x_dtype=np.float32, y_dtype=y_dtype)

    if not flatten:
        train_x = tk.utils.numpy_channel_from_last_to_default2d(train_x)
        test_x = tk.utils.numpy_channel_from_last_to_default2d(test_x)

    # scale pixels to the desired range
    if x_range is not None:
        train_x = _scale_pixels_to_range(train_x, *x_range)
        test_x = _scale_pixels_to_range(test_x, *x_range)

    # split train & valid set, and construct the streams
    def make_stream(arrays, **kwargs):
        if not use_y:
            arrays = (arrays[0],)
        fixed = kwargs.pop('fixed', False)
        ret = mltk.DataStream.arrays(arrays, **kwargs)
        if mapper is not None:
            preserve_shapes = isinstance(mapper, SamplerMapper)
            ret = ret.map(mapper, preserve_shapes=preserve_shapes)
            if fixed:
                ret = ret.to_arrays_stream()
        if as_tensor_stream:
            ret = tk.utils.as_tensor_stream(ret, prefetch=prefetch)
        return ret

    if val_portion is not None:
        (train_x, train_y), (val_x, val_y) = \
            mltk.utils.split_numpy_arrays([train_x, train_y], portion=val_portion)
        val_stream = make_stream([val_x, val_y], fixed=fix_test_val_stream,
                                 batch_size=val_batch_size)
    else:
        val_stream = None

    train_stream = make_stream(
        [train_x, train_y], batch_size=batch_size, shuffle=True,
        skip_incomplete=True)
    test_stream = make_stream([test_x, test_y], fixed=fix_test_val_stream,
                              batch_size=test_batch_size)

    # return the streams
    return train_stream, val_stream, test_stream
