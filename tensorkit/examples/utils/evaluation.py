import os
from typing import *

import imageio
import mltk
import numpy as np

from tensorkit import tensor as T


__all__ = [
    'save_images_collection',
]


def save_images_collection(images: Sequence[np.ndarray],
                           filename: str,
                           grid_size: Tuple[int, int],
                           border_size: int = 0,
                           value_range: Tuple[float, float] = (0., 255.),
                           channels_last: bool = mltk.NOT_SET):
    """
    Save a collection of images as a large image, arranged in grid.

    Args:
        images: The images collection.  Each element should be a Numpy array,
            in the shape of ``(H, W)``, ``(H, W, C)`` (if `channels_last` is
            :obj:`True`) or ``(C, H, W)``.
        filename: The target filename.
        grid_size: The number of rows and columns of the grid.
        border_size: Width of the border between images.
        value_range: The range of the input `images` value ranges.
        channels_last: Whether or not the channel axis of the `images` is
            the last axis?  If not specified, use ``T.IS_CHANNEL_LAST``.
    """
    # check the arguments
    def normalize_images(img):
        # normalize the shape
        if len(img.shape) == 2:
            img = np.reshape(img, img.shape + (1,))
        elif len(images[0].shape) == 3:
            if img.shape[2 if channels_last else 0] not in (1, 3, 4):
                raise ValueError('Unexpected image shape: {!r}'.
                                 format(img.shape))
            if not channels_last:
                img = np.transpose(img, (1, 2, 0))
        else:
            raise ValueError('Unexpected image shape: {!r}'.format(img.shape))

        # normalize the values
        v_min, v_max = value_range
        img = np.clip((img - v_min) * (255. / (v_max - v_min)), 0., 255.).astype(np.uint8)

        return img

    if channels_last is mltk.NOT_SET:
        channels_last = T.IS_CHANNEL_LAST

    images = [normalize_images(img) for img in images]
    h, w = images[0].shape[:2]
    rows, cols = grid_size[0], grid_size[1]
    buf_h = rows * h + (rows - 1) * border_size
    buf_w = cols * w + (cols - 1) * border_size

    # copy the images to canvas
    n_channels = images[0].shape[2]
    buf = np.zeros((buf_h, buf_w, n_channels), dtype=images[0].dtype)
    for j in range(rows):
        for i in range(cols):
            img = images[j * cols + i]
            buf[j * (h + border_size): (j + 1) * h + j * border_size,
                i * (w + border_size): (i + 1) * w + i * border_size,
                :] = img[:, :, :]

    # save the image
    if n_channels == 1:
        buf = np.reshape(buf, (buf_h, buf_w))
    parent_dir = os.path.split(os.path.abspath(filename))[0]
    if not os.path.isdir(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    imageio.imwrite(filename, buf)
