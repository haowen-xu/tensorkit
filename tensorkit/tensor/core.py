from typing import *

from . import backend as B

__all__ = [
    'broadcast_shape', 'broadcast_to', 'explicit_broadcast',
]


# ---- shape utils ----
def broadcast_shape(x: B.ShapeArgType, y: B.ShapeArgType) -> B.ShapeTuple:
    x = tuple(x)
    y = tuple(y)
    common_len = min(len(x), len(y))

    right = []
    for a, b in zip(x[-common_len:], y[-common_len:]):
        if a == 1:
            right.append(b)
        elif b == 1:
            right.append(a)
        elif a != b:
            raise ValueError(f'Shape x and y cannot broadcast against '
                             f'each other: {x} vs {y}.')
        else:
            right.append(a)

    left = x[:-common_len] or y[:-common_len]
    return left + tuple(right)


def _broadcast_to_internal(t, t_shape, out_shape):
    if len(t_shape) < len(out_shape):
        t_shape = (1,) * (len(out_shape) - len(t_shape)) + t_shape
        t = B.reshape(t, t_shape)
    t_repeats = tuple(b if a == 1 else 1
                      for a, b in zip(t_shape, out_shape))
    if any(s != 1 for s in t_repeats):
        t = B.tile(t, t_repeats)
    return t


def broadcast_to(x: B.TensorLike, shape: B.ShapeArgType) -> B.Tensor:
    x = B.as_tensor(x)
    x_shape = B.shape(x)
    shape = tuple(shape)

    # check whether or not x can broadcast to shape
    can_broadcast = len(x_shape) <= len(shape)
    if can_broadcast:
        for a, b in zip(reversed(x_shape), shape):
            if a != 1 and a != b:
                can_broadcast = False
                break
    if not can_broadcast:
        raise ValueError(f'`x` cannot be broadcast to `shape`: '
                         f'shape(x) {x_shape} vs shape {shape}')

    return _broadcast_to_internal(x, x_shape, shape)


def explicit_broadcast(x: B.TensorLike,
                       y: B.TensorLike
                       ) -> Tuple[B.Tensor, B.Tensor]:
    x = B.as_tensor(x)
    y = B.as_tensor(y)
    x_shape = B.shape(x)
    y_shape = B.shape(y)
    out_shape = broadcast_shape(x_shape, y_shape)
    x = _broadcast_to_internal(x, x_shape, out_shape)
    y = _broadcast_to_internal(y, y_shape, out_shape)
    return x, y
