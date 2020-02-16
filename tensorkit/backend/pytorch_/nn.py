from typing import *

import torch

from .core import *

__all__ = [
    # constants
    'LEAKY_RELU_DEFAULT_SLOPE', 'AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS',

    # activation functions
    'relu', 'leaky_relu',
    'sigmoid', 'log_sigmoid',
    'softmax', 'log_softmax',
    'softplus',

    # cross entropy functions
    'binary_cross_entropy_with_logits', 'cross_entropy_with_logits',
    'sparse_cross_entropy_with_logits',

    # conv shape utils
    'channel_first_to_last1d', 'channel_first_to_last2d', 'channel_first_to_last3d',
    'channel_last_to_first1d', 'channel_last_to_first2d', 'channel_last_to_first3d',
    'space_to_depth1d', 'space_to_depth2d', 'space_to_depth3d',
    'depth_to_space1d', 'depth_to_space2d', 'depth_to_space3d',

    # pooling functions
    'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
    'max_pool1d', 'max_pool2d', 'max_pool3d',
]


# ---- activation functions ----
LEAKY_RELU_DEFAULT_SLOPE = 0.01
AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS = False


@jit
def relu(x: Tensor) -> Tensor:
    return torch.relu(x)


@jit
def leaky_relu(x: Tensor,
               negative_slope: float = LEAKY_RELU_DEFAULT_SLOPE
               ) -> Tensor:
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)


@jit
def sigmoid(x: Tensor) -> Tensor:
    return torch.sigmoid(x)


@jit
def log_sigmoid(x: Tensor) -> Tensor:
    # using `neg_x` and `pos_x` separately can avoid having NaN or Infinity
    # on either of the path.
    neg_x = torch.min(x, torch.as_tensor(0., dtype=x.dtype))
    pos_x = torch.max(x, torch.as_tensor(0., dtype=x.dtype))
    return torch.where(
        x < 0.,
        neg_x - log1p(exp(neg_x)),  # log(exp(x) / (1 + exp(x)))
        -log1p(exp(-pos_x))     # log(1 / (1 + exp(-x)))
    )


@jit
def softmax(x: Tensor, axis: int = -1) -> Tensor:
    return torch.softmax(x, dim=axis)


@jit
def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    return torch.log_softmax(x, dim=axis)


@jit
def softplus(x: Tensor) -> Tensor:
    return torch.nn.functional.softplus(x)


# ---- cross entropy functions ----
@jit
def binary_cross_entropy_with_logits(logits: Tensor,
                                     labels: Tensor,
                                     reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                                     negative: bool = False) -> Tensor:
    if labels.dtype != logits.dtype:
        labels = labels.to(dtype=logits.dtype)
    logits, labels = explicit_broadcast(logits, labels)
    ret = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction=reduction)
    if negative:
        ret = -ret
    return ret


@jit
def cross_entropy_with_logits(logits: Tensor,
                              labels: Tensor,
                              reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                              negative: bool = False) -> Tensor:

    if logits.shape[:-1] != labels.shape:
        logits_shape = list(logits.shape)
        labels_shape = list(labels.shape)
        b_shape = broadcast_shape(logits_shape[:-1], labels_shape)
        logits = broadcast_to(logits, b_shape + logits_shape[-1:])
        labels = broadcast_to(labels, b_shape)

    if len(logits.shape) < 2 or len(labels.shape) < 1:
        raise ValueError('`logits` must be at least 2d, and `labels` must '
                         'be at least 1d: logits.shape is {}, while '
                         'labels.shape is {}.'.
                         format(logits.shape, labels.shape))

    logits, front_shape = flatten_to_ndims(logits, 2)
    labels, _ = flatten_to_ndims(labels, 1)
    if labels.dtype != torch.int64:
        labels = labels.to(torch.int64)

    ret = torch.nn.functional.cross_entropy(
        logits, labels, reduction=reduction)
    if negative:
        ret = -ret

    if reduction == 'none':
        ret = unflatten_from_ndims(ret, front_shape)
    return ret


@jit
def sparse_cross_entropy_with_logits(logits: Tensor,
                                     labels: Tensor,
                                     reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                                     negative: bool = False) -> Tensor:
    if reduction != 'none' and reduction != 'sum' and reduction != 'mean':
        raise ValueError('`reduce` is not one of "none", "sum" and '
                         '"mean": got {}'.format(reduction))

    logits, labels = explicit_broadcast(logits, labels)
    logits_shape = logits.shape
    labels_shape = labels.shape

    if len(logits_shape) < 2 or len(labels_shape) < 2:
        raise ValueError('`logits` and `labels` must be at least 2d: '
                         'logits.shape is {}, while labels.shape '
                         'is {}.'.format(logits.shape, labels.shape))

    log_sum_exp_logits = torch.logsumexp(logits, dim=-1, keepdim=True)

    if negative:
        ret = labels * (logits - log_sum_exp_logits)
    else:
        ret = labels * (log_sum_exp_logits - logits)

    if reduction == 'sum':
        ret = torch.sum(ret)
    elif reduction == 'mean':
        ret = torch.mean(torch.sum(ret, dim=-1))
    else:
        ret = torch.sum(ret, dim=-1)

    return ret


# ---- convolution shape utils ----
@jit
def channel_first_to_last_nd(input: Tensor, spatial_ndims: int) -> Tensor:
    input_rank = input.dim()
    if input.dim() < spatial_ndims + 1:
        raise ValueError('`input` must be at-least {}d: got input shape `{}`'.
                         format(spatial_ndims + 1, shape(input)))
    return input.permute(
        int_range(0, input_rank - spatial_ndims - 1) +
        int_range(-spatial_ndims, 0) +
        [-(spatial_ndims + 1)]  # transpose the channel axis to the last
    )


@jit
def channel_first_to_last1d(input: Tensor) -> Tensor:
    return channel_first_to_last_nd(input, 1)


@jit
def channel_first_to_last2d(input: Tensor) -> Tensor:
    return channel_first_to_last_nd(input, 2)


@jit
def channel_first_to_last3d(input: Tensor) -> Tensor:
    return channel_first_to_last_nd(input, 3)


@jit
def channel_last_to_first_nd(input: Tensor, spatial_ndims: int) -> Tensor:
    input_rank = input.dim()
    if input.dim() < spatial_ndims + 1:
        raise ValueError('`input` must be at-least {}d: got input shape `{}`'.
                         format(spatial_ndims + 1, shape(input)))
    return input.permute(
        int_range(0, input_rank - spatial_ndims - 1) +
        [-1] +  # transpose the channel axis to the first
        int_range(-(spatial_ndims + 1), -1)
    )


@jit
def channel_last_to_first1d(input: Tensor) -> Tensor:
    return channel_last_to_first_nd(input, 1)


@jit
def channel_last_to_first2d(input: Tensor) -> Tensor:
    return channel_last_to_first_nd(input, 2)


@jit
def channel_last_to_first3d(input: Tensor) -> Tensor:
    return channel_last_to_first_nd(input, 3)


@jit
def space_to_depth1d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = shape(input)
    input_rank = len(input_shape)
    if input_rank < 3:
        raise ValueError('`input` must be at-least 3d: got input shape `{}`'.
                         format(input_shape))

    L = input_shape[-1]
    if L % block_size != 0:
        raise ValueError('Not all dimensions of the `spatial_shape` are '
                         'multiples of `block_size`: `spatial_shape` is '
                         '`{}`, while `block_size` is {}.'.
                         format(input_shape[-1:], block_size))

    # do transformation
    batch_shape = input_shape[: -2]
    channel_size = input_shape[-2]
    L_reduced = L // block_size

    output = input.reshape(batch_shape + [channel_size, L_reduced, block_size])
    output = output.permute(
        int_range(0, len(batch_shape)) +
        [-1, -3, -2]
    )
    output = output.reshape(batch_shape + [-1, L_reduced])

    return output


@jit
def space_to_depth2d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = shape(input)
    input_rank = len(input_shape)
    if input_rank < 4:
        raise ValueError('`input` must be at-least 4d: got input shape `{}`'.
                         format(input_shape))

    H = input_shape[-2]
    W = input_shape[-1]
    if H % block_size != 0 or W % block_size != 0:
        raise ValueError('Not all dimensions of the `spatial_shape` are '
                         'multiples of `block_size`: `spatial_shape` is '
                         '`{}`, while `block_size` is {}.'.
                         format(input_shape[-2:], block_size))

    # do transformation
    batch_shape = input_shape[: -3]
    channel_size = input_shape[-3]
    H_reduced = H // block_size
    W_reduced = W // block_size

    output = input.reshape(
        batch_shape +
        [channel_size, H_reduced, block_size, W_reduced, block_size]
    )
    output = output.permute(
        int_range(0, len(batch_shape)) +
        [-3, -1, -5, -4, -2]
    )
    output = output.reshape(batch_shape + [-1, H_reduced, W_reduced])

    return output


@jit
def space_to_depth3d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = shape(input)
    input_rank = len(input_shape)
    if input_rank < 5:
        raise ValueError('`input` must be at-least 5d: got input shape `{}`'.
                         format(input_shape))

    D = input_shape[-3]
    H = input_shape[-2]
    W = input_shape[-1]
    if D % block_size != 0 or H % block_size != 0 or W % block_size != 0:
        raise ValueError('Not all dimensions of the `spatial_shape` are '
                         'multiples of `block_size`: `spatial_shape` is '
                         '`{}`, while `block_size` is {}.'.
                         format(input_shape[-3:], block_size))

    # do transformation
    batch_shape = input_shape[: -4]
    channel_size = input_shape[-4]
    D_reduced = D // block_size
    H_reduced = H // block_size
    W_reduced = W // block_size

    output = input.reshape(
        batch_shape +
        [channel_size, D_reduced, block_size, H_reduced, block_size,
         W_reduced, block_size]
    )
    output = output.permute(
        int_range(0, len(batch_shape)) +
        [-5, -3, -1, -7, -6, -4, -2]
    )
    output = output.reshape(batch_shape + [-1, D_reduced, H_reduced, W_reduced])

    return output


@jit
def depth_to_space1d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = shape(input)
    input_rank = len(input_shape)
    if input_rank < 3:
        raise ValueError('`input` must be at-least 3d: got input shape `{}`'.
                         format(input_shape))

    channel_size = input_shape[-2]
    if channel_size % block_size != 0:
        raise ValueError('`channel_size` is not multiples of `block_size`: '
                         '`channel_size` is `{}`, while `block_size` is {}.'.
                         format(channel_size, block_size))

    # do transformation
    batch_shape = input_shape[: -2]
    L = input_shape[-1]

    output = input.reshape(batch_shape + [block_size, -1, L])
    output = output.permute(
        int_range(0, len(batch_shape)) +
        [-2, -1, -3]
    )
    output = output.reshape(batch_shape + [-1, L * block_size])

    return output


@jit
def depth_to_space2d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = shape(input)
    input_rank = len(input_shape)
    if input_rank < 4:
        raise ValueError('`input` must be at-least 4d: got input shape `{}`'.
                         format(input_shape))

    channel_size = input_shape[-3]
    if channel_size % (block_size * block_size) != 0:
        raise ValueError('`channel_size` is not multiples of '
                         '`block_size * block_size`: '
                         '`channel_size` is `{}`, while `block_size` is {}.'.
                         format(channel_size, block_size))

    # do transformation
    batch_shape = input_shape[: -3]
    H = input_shape[-2]
    W = input_shape[-1]

    output = input.reshape(batch_shape + [block_size, block_size, -1, H, W])
    output = output.permute(
        int_range(0, len(batch_shape)) +
        [-3, -2, -5, -1, -4]
    )
    output = output.reshape(batch_shape + [-1, H * block_size, W * block_size])

    return output


@jit
def depth_to_space3d(input: Tensor, block_size: int) -> Tensor:
    # check the arguments
    input_shape = shape(input)
    input_rank = len(input_shape)
    if input_rank < 5:
        raise ValueError('`input` must be at-least 5d: got input shape `{}`'.
                         format(input_shape))

    channel_size = input_shape[-4]
    if channel_size % (block_size * block_size * block_size) != 0:
        raise ValueError('`channel_size` is not multiples of '
                         '`block_size * block_size * block_size`: '
                         '`channel_size` is `{}`, while `block_size` is {}.'.
                         format(channel_size, block_size))

    # do transformation
    batch_shape = input_shape[: -4]
    D = input_shape[-3]
    H = input_shape[-2]
    W = input_shape[-1]

    output = input.reshape(
        batch_shape + [block_size, block_size, block_size, -1, D, H, W])
    output = output.permute(
        int_range(0, len(batch_shape)) +
        [-4, -3, -7, -2, -6, -1, -5]
    )
    output = output.reshape(
        batch_shape + [-1, D * block_size, H * block_size, W * block_size])

    return output


# ---- pooling functions ----
@jit
def avg_pool1d(input: Tensor,
               kernel_size: List[int],
               stride: List[int],
               padding: List[int],
               count_padded_zeros: bool = AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS):
    return torch.nn.functional.avg_pool1d(
        input, kernel_size=kernel_size, stride=stride, padding=padding,
        count_include_pad=count_padded_zeros,
    )


@jit
def avg_pool2d(input: Tensor,
               kernel_size: List[int],
               stride: List[int],
               padding: List[int],
               count_padded_zeros: bool = AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS):
    return torch.nn.functional.avg_pool2d(
        input, kernel_size=kernel_size, stride=stride, padding=padding,
        count_include_pad=count_padded_zeros,
    )


@jit
def avg_pool3d(input: Tensor,
               kernel_size: List[int],
               stride: List[int],
               padding: List[int],
               count_padded_zeros: bool = AVG_POOL_DEFAULT_COUNT_PADDED_ZEROS):
    return torch.nn.functional.avg_pool3d(
        input, kernel_size=kernel_size, stride=stride, padding=padding,
        count_include_pad=count_padded_zeros,
    )


@jit
def max_pool1d(input: Tensor,
               kernel_size: List[int],
               stride: List[int],
               padding: List[int]):
    return torch.nn.functional.max_pool1d(
        input, kernel_size=kernel_size, stride=stride, padding=padding)


@jit
def max_pool2d(input: Tensor,
               kernel_size: List[int],
               stride: List[int],
               padding: List[int]):
    return torch.nn.functional.max_pool2d(
        input, kernel_size=kernel_size, stride=stride, padding=padding)


@jit
def max_pool3d(input: Tensor,
               kernel_size: List[int],
               stride: List[int],
               padding: List[int]):
    return torch.nn.functional.max_pool3d(
        input, kernel_size=kernel_size, stride=stride, padding=padding)
