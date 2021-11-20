import math
from typing import *

import torch

from .core import *
from .dtypes import categorical_dtype
from .nn import *
from ...settings_ import settings

__all__ = [
    'seed', 'set_deterministic',

    # uniform
    'rand', 'uniform', 'randint',

    # shuffle and random permutation
    'shuffle', 'random_permutation',

    # normal
    'randn', 'randn_log_pdf',
    'normal', 'normal_log_pdf',

    # bernoulli
    'bernoulli_logits_to_probs', 'bernoulli_probs_to_logits',
    'bernoulli', 'bernoulli_log_prob',

    # categorical
    'categorical_logits_to_probs', 'categorical_probs_to_logits',
    'categorical', 'categorical_log_prob',
    'one_hot_categorical', 'one_hot_categorical_log_prob',

    # initializers
    'normal_init', 'uniform_init',
]


def seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic(deterministic: bool = True):
    # if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
    #     torch.backends.cudnn.enabled = not deterministic
    #     torch.backends.cudnn.benchmark = not deterministic
    #     torch.backends.cudnn.deterministic = deterministic
    if hasattr(torch, 'set_deterministic'):
        torch.set_deterministic(deterministic)
    else:
        torch.use_deterministic_algorithms(deterministic)


# ---- uniform distribution ----
@jit
def rand(shape: List[int],
         dtype: str = settings.float_x,
         device: Optional[str] = None) -> Tensor:
    if dtype == 'float32':
        real_dtype = torch.float32
    else:
        real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]

    if device is None:
        device = current_device()
    return torch.rand(shape, dtype=real_dtype, device=device)


@jit
def uniform(shape: List[int], low: float, high: float,
            dtype: str = settings.float_x,
            device: Optional[str] = None) -> Tensor:
    if low >= high:
        raise ValueError('`low` < `high` does not hold: low == {}, high == {}'.
                         format(low, high))
    scale = high - low
    return rand(shape, dtype, device=device) * scale + low


@jit
def randint(low: int, high: int, shape: List[int],
            dtype: str = 'int32',
            device: Optional[str] = None) -> Tensor:
    if low >= high:
        raise ValueError('`low` < `high` does not hold: low == {}, high == {}'.
                         format(low, high))

    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    if device is None:
        device = current_device()
    return torch.randint(low, high, size=shape, dtype=target_dtype, device=device)


# ---- shuffle and random permutation ----
@jit
def shuffle(input: Tensor, axis: int = 0) -> Tensor:
    input_shape = input.shape
    shuffle_size = input_shape[axis]
    permutation = torch.randperm(
        shuffle_size, dtype=torch.long, device=input.device)
    if axis == 0:
        return input[permutation]
    else:
        return index_select(input, permutation, axis=axis)


@jit
def random_permutation(n: int,
                       dtype: str = 'int32',
                       device: Optional[str] = None) -> Tensor:
    if dtype == 'int32':
        int_dtype = torch.int32
    else:
        int_dtype = {'int8': torch.int8, 'int16': torch.int16, 'int64': torch.int64}[dtype]

    if device is None:
        device = current_device()
    return torch.randperm(n, dtype=int_dtype, device=device)


# ---- normal distribution ----
@jit
def randn(shape: List[int],
          dtype: str = settings.float_x,
          device: Optional[str] = None,) -> Tensor:
    if dtype == 'float32':
        real_dtype = torch.float32
    else:
        real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]

    if device is None:
        device = current_device()
    return torch.randn(shape, dtype=real_dtype, device=device)


@jit
def randn_log_pdf(given: Tensor, group_ndims: int = 0) -> Tensor:
    batch_ndims = len(given.shape)
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))

    c = -0.5 * math.log(2. * math.pi)
    ret = c - 0.5 * given ** 2
    if group_ndims > 0:
        ret = torch.sum(ret, dim=int_range(-group_ndims, 0))
    return ret


@jit
def normal(mean: Tensor,
           std: Tensor,
           n_samples: Optional[int] = None,
           reparameterized: bool = True) -> Tensor:
    if mean.dtype != std.dtype:
        raise ValueError('`mean.dtype` != `std.dtype`: {} vs {}'.
                         format(mean.dtype, std.dtype))
    param_shape = get_broadcast_shape(shape(mean), shape(std))
    if n_samples is not None:
        param_shape = [n_samples] + param_shape
    r = std * torch.randn(param_shape, dtype=mean.dtype, device=mean.device) + mean
    if not reparameterized:
        r = r.detach()
    return r


@jit
def normal_log_pdf(given: Tensor,
                   mean: Tensor,
                   logstd: Tensor,
                   group_ndims: int = 0,
                   validate_tensors: bool = False) -> Tensor:
    batch_ndims = max(len(given.shape), len(mean.shape), len(logstd.shape))
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))

    c = -0.5 * math.log(2. * math.pi)
    precision = exp(-2. * logstd)
    if validate_tensors:
        precision = assert_finite(precision, 'precision')
    precision = 0.5 * precision

    ret = c - logstd - precision * (given - mean) ** 2
    if group_ndims > 0:
        ret = torch.sum(ret, dim=int_range(-group_ndims, 0))
    return ret


# ---- bernoulli distribution ----
@jit
def bernoulli_logits_to_probs(logits: Tensor) -> Tensor:
    return sigmoid(logits)


@jit
def bernoulli_probs_to_logits(probs: Tensor,
                              epsilon: float = EPSILON) -> Tensor:
    probs_clipped = clip(probs, epsilon, 1. - epsilon)
    return log(probs_clipped) - log1p(-probs_clipped)


@jit
def bernoulli(probs: Tensor,
              n_samples: Optional[int] = None,
              dtype: str = 'int32') -> Tensor:
    # validate arguments
    if n_samples is not None and n_samples < 1:
        raise ValueError('`n_samples` must be at least 1: got {}'.
                         format(n_samples))

    if dtype == 'float32':
        target_dtype = torch.float32
    elif dtype == 'int32':
        target_dtype = torch.int32
    else:
        target_dtype = {'int8': torch.int8, 'uint8': torch.uint8, 'int16': torch.int16, 'int64': torch.int64, 'float16': torch.float16, 'float64': torch.float64, 'bool': torch.bool}[dtype]

    # do sample
    probs = probs.detach()
    sample_shape = probs.shape
    if n_samples is not None:
        sample_shape = (n_samples,) + sample_shape
        probs = probs.unsqueeze(dim=0).expand(sample_shape)
    out = torch.zeros(sample_shape, dtype=target_dtype, device=probs.device)
    return torch.bernoulli(probs, out=out).detach()


@jit
def bernoulli_log_prob(given: Tensor,
                       logits: Tensor,
                       group_ndims: int = 0) -> Tensor:
    batch_ndims = max(len(given.shape), len(logits.shape))
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))
    elif group_ndims == batch_ndims:
        return binary_cross_entropy_with_logits(
            logits=logits, labels=given, negative=True, reduction='sum')
    elif group_ndims > 0:
        ret = binary_cross_entropy_with_logits(
            logits=logits, labels=given, negative=True)
        return torch.sum(ret, dim=int_range(-group_ndims, 0))
    else:
        return binary_cross_entropy_with_logits(
            logits=logits, labels=given, negative=True)


# ---- categorical distribution ----
@jit
def categorical_logits_to_probs(logits: Tensor) -> Tensor:
    return softmax(logits)


@jit
def categorical_probs_to_logits(probs: Tensor,
                                epsilon: float = EPSILON) -> Tensor:
    return log(clip(probs, epsilon, 1. - epsilon))


@jit
def _categorical_sub(probs: Tensor, n_samples: Optional[int], dtype: str) -> Tensor:
    # do sample
    probs = probs.detach()
    if len(probs.shape) > 2:
        probs, front_shape = flatten_to_ndims(probs, 2)
    else:
        probs, front_shape = probs, None

    if n_samples is None:
        ret = torch.multinomial(probs, 1, replacement=True)
        ret = torch.squeeze(ret, -1)
        if front_shape is not None:
            ret = unflatten_from_ndims(ret, front_shape)
    else:
        ret = torch.multinomial(probs, n_samples, replacement=True)
        if front_shape is not None:
            ret = unflatten_from_ndims(ret, front_shape)
        ret = ret.permute([-1] + int_range(0, len(ret.shape) - 1))

    ret = cast(ret, dtype)
    return ret.detach()


@jit
def categorical(probs: Tensor,
                n_samples: Optional[int] = None,
                dtype: str = categorical_dtype) -> Tensor:
    if n_samples is not None and n_samples < 1:
        raise ValueError('`n_samples` must be at least 1: got {}'.
                         format(n_samples))
    if len(probs.shape) < 1:
        raise ValueError(
            'The rank of `probs` must be at least 1: '
            'got {}'.format(len(probs.shape))
        )
    return _categorical_sub(probs, n_samples, dtype)


@jit
def categorical_log_prob(given: Tensor,
                         logits: Tensor,
                         group_ndims: int = 0) -> Tensor:
    batch_ndims = max(len(given.shape) + 1, len(logits.shape)) - 1
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))
    else:
        if given.dtype != torch.int64:
            given = given.to(torch.int64)
        if group_ndims == batch_ndims:
            return cross_entropy_with_logits(
                logits=logits, labels=given, negative=True, reduction='sum')
        elif group_ndims > 0:
            ret = cross_entropy_with_logits(
                logits=logits, labels=given, negative=True)
            return torch.sum(ret, dim=int_range(-group_ndims, 0))
        else:
            return cross_entropy_with_logits(
                logits=logits, labels=given, negative=True)


@jit
def one_hot_categorical(probs: Tensor,
                        n_samples: Optional[int] = None,
                        dtype: str = 'int32') -> Tensor:
    if n_samples is not None and n_samples < 1:
        raise ValueError('`n_samples` must be at least 1: got {}'.
                         format(n_samples))
    if len(probs.shape) < 1:
        raise ValueError(
            'The rank of `probs` must be at least 1: '
            'got {}'.format(len(probs.shape))
        )
    n_classes = probs.shape[-1]
    ret = _categorical_sub(probs, n_samples, dtype='int64')
    ret = one_hot(ret, n_classes)
    ret = cast(ret, dtype)
    return ret


@jit
def one_hot_categorical_log_prob(given: Tensor,
                                 logits: Tensor,
                                 group_ndims: int = 0) -> Tensor:
    batch_ndims = max(len(given.shape), len(logits.shape)) - 1
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))
    elif group_ndims == batch_ndims:
        return sparse_cross_entropy_with_logits(
            logits=logits, labels=given, negative=True, reduction='sum')
    elif group_ndims > 0:
        ret = sparse_cross_entropy_with_logits(
            logits=logits, labels=given, negative=True)
        return torch.sum(ret, dim=int_range(-group_ndims, 0))
    else:
        return sparse_cross_entropy_with_logits(
            logits=logits, labels=given, negative=True)


# ---- initializers ----
def normal_init(tensor: Tensor, mean: float, std: float) -> Tensor:
    torch.nn.init.normal_(tensor, mean, std)
    return tensor


def uniform_init(tensor: Tensor, low: float, high: float) -> Tensor:
    torch.nn.init.uniform_(tensor, low, high)
    return tensor
