import math
from typing import *

import torch

from ._utils import int_range
from .core import *
from .dtypes import categorical_dtype
from .nn import *
from ...settings_ import settings

__all__ = [
    # common utils
    'LOG_ZERO_VALUE', 'seed',

    # uniform
    'rand', 'uniform',

    # normal
    'randn', 'randn_log_pdf', 'truncated_randn', 'truncated_randn_log_pdf',
    'normal', 'normal_log_pdf', 'truncated_normal', 'truncated_normal_log_pdf',

    # bernoulli
    'bernoulli_logits_to_probs', 'bernoulli_probs_to_logits',
    'bernoulli', 'bernoulli_log_prob',

    # categorical
    'categorical_logits_to_probs', 'categorical_probs_to_logits',
    'categorical', 'categorical_log_prob',
    'one_hot_categorical', 'one_hot_categorical_log_prob',
]

LOG_ZERO_VALUE = -1e7
"""A sufficiently small value to represent log(0), to avoid having ``-inf``."""


def seed(seed: int):
    torch.manual_seed(seed)


# ---- uniform distribution ----
@jit
def rand(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    if dtype == 'float32':
        real_dtype = torch.float32
    else:
        real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]
    return torch.rand(shape, dtype=real_dtype)


@jit
def uniform(shape: List[int], low: float, high: float,
            dtype: str = settings.float_x) -> Tensor:
    if low >= high:
        raise ValueError('`low` < `high` does not hold: low == {}, high == {}'.
                         format(low, high))
    scale = high - low
    return rand(shape, dtype) * scale + low


# ---- normal distribution ----
@jit
def randn(shape: List[int], dtype: str = settings.float_x) -> Tensor:
    if dtype == 'float32':
        real_dtype = torch.float32
    else:
        real_dtype = {'float16': torch.float16, 'float64': torch.float64}[dtype]
    return torch.randn(shape, dtype=real_dtype)


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
    param_shape = broadcast_shape(shape(mean), shape(std))
    if n_samples is not None:
        param_shape = [n_samples] + param_shape
    r = std * torch.randn(param_shape, dtype=mean.dtype) + mean
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


def _unit_normal_cdf_float(x: float) -> float:
    return 0.5 * (1. + math.erf(x / math.sqrt(2)))


@jit
def _unit_normal_cdfinv(x: Tensor) -> Tensor:
    return math.sqrt(2) * torch.erfinv(2. * x - 1.)


@jit
def truncated_randn(shape: List[int],
                    low: Optional[float] = None,
                    high: Optional[float] = None,
                    dtype: str = settings.float_x,
                    epsilon: float = 1e-7) -> Tensor:
    # fast routine: low is None and high is None, use standard randn
    if low is None and high is None:
        return randn(shape, dtype)

    # compute cdf(low) and cdf(high)
    if low is None:
        low_cdf = 0.
    else:
        low_cdf = _unit_normal_cdf_float(low)

    if high is None:
        high_cdf = 1.
    else:
        high_cdf = _unit_normal_cdf_float(high)

    # sample u ~ uniform(0, 1)
    u = rand(shape, dtype)

    # transform uniform random variable into truncated normal
    if low_cdf == 0.:
        cdf = u * high_cdf
    elif high_cdf == 1.:
        cdf = u + (1. - u) * low_cdf
    else:
        cdf = (1. - u) * low_cdf + u * high_cdf

    cdf = torch.clamp(cdf, epsilon, 1. - epsilon)
    return _unit_normal_cdfinv(cdf)


def _truncated_normal_add_log_Z(log_pdf: Tensor,
                                low: Optional[float],
                                high: Optional[float],
                                log_zero: float) -> Tensor:
    if low is None:
        low_cdf = 0.
    else:
        low_cdf = _unit_normal_cdf_float(low)

    if high is None:
        high_cdf = 1.
    else:
        high_cdf = _unit_normal_cdf_float(high)

    Z = high_cdf - low_cdf
    if Z > 0.:
        return log_pdf - math.log(Z)
    else:
        return torch.full_like(log_pdf, -log_zero)  # i.e., +inf


@jit
def truncated_randn_log_pdf(given: Tensor,
                            low: Optional[float] = None,
                            high: Optional[float] = None,
                            group_ndims: int = 0,
                            log_zero: float = LOG_ZERO_VALUE) -> Tensor:
    batch_ndims = len(given.shape)
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))

    # get the original log_pdf
    log_pdf = randn_log_pdf(given)

    # add ``-log(Z)`` to log_pdf
    log_pdf = _truncated_normal_add_log_Z(log_pdf, low, high, log_zero)

    # we should zero out pdf (i.e., set log_pdf to `log_zero`) which is
    # outside of `[low, high]`.
    if low is not None and high is not None:
        log_pdf = where(
            logical_and(low <= given, given <= high),
            log_pdf,
            as_tensor(log_zero, dtype=log_pdf.dtype))
    elif low is not None:
        log_pdf = where(
            low <= given,
            log_pdf,
            as_tensor(log_zero, dtype=log_pdf.dtype))
    elif high is not None:
        log_pdf = where(
            given <= high,
            log_pdf,
            as_tensor(log_zero, dtype=log_pdf.dtype))
    else:
        log_pdf = log_pdf  # do nothing, but JIT requires this branch

    if group_ndims > 0:
        log_pdf = torch.sum(log_pdf, dim=int_range(-group_ndims, 0))
    return log_pdf


@jit
def truncated_normal(mean: Tensor,
                     std: Tensor,
                     low: Optional[float] = None,
                     high: Optional[float] = None,
                     n_samples: Optional[int] = None,
                     reparameterized: bool = True,
                     epsilon: float = 1e-7) -> Tensor:
    if mean.dtype != std.dtype:
        raise ValueError('`mean.dtype` != `std.dtype`: {} vs {}'.
                         format(mean.dtype, std.dtype))
    param_shape = broadcast_shape(shape(mean), shape(std))
    if n_samples is not None:
        param_shape = [n_samples] + param_shape
    r = truncated_randn(param_shape, low=low, high=high, dtype=get_dtype(mean),
                        epsilon=epsilon)
    r = r * std + mean
    if not reparameterized:
        r = r.detach()
    return r


@jit
def truncated_normal_log_pdf(given: Tensor,
                             mean: Tensor,
                             std: Tensor,
                             logstd: Tensor,
                             low: Optional[float] = None,  # k-sigma, not absolute value
                             high: Optional[float] = None,  # k-sigma, not absolute value
                             group_ndims: int = 0,
                             log_zero: float = LOG_ZERO_VALUE,
                             validate_tensors: bool = False) -> Tensor:
    batch_ndims = max(len(given.shape), len(mean.shape), len(logstd.shape))
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))

    # get the original log_pdf
    log_pdf = normal_log_pdf(
        given, mean, logstd, validate_tensors=validate_tensors)

    # add ``-log(Z)`` to log_pdf
    log_pdf = _truncated_normal_add_log_Z(log_pdf, low, high, log_zero)

    # we should zero out pdf (i.e., set log_pdf to `log_zero`) which is
    # outside of `[low, high]`.
    if low is not None and high is not None:
        log_pdf = where(
            logical_and((low * std + mean) <= given,
                        given <= (high * std + mean)),
            log_pdf,
            as_tensor(log_zero, dtype=log_pdf.dtype))
    elif low is not None:
        log_pdf = where(
            (low * std + mean) <= given,
            log_pdf,
            as_tensor(log_zero, dtype=log_pdf.dtype))
    elif high is not None:
        log_pdf = where(
            given <= (high * std + mean),
            log_pdf,
            as_tensor(log_zero, dtype=log_pdf.dtype))
    else:
        log_pdf = log_pdf  # do nothing, but JIT requires this branch

    if group_ndims > 0:
        log_pdf = torch.sum(log_pdf, dim=int_range(-group_ndims, 0))
    return log_pdf


# ---- bernoulli distribution ----
@jit
def bernoulli_logits_to_probs(logits: Tensor) -> Tensor:
    return sigmoid(logits)


@jit
def bernoulli_probs_to_logits(probs: Tensor,
                              epsilon: float = 1e-7) -> Tensor:
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
    out = torch.zeros(sample_shape, dtype=target_dtype)
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
                                epsilon: float = 1e-7) -> Tensor:
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
