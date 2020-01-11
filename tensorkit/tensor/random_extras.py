import math
from typing import *

from ..backend import *
from ..backend.random import *

__all__ = [
    'LOG_ZERO_VALUE',

    # normal and its variants
    'truncated_randn', 'truncated_randn_log_pdf',
    'truncated_normal', 'truncated_normal_log_pdf',
]

LOG_ZERO_VALUE = -1e7
"""A sufficiently small value to represent log(0), to avoid having ``-inf``."""


# ---- normal distribution and its variants ----
def _unit_normal_cdf_float(x: float) -> float:
    return 0.5 * (1. + math.erf(x / math.sqrt(2)))


@jit
def _unit_normal_cdfinv(x: Tensor) -> Tensor:
    return math.sqrt(2) * erfinv(2. * x - 1.)


@jit
def truncated_randn(shape: List[int],
                    low: Optional[float] = None,
                    high: Optional[float] = None,
                    dtype: str = float_x(),
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

    cdf = clip(cdf, epsilon, 1. - epsilon)
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
        return full_like(log_pdf, -log_zero)  # i.e., +inf


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
        log_pdf = reduce_sum(log_pdf, axes=int_range(-group_ndims, 0))
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
        log_pdf = reduce_sum(log_pdf, axes=int_range(-group_ndims, 0))
    return log_pdf
