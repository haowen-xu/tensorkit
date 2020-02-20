import math
from typing import *

from ..backend import *
from ..backend.nn import *
from ..backend.random import *

__all__ = [
    'LOG_ZERO_VALUE',

    # normal and its variants
    'truncated_randn', 'truncated_randn_log_pdf',
    'truncated_normal', 'truncated_normal_log_pdf',

    # discretized logistic
    'discretized_logistic', 'discretized_logistic_log_prob',
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
                    device: Optional[str] = None,
                    epsilon: float = EPSILON) -> Tensor:
    # fast routine: low is None and high is None, use standard randn
    if low is None and high is None:
        return randn(shape, dtype, device)

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
    u = rand(shape, dtype, device)

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
            float_scalar_like(log_zero, log_pdf))
    elif low is not None:
        log_pdf = where(
            low <= given,
            log_pdf,
            float_scalar_like(log_zero, log_pdf))
    elif high is not None:
        log_pdf = where(
            given <= high,
            log_pdf,
            float_scalar_like(log_zero, log_pdf))
    else:
        log_pdf = log_pdf  # do nothing, but JIT requires this branch

    if group_ndims > 0:
        log_pdf = reduce_sum(log_pdf, axis=int_range(-group_ndims, 0))
    return log_pdf


@jit
def truncated_normal(mean: Tensor,
                     std: Tensor,
                     low: Optional[float] = None,
                     high: Optional[float] = None,
                     n_samples: Optional[int] = None,
                     reparameterized: bool = True,
                     epsilon: float = EPSILON) -> Tensor:
    if mean.dtype != std.dtype:
        raise ValueError('`mean.dtype` != `std.dtype`: {} vs {}'.
                         format(mean.dtype, std.dtype))
    param_shape = broadcast_shape(shape(mean), shape(std))
    if n_samples is not None:
        param_shape = [n_samples] + param_shape
    r = truncated_randn(param_shape, low=low, high=high, dtype=get_dtype(mean),
                        epsilon=epsilon, device=get_device(mean))
    r = r * std + mean
    if not reparameterized:
        r = stop_grad(r)
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
            float_scalar_like(log_zero, log_pdf))
    elif low is not None:
        log_pdf = where(
            (low * std + mean) <= given,
            log_pdf,
            float_scalar_like(log_zero, log_pdf))
    elif high is not None:
        log_pdf = where(
            given <= (high * std + mean),
            log_pdf,
            float_scalar_like(log_zero, log_pdf))
    else:
        log_pdf = log_pdf  # do nothing, but JIT requires this branch

    if group_ndims > 0:
        log_pdf = reduce_sum(log_pdf, axis=int_range(-group_ndims, 0))
    return log_pdf


# ---- discretized logistic ----
@jit
def _discretized_logistic_discretize(x: Tensor,
                                     bin_size: float,
                                     min_val: Optional[float] = None,
                                     max_val: Optional[float] = None):
    if min_val is not None:
        x = x - min_val
    x = floor(x / bin_size + .5) * bin_size
    if min_val is not None:
        x = x + min_val
    x = maybe_clip(x, min_val, max_val)
    return x


@jit
def discretized_logistic(mean: Tensor,
                         log_scale: Tensor,
                         bin_size: float,
                         min_val: Optional[float] = None,
                         max_val: Optional[float] = None,
                         discretize: bool = True,
                         reparameterized: bool = False,
                         n_samples: Optional[int] = None,
                         epsilon: float = EPSILON,
                         validate_tensors: bool = False) -> Tensor:
    if (min_val is not None and max_val is None) or \
            (min_val is None and max_val is not None):
        raise ValueError('`min_val` and `max_val` must be both None or neither None.')
    if discretize and reparameterized:
        raise ValueError('`discretize` and `reparameterized` cannot be both True.')

    # sample from uniform distribution
    sample_shape = broadcast_shape(shape(mean), shape(log_scale))
    if n_samples is not None:
        sample_shape = [n_samples] + sample_shape
    mean_dtype = get_dtype(mean)
    log_scale_dtype = get_dtype(log_scale)
    if mean_dtype != log_scale_dtype:
        raise ValueError('`mean.dtype` != `log_scale.dtype`: {} vs {}'.
                         format(mean_dtype, log_scale_dtype))

    u = uniform(shape=sample_shape, low=epsilon, high=1. - epsilon,
                dtype=mean_dtype, device=get_device(mean))

    # inverse CDF of the logistic
    inverse_logistic_cdf = log(u) - log1p(-u)
    if validate_tensors:
        inverse_logistic_cdf = assert_finite(
            inverse_logistic_cdf, 'inverse_logistic_cdf')

    # obtain the actual sample
    scale = exp(log_scale)
    if validate_tensors:
        scale = assert_finite(scale, 'scale')

    sample = mean + scale * inverse_logistic_cdf
    if discretize:
        sample = _discretized_logistic_discretize(
            sample, bin_size, min_val, max_val)

    if not reparameterized:
        sample = stop_grad(sample)

    return sample


@jit
def discretized_logistic_log_prob(given: Tensor,
                                  mean: Tensor,
                                  log_scale: Tensor,
                                  bin_size: float,
                                  min_val: Optional[float] = None,
                                  max_val: Optional[float] = None,
                                  biased_edges: bool = True,
                                  discretize: bool = True,
                                  group_ndims: int = 0,
                                  epsilon: float = EPSILON,
                                  log_zero: float = LOG_ZERO_VALUE,
                                  validate_tensors: bool = False) -> Tensor:
    if (min_val is not None and max_val is None) or \
            (min_val is None and max_val is not None):
        raise ValueError('`min_val` and `max_val` must be both None or neither None.')

    batch_ndims = len(given.shape)
    if group_ndims > batch_ndims:
        raise ValueError(
            '`group_ndims` is too large: the maximum possible value is {}, '
            'but got {}'.format(batch_ndims, group_ndims))

    if discretize:
        given = _discretized_logistic_discretize(given, bin_size, min_val, max_val)

    # inv_scale = 1. / exp(log_scale)
    inv_scale = exp(-log_scale)
    if validate_tensors:
        inv_scale = assert_finite(inv_scale, 'inv_scale')
    # half_bin = bin_size / 2
    half_bin = bin_size * .5
    # delta = bin_size / scale, half_delta = delta / 2
    half_delta = half_bin * inv_scale

    # x_mid = (x - mean) / scale
    x_mid = (given - mean) * inv_scale

    # x_low = (x - mean - bin_size * 0.5) / scale
    x_low = x_mid - half_delta
    # x_high = (x - mean + bin_size * 0.5) / scale
    x_high = x_mid + half_delta

    cdf_low = sigmoid(x_low)
    cdf_high = sigmoid(x_high)
    cdf_delta = cdf_high - cdf_low

    # the middle bins cases:
    #   log(sigmoid(x_high) - sigmoid(x_low))
    # middle_bins_pdf = tf.log(cdf_delta + self._epsilon)
    epsilon_tensor = float_scalar_like(epsilon, cdf_delta)
    middle_bins_pdf = log(maximum(cdf_delta, epsilon_tensor))

    # # but in extreme cases where `sigmoid(x_high) - sigmoid(x_low)`
    # # is very small, we use an alternative form, as in PixelCNN++.
    # log_delta = log(bin_size) - log_scale
    # middle_bins_pdf = where(
    #     cdf_delta > epsilon_tensor,
    #     # to avoid NaNs pollute the select statement, we have to use
    #     # `maximum(cdf_delta, 1e-12)`
    #     log(maximum(cdf_delta, float_scalar_like(1e-12, cdf_delta))),
    #     # the alternative form.  basically it can be derived by using
    #     # the mean value theorem for integration.
    #     x_mid + log_delta - 2. * softplus(x_mid)
    # )

    log_prob = middle_bins_pdf
    if validate_tensors:
        log_prob = assert_finite(log_prob, 'log_prob')

    if min_val is not None and max_val is not None:
        if biased_edges:
            # broadcasted given, shape == x_mid
            broadcast_given = broadcast_to(given, shape(x_low))

            # the left-edge bin case
            #   log(sigmoid(x_high) - sigmoid(-infinity))
            left_edge = float_scalar_like(min_val + half_bin, broadcast_given)
            left_edge_pdf = -softplus(-x_high)
            if validate_tensors:
                left_edge_pdf = assert_finite(left_edge_pdf, 'left_edge_pdf')

            log_prob = where(
                less(broadcast_given, left_edge),
                left_edge_pdf,
                log_prob
            )

            # the right-edge bin case
            #   log(sigmoid(infinity) - sigmoid(x_low))
            right_edge = float_scalar_like(max_val - half_bin, broadcast_given)
            right_edge_pdf = -softplus(x_low)
            if validate_tensors:
                right_edge_pdf = assert_finite(right_edge_pdf, 'right_edge_pdf')

            log_prob = where(
                greater_equal(broadcast_given, right_edge),
                right_edge_pdf,
                log_prob
            )

        # we should zero out prob (i.e., set log_prob to `log_zero`) which is
        # outside of `[min_val - half_bin, max_val + half_bin]`.
        if not discretize:
            log_prob = where(
                logical_and(given >= min_val - half_bin,
                            given <= max_val + half_bin),
                log_prob,
                float_scalar_like(log_zero, log_prob))

    # now reduce the group_ndims
    if group_ndims > 0:
        log_prob = reduce_sum(log_prob, axis=int_range(-group_ndims, 0))

    return log_prob
