from .core import Tensor, jit, rank, length, shape, reduce_sum, reduce_mean
from .nn import log_sigmoid, softplus

__all__ = ['negative_sampling']


@jit
def negative_sampling(pos_logits: Tensor,
                      neg_logits: Tensor,
                      reduction: str = 'none',  # {'sum', 'mean' or 'none'}
                      negative: bool = False
                      ) -> Tensor:
    if rank(pos_logits) != 1 or rank(neg_logits) != 2 or \
            length(pos_logits) != length(neg_logits):
        raise ValueError(
            '`pos_logits` must be 1d, `neg_logits` must be 2d, and they must '
            'have identical length in their first dimension: shape {} vs {}.'.
            format(shape(pos_logits), shape(neg_logits)))

    pos_logits = log_sigmoid(pos_logits)
    neg_logits = reduce_sum(softplus(neg_logits), axis=[-1])

    if negative:
        output = neg_logits - pos_logits
    else:
        output = pos_logits - neg_logits

    if reduction == 'sum':
        output = reduce_sum(output)
    elif reduction == 'mean':
        output = reduce_mean(output)
    elif reduction != 'none':
        raise ValueError('Invalid value for `reduction`: {!r}'.format(reduction))

    return output
