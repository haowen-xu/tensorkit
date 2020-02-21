from ..tensor import jit, reduce_mean, reduce_sum, Tensor

__all__ = ['apply_reduction']


@jit
def apply_reduction(input: Tensor, reduction: str) -> Tensor:
    if reduction == 'mean':
        return reduce_mean(input)
    elif reduction == 'sum':
        return reduce_sum(input)
    elif reduction == 'none':
        return input
    else:
        raise ValueError('Unsupported reduction: {}'.format(reduction))
