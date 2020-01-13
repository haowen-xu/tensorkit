import copy
from typing import *

from ..tensor import (jit, Tensor, where, as_tensor_jit, as_tensor, get_dtype,
                      float_x)

__all__ = [
    'get_overrided_parameterized',
    'get_prob_reduce_ndims',
    'get_tail_size',
    'log_pdf_mask',
    'check_tensor_arg_types',
    'copy_distribution',
]

TDistribution = TypeVar('TDistribution')


def get_overrided_parameterized(base: bool,
                                override: Optional[bool],
                                cls: Optional[Type] = None
                                ) -> bool:
    if override and not base:
        cls_name = f'`{cls.__qualname__}` ' if cls is not None else ''
        raise ValueError(
            f'Distribution {cls_name}is not re-parameterizable, thus '
            f'`reparameterized` cannot be set to True.'
        )
    if override is None:
        override = base
    return override


@jit
def get_prob_reduce_ndims(sample_ndims: int,
                          min_event_ndims: int,
                          event_ndims: int,
                          group_ndims: int) -> int:
    ret = group_ndims + (event_ndims - min_event_ndims)
    if not (0 <= ret <= sample_ndims - min_event_ndims):
        raise ValueError(
            '`min_event_ndims - event_ndims <= group_ndims <= '
            'sample_ndims - event_ndims` does not hold: '
            '`group_ndims` == {}, '
            '`event_ndims` == {}, '
            '`min_event_ndims` == {}, and '
            '`sample_ndims` == {}'.
            format(group_ndims, event_ndims, min_event_ndims, sample_ndims)
        )
    return ret


@jit
def get_tail_size(shape: List[int], ndims: int) -> int:
    """
    Get the total size of the tail of a given shape.

    Args:
        shape: The whole shape.
        ndims: The number of tail dimensions.
            ``ndims <= len(shape)`` is required.

    Returns:
        The total size of the tail.
    """
    rank = len(shape)
    if rank < ndims:
        raise ValueError('`ndims <= len(shape)` does not hold: '
                         '`ndims` == {}, `shape` == {}'.format(ndims, shape))
    r = 1
    for i in range(rank - ndims, rank):
        r *= shape[i]
    return r


@jit
def log_pdf_mask(condition: Tensor,
                 log_pdf: Tensor,
                 log_zero: float) -> Tensor:
    """
    Take the value of `log_pdf` only where `condition == True`, and zero
    out remaining positions (i.e., set log-pdf of these locations to
    `log_zero`).
    """
    return where(condition, log_pdf, as_tensor_jit(log_zero, dtype=log_pdf.dtype))


def check_tensor_arg_types(*args,
                           dtype: Optional[str] = None,
                           default_dtype: str = float_x()
                           ) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], ...]:
    """
    Validate tensor argument types.

    Args:
        *args: Each argument should be one of the following two cases:
            1. A tuple of ``(name, data)``: the name of the argument, as well
               as the data, which could be casted into tensor.  The `data`
               must not be None.
            2. A list of tuples of ``(name, data)``: the names and the
               corresponding data.  One and of the data should be not None,
               while the others must be None.
        dtype: If specified, all arguments must be tensors of this dtype,
            or Python numbers (which can be casted into this dtype).
        default_dtype: The default dtype to cast Python numbers into,
            if `dtype` is not specified, and all arguments are Python numbers
            (thus no dtype can be inferred).

    Returns:
        A list of validated tensors.

    Raises:
        ValueError: If any argument is invalid.
    """
    from ..stochastic import StochasticTensor

    def check_dtype(name, data):
        if isinstance(data, StochasticTensor):
            data = data.tensor
        if isinstance(data, Tensor):
            data_dtype = get_dtype(data)
            if inferred_dtype[1] is None:
                inferred_dtype[0] = f'{name}.dtype'
                inferred_dtype[1] = data_dtype
            elif inferred_dtype[1] != data_dtype:
                raise ValueError(f'`{name}.dtype` != `{inferred_dtype[0]}`: '
                                 f'{data_dtype} vs {inferred_dtype[1]}')

    def check_arg(arg):
        if isinstance(arg, tuple):
            name, data = arg
            if data is None:
                raise ValueError(f'`{name}` must be specified.')
            check_dtype(name, data)
        else:
            not_none_count = 0
            for i, (name, data) in enumerate(arg):
                if data is not None:
                    not_none_count += 1
                    if not_none_count != 1:
                        break
                    check_dtype(name, data)
            if not_none_count != 1:
                names = [f'`{n}`' for n, _ in arg]
                if len(names) == 2:
                    names = ' or '.join(names)
                    raise ValueError(f'Either {names} must be specified, '
                                     f'but not both.')
                else:
                    names = ' and '.join((', '.join(names[:-1]), names[-1]))
                    raise ValueError(f'One and exactly one of {names} must '
                                     f'be specified.')

    # check the arguments
    if dtype is not None:
        inferred_dtype = ['dtype', dtype]
    else:
        inferred_dtype = [None, None]

    for a in args:
        check_arg(a)

    # do cast the tensors
    target_dtype = inferred_dtype[1] or default_dtype
    ret: List[Union[Tensor, Tuple[Tensor, ...]]] = []
    for arg in args:
        if isinstance(arg, tuple):
            ret.append(as_tensor(arg[1], dtype=target_dtype))
        else:
            ret.append(tuple(
                (as_tensor(data, dtype=target_dtype)
                 if data is not None else None)
                for _, data in arg
            ))
    return tuple(ret)


def copy_distribution(cls: Type[TDistribution],
                      base,
                      attrs: Sequence[Union[str, Tuple[str, str]]],
                      mutual_attrs: Sequence[Sequence[str]] = (),
                      cached_attrs: Sequence[str] = (),
                      compute_deps: Optional[Dict[str, Sequence[str]]] = None,
                      original_mutual_params: Optional[Dict[str, Any]] = None,
                      overrided_params: Optional[Dict[str, Any]] = None,
                      ) -> TDistribution:
    """
    Copy a distribution object.

    Args:
        cls: The class of the distribution to be constructed.
        base: The distribution object.
        attrs: List of individual attributes, which can be directly copied
            without further processing.  If given a tuple of str, it will
            be interpreted as ``(arg_name, attr_name)``.
        mutual_attrs: List of groups of mutual attributes, where one and
            only one attribute within each group should be specified in
            the constructor, and the rest attributes of the group will be
            computed according to the specified one.
        cached_attrs: List of cached attributes.
        compute_deps: Dict of computation dependencies of the mutual
            or cached attributes which cannot be inferred automatically.
        original_mutual_params: The original mutual attributes specified
            in the constructor of `base` object.
        overrided_params: The overrided attributes.

    Returns:
        The copied distribution object.
    """
    for k in ('validate_tensors', 'event_ndims'):
        assert(k in attrs)

    compute_deps = compute_deps or {}
    original_mutual_params = original_mutual_params or {}
    overrided_params = copy.copy(overrided_params) \
        if overrided_params is not None else {}

    # merge individual attributes
    for attr in attrs:
        if isinstance(attr, tuple):
            arg_name, attr = attr
        else:
            arg_name, attr = attr, attr
        overrided_params.setdefault(arg_name, getattr(base, attr))

    # merge mutual attributes
    mutual_group_overrided: List[bool] = [False] * len(mutual_attrs)
    mutual_group_copied_attr: List[Optional[str]] = \
        [None] * len(mutual_attrs)

    for i, g in enumerate(mutual_attrs):
        if not any(attr in overrided_params for attr in g):
            # if none of the attribute within this mutual group has been
            # overrided, just use the original attribute of the class
            for attr in g:
                if attr in original_mutual_params:
                    overrided_params[attr] = original_mutual_params[attr]
                    mutual_group_copied_attr[i] = attr
                    break
        else:
            # otherwise set the overrided flag for this group
            mutual_group_overrided[i] = True

    # construct the object
    ret = cls(**overrided_params)

    # now copy the mutual or cached attributes
    # copy only if validate_tensors matches, or `ret.validate_tensors` is False
    def copy_computed(attr):
        if all(getattr(base, k) == getattr(ret, k)
               for k in compute_deps.get(attr, ())):
            private_attr = f'_{attr}'
            setattr(ret, private_attr, getattr(base, private_attr))

    if base.validate_tensors or not ret.validate_tensors:
        for i, g in enumerate(mutual_attrs):
            if not mutual_group_overrided[i]:
                copied_attr = mutual_group_copied_attr[i]
                for attr in g:
                    if attr != copied_attr:
                        copy_computed(attr)
        for attr in cached_attrs:
            copy_computed(attr)

    return ret
