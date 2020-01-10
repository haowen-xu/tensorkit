import copy
from typing import *

from ..backend import jit, Tensor, where, as_tensor

__all__ = [
    'get_overrided_parameterized',
    'get_prob_reduce_ndims',
    'get_tail_size',
    'log_pdf_mask',
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
    return where(condition, log_pdf, as_tensor(log_zero, dtype=log_pdf.dtype))


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
