from typing import *

__all__ = ['validate_int_tuple_arg']


def validate_int_tuple_arg(arg_name: str,
                           arg_value: Union[int, Tuple[int, ...]],
                           nullable: bool = False):
    try:
        if arg_value is None and nullable:
            pass
        elif hasattr(arg_value, '__iter__'):
            arg_value = tuple(int(v) for v in arg_value)
        else:
            arg_value = (int(arg_value),)
    except (ValueError, TypeError):
        raise ValueError(f'Invalid value for argument `{arg_name}`: expected '
                         f'to be a tuple of integers, but got {arg_value!r}')

    return arg_value
