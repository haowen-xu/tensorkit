from typing import *

from ..tensor import Tensor, jit_method
from .core import *

__all__ = [
    'IgnoreContext', 'AddContext', 'MultiplyContext',
]


class IgnoreContext(BaseContextualLayer):
    """
    A module which simply returns the input, ignoring any context.
    """

    @jit_method
    def _call(self, input: Tensor, context: Optional[Tensor]) -> Tensor:
        return input


class AddContext(BaseContextualLayer):
    """
    A module which adds the input with the context.
    """

    @jit_method
    def _call(self, input: Tensor, context: Optional[Tensor]) -> Tensor:
        if context is None:
            raise RuntimeError('`context` is required.')
        return input + context


class MultiplyContext(BaseContextualLayer):
    """
    A module which multiplies the input with the context.
    """

    @jit_method
    def _call(self, input: Tensor, context: Optional[Tensor]) -> Tensor:
        if context is None:
            raise RuntimeError('`context` is required.')
        return input * context
