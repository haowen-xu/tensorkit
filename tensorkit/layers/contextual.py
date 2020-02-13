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
    def _forward(self, input: Tensor, context: List[Tensor]) -> Tensor:
        return input


class AddContext(BaseContextualLayer):
    """
    A module which adds the input with the contexts.
    """

    @jit_method
    def _forward(self, input: Tensor, context: List[Tensor]) -> Tensor:
        output = input
        for t in context:
            output = output + t
        return output


class MultiplyContext(BaseContextualLayer):
    """
    A module which multiplies the input with the contexts.
    """

    @jit_method
    def _forward(self, input: Tensor, context: List[Tensor]) -> Tensor:
        output = input
        for t in context:
            output = output * t
        return output
