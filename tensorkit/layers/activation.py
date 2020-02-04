from ..tensor import Tensor, tanh
from ..tensor.nn import LEAKY_RELU_DEFAULT_SLOPE, relu, leaky_relu, sigmoid
from .core import *

__all__ = [
    'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid',
]


class ReLU(BaseSingleVariateLayer):

    def _call(self, input: Tensor) -> Tensor:
        return relu(input)


class LeakyReLU(BaseSingleVariateLayer):

    __constants__ = ('negative_slope',)

    negative_slope: float

    def __init__(self, negative_slope=LEAKY_RELU_DEFAULT_SLOPE):
        super().__init__()
        self.negative_slope = negative_slope

    def _call(self, input: Tensor) -> Tensor:
        return leaky_relu(input, negative_slope=self.negative_slope)


class Tanh(BaseSingleVariateLayer):

    def _call(self, input: Tensor) -> Tensor:
        return tanh(input)


class Sigmoid(BaseSingleVariateLayer):

    def _call(self, input: Tensor) -> Tensor:
        return sigmoid(input)
