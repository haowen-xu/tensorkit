from ..tensor import Tensor, tanh
from ..tensor.nn import *
from .core import *

__all__ = [
    'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'LogSoftmax',
]


class ReLU(BaseLayer):

    def forward(self, input: Tensor) -> Tensor:
        return relu(input)


class LeakyReLU(BaseLayer):

    __constants__ = ('negative_slope',)

    negative_slope: float

    def __init__(self, negative_slope=LEAKY_RELU_DEFAULT_SLOPE):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return leaky_relu(input, negative_slope=self.negative_slope)


class Tanh(BaseLayer):

    def forward(self, input: Tensor) -> Tensor:
        return tanh(input)


class Sigmoid(BaseLayer):

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid(input)


class LogSoftmax(BaseLayer):

    def forward(self, input: Tensor) -> Tensor:
        return log_softmax(input)
