from ..tensor import Tensor, tanh, clip
from ..tensor.nn import *
from .core import *

__all__ = [
    'ReLU', 'LeakyReLU', 'Tanh', 'HardTanh', 'Sigmoid', 'LogSoftmax',
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


class HardTanh(BaseLayer):

    __constants__ = ('min_val', 'max_val')

    min_val: float
    max_val: float

    def __init__(self, min_val: float = -1., max_val: float = 1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, input: Tensor) -> Tensor:
        return clip(input, self.min_val, self.max_val)


class Sigmoid(BaseLayer):

    def forward(self, input: Tensor) -> Tensor:
        return sigmoid(input)


class LogSoftmax(BaseLayer):

    def forward(self, input: Tensor) -> Tensor:
        return log_softmax(input)
