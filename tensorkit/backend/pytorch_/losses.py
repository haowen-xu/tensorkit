from .core import *
from .layers import *


__all__ = [
    'BaseSupervisedLossLayer',
]


class BaseSupervisedLossLayer(BaseLayer):

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        ret = self._forward(output, target)
        if ret.numel() > 1:
            ret = ret.mean()
        return ret
