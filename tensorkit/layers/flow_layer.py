from ..backend.flows import Flow
from ..tensor import Tensor, Module, is_jit_layer
from .core import *

__all__ = [
    'FlowLayer',
    'ActNorm', 'ActNorm1d', 'ActNorm2d', 'ActNorm3d',
]


class FlowLayer(BaseLayer):
    """
    Wrap a :class:`tk.flows.BaseFlow` into a single-input, single-output layer.
    """

    __constants__ = ('flow',)

    flow: Module

    def __init__(self, flow: Flow):
        if not is_jit_layer(flow) and not isinstance(flow, Flow):
            raise TypeError(f'`flow` must be a flow: got {flow!r}')
        super().__init__()
        self.flow = flow

    def forward(self, input: Tensor) -> Tensor:
        output, output_log_det = self.flow(input, compute_log_det=False)
        return output


class ActNorm(FlowLayer):

    def __init__(self, num_features: int, *args, **kwargs):
        from ..flows import ActNorm as FlowClass
        super().__init__(FlowClass(num_features, *args, **kwargs))


class ActNorm1d(FlowLayer):

    def __init__(self, num_features: int, *args, **kwargs):
        from ..flows import ActNorm1d as FlowClass
        super().__init__(FlowClass(num_features, *args, **kwargs))


class ActNorm2d(FlowLayer):

    def __init__(self, num_features: int, *args, **kwargs):
        from ..flows import ActNorm2d as FlowClass
        super().__init__(FlowClass(num_features, *args, **kwargs))


class ActNorm3d(FlowLayer):

    def __init__(self, num_features: int, *args, **kwargs):
        from ..flows import ActNorm3d as FlowClass
        super().__init__(FlowClass(num_features, *args, **kwargs))
