from typing import *

from ..tensor import Tensor, float_scalar_like
from .core import *

__all__ = ['IdentityFlow']


class IdentityFlow(Flow):

    def __init__(self, event_ndims: int):
        super().__init__(x_event_ndims=event_ndims, y_event_ndims=event_ndims,
                         explicitly_invertible=True)

    def _transform(self,
                   input: Tensor,
                   input_log_det: Optional[Tensor],
                   inverse: bool,
                   compute_log_det: bool) -> Tuple[Tensor, Optional[Tensor]]:
        output = input
        output_log_det = input_log_det
        if compute_log_det:
            if output_log_det is None:
                output_log_det = float_scalar_like(0., input)
        return output, output_log_det
