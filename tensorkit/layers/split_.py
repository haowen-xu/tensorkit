from typing import *

from ..tensor import Tensor, Module
from .core import *

__all__ = ['Branch']


class Branch(BaseLayer):
    """
    A module that maps the input tensor into multiple tensors via sub-modules.

    ::

        shared_output = shared(input)
        outputs = [branch(shared_output) for branch in branches]
    """

    __constants__ = ('shared', 'branches')

    shared: Module
    branches: ModuleList

    def __init__(self,
                 branches: Sequence[Module],
                 shared: Optional[Module] = None):
        """
        Construct a enw :class:`Branch` module.

        Args:
            branches: The branch sub-modules.
            shared: The shared module to apply before the branch sub-modules.
        """
        if shared is None:
            shared = Identity()

        super().__init__()
        self.branches = ModuleList(list(branches))
        self.shared = shared

    def forward(self, input: Tensor) -> List[Tensor]:
        outputs: List[Tensor] = []
        input = self.shared(input)
        for branch in self.branches:
            outputs.append(branch(input))
        return outputs
