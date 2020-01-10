from typing import *

from torch.jit import script as jit

from ...settings_ import settings

__all__ = ['int_range']


if settings.disable_jit:
    def int_range(start: int, end: int, step: int = 1) -> List[int]:
        return list(range(start, end, step))
else:
    @jit
    def int_range(start: int, end: int, step: int = 1) -> List[int]:
        ret: List[int] = []
        for i in range(start, end, step):
            ret.append(i)
        return ret
