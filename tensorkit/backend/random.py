from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import random
    from .pytorch_.random import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = random.__all__
