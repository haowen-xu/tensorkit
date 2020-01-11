from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import nn
    from .pytorch_.nn import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = nn.__all__
