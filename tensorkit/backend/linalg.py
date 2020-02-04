from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import linalg
    from .pytorch_.linalg import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = linalg.__all__
