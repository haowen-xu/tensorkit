from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import train
    from .pytorch_.train import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = train.__all__
