from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import optim
    from .pytorch_.optim import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = optim.__all__
