from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import losses
    from .pytorch_.losses import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = losses.__all__
