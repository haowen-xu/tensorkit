from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import init
    from .pytorch_.init import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = init.__all__
