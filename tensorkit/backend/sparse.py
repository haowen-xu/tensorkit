from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import sparse
    from .pytorch_.sparse import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = sparse.__all__
