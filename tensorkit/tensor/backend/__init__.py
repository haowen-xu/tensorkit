# setup the backend
from tensorkit.settings_ import settings

if settings.backend == 'pytorch':
    from . import pytorch_ as backend
    from .pytorch_ import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

del settings


# inject docstrings
from . import _docstrings
del _docstrings


# export symbols
__all__ = backend.__all__
