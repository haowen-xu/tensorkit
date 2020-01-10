# setup the backend
from tensorkit.settings_ import settings

if settings.backend == 'PyTorch':
    from . import pytorch_ as backend
    from .pytorch_ import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

del settings


# import common utilities
from .common import *
