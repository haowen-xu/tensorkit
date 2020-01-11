from ..settings_ import settings

if settings.backend == 'PyTorch':
    from .pytorch_ import name as backend_name, core, dtypes
    from .pytorch_.core import *
    from .pytorch_.dtypes import *
else:
    RuntimeError(f'Backend {settings.backend} not supported.')

__all__ = ['float_x', 'backend_name'] + core.__all__ + dtypes.__all__


def float_x() -> str:
    """Get the default dtype for floating point numbers."""
    return settings.float_x
