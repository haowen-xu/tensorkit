from . import core, dtypes
from .core import *
from .dtypes import *
from .nn_ import *
from .random_ import *

name = 'pytorch'

__all__ = core.__all__ + dtypes.__all__ + ['nn', 'random']
del core, dtypes
