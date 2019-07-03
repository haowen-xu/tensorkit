from . import core, dtypes, random_

from .core import *
from .dtypes import *
from .random_ import *

__all__ = ['name'] + (
    core.__all__ + dtypes.__all__ + random_.__all__
)
name = 'pytorch'
