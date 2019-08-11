from . import core, extended_tensor
from .core import *
from .extended_tensor import *

name = 'pytorch'

__all__ = ['name'] + core.__all__ + extended_tensor.__all__
del core, extended_tensor
