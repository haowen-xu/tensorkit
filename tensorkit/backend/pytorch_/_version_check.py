from distutils.version import LooseVersion

import torch

__all__ = []

if LooseVersion(torch.__version__) < LooseVersion('1.2.0'):
    raise RuntimeError('PyTorch >= 1.2.0 is required')  # pragma: no cover

