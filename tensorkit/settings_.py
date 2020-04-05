from enum import Enum
from typing import *

from mltk import Config, ConfigField, field_checker

__all__ = ['JitMode', 'Settings', 'settings']


KNOWN_BACKENDS = ('PyTorch', 'TensorFlow')


def auto_choose_backend() -> Optional[str]:
    """
    Choose the backend automatically.

    If the dependencies of one backend has been imported, then it will be
    chosen as the preferred backend.  If the dependencies of multiple backends
    have been imported, then the backend will be chosen according to the
    following priority order:

        PyTorch, TensorFlow

    Returns:
        The backend name, or `None` if no backend can be automatically chosen.
    """
    import sys

    activated_backends = []

    for name, module in sys.modules.items():
        for backend in KNOWN_BACKENDS:
            if backend.lower() in name.lower() and backend not in activated_backends:
                activated_backends.append(backend)

    for backend in KNOWN_BACKENDS:
        if backend in activated_backends:
            return backend


class JitMode(str, Enum):
    """Enum of the JIT mode."""

    ALL = 'all'
    """Enable JIT on both functions and modules (layers)."""

    FUNCTION_ONLY = 'function_only'
    """Enable JIT only on functions."""

    NONE = 'none'
    """Disable JIT."""


class Settings(Config):

    backend: str = ConfigField(
        default=auto_choose_backend() or KNOWN_BACKENDS[0],
        choices=KNOWN_BACKENDS,
        envvar='TENSORKIT_BACKEND',
        description='The backend to use.'
                    'Changing the value of this configuration at runtime '
                    'will not take effect.'
    )
    float_x: str = ConfigField(
        default='float32',
        choices=['float32', 'float64'],
        envvar='TENSORKIT_FLOATX',
        description='The default dtype for floating-point numbers. '
                    'Changing the value of this configuration at runtime may '
                    'not take effect.'
    )
    validate_tensors: bool = ConfigField(
        default=False,
        envvar='TENSORKIT_VALIDATE_TENSORS',
        description='Whether or not to perform time-consuming validation on '
                    'tensors for input arguments of functions or classes, '
                    'and on intermediate computation results, to ensure there '
                    'are no numerical issues (i.e., no NaN or Infinity values), '
                    'and no semantic or logical errors (e.g., `low` > `high`)?'
    )
    jit_mode: Optional[JitMode] = ConfigField(
        default=None,
        envvar='TENSORKIT_JIT_MODE',
        description='The mode of JIT engine. If not specified, determined by '
                    'the backend. ' 
                    'Changing the value of this configuration at runtime will '
                    'not take effect.'
    )
    sparse_enable_jit: Optional[bool] = ConfigField(
        default=None,
        envvar='TENSORKIT_SPARSE_ENABLE_JIT',
        description='Whether or not to enable JIT engine on sparse functions '
                    'and modules?  If not specified, determined by the backend. '
                    'Changing the value of this configuration at runtime will '
                    'not take effect.'
    )


settings = Settings()
"""The global configuration for TensorKit."""
