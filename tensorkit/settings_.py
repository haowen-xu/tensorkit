from typing import *

from mltk import Config, ConfigField

__all__ = ['Settings', 'settings']


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


class Settings(Config):

    backend: str = ConfigField(
        str,
        default=auto_choose_backend() or KNOWN_BACKENDS[0],
        choices=KNOWN_BACKENDS,
        envvar='TENSORKIT_BACKEND',
        description='The backend to use.'
                    'Changing the value of this configuration at runtime '
                    'will not take effect.'
    )
    float_x: str = ConfigField(
        str,
        default='float32',
        choices=['float32', 'float64'],
        envvar='TENSORKIT_FLOATX',
        description='The default dtype for floating-point numbers. '
                    'Changing the value of this configuration at runtime may '
                    'not take effect.'
    )
    validate_tensors: bool = ConfigField(
        bool,
        default=False,
        envvar='TENSORKIT_VALIDATE_TENSORS',
        description='Whether or not to perform time-consuming validation on '
                    'tensors for input arguments of functions or classes, '
                    'and on intermediate computation results, to ensure there '
                    'are no numerical issues (i.e., no NaN or Infinity values), '
                    'and no semantic or logical errors (e.g., `low` > `high`)?'
    )
    disable_jit: bool = ConfigField(
        bool,
        default=False,
        envvar='TENSORKIT_DISABLE_JIT',
        description='Whether or not to disable the JIT engine of backend?'
                    'Changing the value of this configuration at runtime '
                    'will not take effect.'
    )


settings = Settings()
"""The global configuration for TensorKit."""
