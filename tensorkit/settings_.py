from typing import *

from mltk import Config, ConfigField

__all__ = ['Settings', 'settings']


KNOWN_BACKENDS = ('pytorch', 'tensorflow')


def auto_choose_backend() -> Optional[str]:
    """
    Choose the backend automatically.

    If the dependencies of one backend has been imported, then it will be
    chosen as the preferred backend.  If the dependencies of multiple backends
    have been imported, then the backend will be chosen according to the
    following priority order:

        pytorch, tensorflow

    Returns:
        The backend name, or `None` if no backend can be automatically chosen.
    """
    import sys

    activated_backends = []

    for name, module in sys.modules.items():
        for backend in KNOWN_BACKENDS:
            if backend in name and backend not in activated_backends:
                activated_backends.append(backend)

    for backend in KNOWN_BACKENDS:
        if backend in activated_backends:
            return backend


class Settings(Config):

    backend: str = ConfigField(
        str,
        default=auto_choose_backend() or 'pytorch',
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
    check_numerics: bool = ConfigField(
        bool,
        default=False,
        envvar='TENSORKIT_CHECK_NUMERICS',
        description='Whether or not to check the numerical issues of '
                    'computational outputs?  Changing the value of this '
                    'configuration at runtime may not take effect.'
    )
    prefer_backend_impl: bool = ConfigField(
        bool,
        default=True,
        envvar='TENSORKIT_PREFER_BACKEND_IMPL',
        description='Whether or not to prefer using backend implementation '
                    'rather than tensorkit implementation?'
                    'Changing the value of this configuration at runtime '
                    'will not take effect.'
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
