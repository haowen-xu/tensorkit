from . import backend


def default_impl(method):
    name = method.__name__
    assert(name is not None)

    if hasattr(backend, name):
        return getattr(backend, name)
    else:
        return method
