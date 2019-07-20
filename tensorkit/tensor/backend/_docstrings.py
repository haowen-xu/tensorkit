from . import backend

__all__ = []


# jit
backend.jit.__doc__ = """
    Compile the decorated function if the backend provides JIT engine, 
    and if `tensorkit.settings.disable_jit` is :obj:`False`.

    Args:
        fn: The function to be compiled.  It must be a function, not
            a class method.
"""


# typing
backend.as_shape.__doc__ = """
    Convert `s` into a backend tensor shape object.

    >>> from tensorkit import tensor as T

    >>> isinstance(T.as_shape([1, 2]), T.Shape)
    True
    >>> tuple(T.as_shape([1, 2]))
    (1, 2)

    Args:
        s: A sequence of integers, interpreted as a tensor shape.
"""


# dtypes
backend.as_dtype.__doc__ = """
    Get the DType for specified dtype-like object.

    >>> import numpy as np
    >>> from tensorkit import tensor as T

    >>> T.as_dtype('float32') is T.float32
    True
    >>> T.as_dtype(np.int64) is T.int64
    True

    Args:
        dtype: The DType-like input, e.g., T.int32, "float32", np.int64.
"""

backend.float_x.__doc__ = """
    Get the default float DType, as configured in `tensorkit.settings.float_x`.

    >>> from tensorkit import tensor as T, settings

    >>> settings.float_x = 'float64'
    >>> T.float_x() is T.float64
    True

    >>> settings.float_x = 'float32'
    >>> T.float_x() is T.float32
    True
"""

backend.iinfo.__doc__ = """
    Get the information of the specified integer DType.

    Args:
        dtype: The queried integer DType.
"""

backend.finfo.__doc__ = """
    Get the information of the specified float DType.

    Args:
        dtype: The queried float DType.
"""

backend.is_floating_point.__doc__ = """
    Query whether or not the specified DType is a float DType.

    >>> from tensorkit import tensor as T

    >>> T.is_floating_point(T.float32)
    True
    >>> T.is_floating_point(T.int32)
    False

    Args: 
        dtype: The queried DType.
"""


backend.cast.__doc__ = """
    Cast the input tensor into specified DType.

    >>> import numpy as np
    >>> from tensorkit import tensor as T

    >>> x = T.as_tensor(np.random.randn(2, 3).astype(np.float32))
    >>> T.dtype(x) is T.float32
    True

    >>> y = T.cast(x, T.float64)
    >>> T.dtype(y) is T.float64
    True

    Args:
        x: The input tensor.
        dtype: The target DType.
"""

backend.dtype.__doc__ = """
    Get the DType of the input type.

    >>> import numpy as np
    >>> from tensorkit import tensor as T

    >>> t = T.as_tensor(np.random.randn(2, 3).astype(np.float32)) 
    >>> T.dtype(t) is T.float32
    True

    Args:
        x: The input tensor.
"""


# tensor constructors
backend.as_tensor.__doc__ = """
    Convert arbitrary data into a Tensor.
    
    >>> import numpy as np
    >>> from tensorkit import tensor as T
    
    >>> t = T.as_tensor([1, 2, 3], dtype=T.int32)
    >>> isinstance(t, T.Tensor)
    True
    >>> tuple(t.shape)
    (3,)
    >>> t.dtype is T.int32
    True
    >>> T.to_numpy(t)
    array([1, 2, 3], dtype=int32)

    Args:
        data: The data to be converted.
        dtype: Cast the data into this DType.
"""


# univariate element-wise math operations
def _f(method, name=None, expr=None):
    if name is None:
        name = method.__name__
    if expr is None:
        expr = f'\\\\{name}(x)'
    method.__doc__ = f"""
    Compute the output of element-wise :math:`{expr}`.
    
    Args:
        x: The input tensor.
    """


_f(backend.abs, expr='|x|')
_f(backend.neg, expr='-x')
_f(backend.exp)
_f(backend.log)
_f(backend.log1p, expr='\\\\log(1+x)')
_f(backend.sin)
_f(backend.cos)
_f(backend.square, expr='x ^ 2')


# bivariate element-wise math operations
def _f(method, name=None, expr=None):
    if name is None:
        name = method.__name__
    if expr is None:
        expr = f'\\\\{name}(x,y)'
    method.__doc__ = f"""
    Compute the output of element-wise :math:`{expr}`.
    
    If `x` and `y` have different shapes, they will be broadcast first.

    Args:
        x: The 1st input tensor.
        y: The 2nd input tensor.
    """


_f(backend.add, expr='x + y')
_f(backend.sub, expr='x - y')
_f(backend.mul, expr='x * y')
_f(backend.mod, expr='x % y')
_f(backend.fmod, expr='x % y')
_f(backend.pow, expr='x ^ y')
_f(backend.floordiv, expr='\\\\lfloor x / y \\\\rfloor')

backend.div.__doc__ = backend.truediv.__doc__ = """
    Compute the output of element-wise :math:`x / y`.
    
    If `x` and `y` have different shapes, they will be broadcast first.

    `x` and `y` must have the same `dtype`.  If `x` and `y` are integer tensors,
    they will be first casted into floating-point tensors.  `uint8` and `int16`
    will be casted into `float32`, while other integers will be casted into
    `float64`.  Then the division will be calculated on the casted tensors.
    
    Args:
        x: The 1st input tensor.
        y: The 2nd input tensor.
        
    Raises:
        TypeError: If `x` and `y` have different `dtype`.
"""
