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

backend.register_as_tensor.__doc__ = """
    Register a function to convert an object of a custom type into a Tensor.
    
    >>> from typing import Optional
    >>> import numpy as np
    >>> from tensorkit import tensor as T
    
    >>> class MyArray(object):
    ...     def __init__(self, data):
    ...         self.data = data

    >>> def my_array_to_tensor(data: MyArray, dtype: Optional[T.DType]
    ...                        ) -> T.Tensor:
    ...     return T.as_tensor(data.data, dtype)

    >>> T.register_as_tensor(MyArray, my_array_to_tensor)
    >>> t = T.as_tensor(MyArray(np.asarray([1, 2, 3], dtype=np.int32)))
    >>> T.to_numpy(t)
    array([1, 2, 3], dtype=int32)

    Args:
        type_: The custom type.
        convertor: A function ``(data: Any, dtype: DType) -> Tensor``,
            to convert the given `data` into a tensor.
"""

backend.zeros.__doc__ = """
    Construct a tensor with all elements equal to zero.

    >>> from tensorkit import tensor as T
    >>> t = T.zeros([2, 3], dtype=T.float32)
    >>> T.to_numpy(t)
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

    Args:
        shape: The shape of the tensor.
        dtype: The dtype of the tensor.
"""

backend.ones.__doc__ = """
    Construct a tensor with all elements equal to one.

    >>> from tensorkit import tensor as T
    >>> t = T.ones([2, 3], dtype=T.float32)
    >>> T.to_numpy(t)
    array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)

    Args:
        shape: The shape of the tensor.
        dtype: The dtype of the tensor.
"""

backend.arange.__doc__ = """
    Construct a integer sequence tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.arange(3)
    >>> T.to_numpy(t)
    array([0, 1, 2], dtype=int32)

    Args:
        start_or_end: The starting number of the sequence, or the ending number
            if `end` is not specified.
        end: The ending number of the sequence (excluded).
        step: The step size of the sequence.
        dtype: The dtype of the returned tensor.
"""


# shape utils
backend.shape.__doc__ = """
    Get the shape of the given tensor.
    
    >>> from tensorkit import tensor as T
    >>> shape = T.shape(T.zeros([2, 3]))
    >>> isinstance(shape, T.Shape)
    True
    >>> tuple(shape)
    (2, 3)
    
    Args:
        x: The tensor.
"""

backend.rank.__doc__ = """
    Get the rank of the given tensor.
    
    >>> from tensorkit import tensor as T
    >>> T.rank(T.zeros([2, 3]))
    2
    
    Args:
        x: The tensor.
"""

backend.reshape.__doc__ = """
    Reshape the given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.zeros([2, 3, 4])
    >>> t2 = T.reshape(t, [3, 8])
    >>> tuple(T.shape(t2))
    (3, 8)
    
    Args:
        x: The tensor to be reshaped.
        shape: The new shape for the tensor.
"""

backend.repeat.__doc__ = """
    Repeat the given tensor along specified axes.
    
    >>> from tensorkit import tensor as T
    >>> t = T.reshape(T.arange(3), [1, 3])
    >>> T.to_numpy(t)
    array([[0, 1, 2]], dtype=int32)
    >>> t2 = T.repeat(t, [1, 3, 2])
    >>> tuple(T.shape(t2))
    (1, 3, 6)
    >>> T.to_numpy(t2)
    array([[[0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2]]], dtype=int32)

    Args:
        x: The tensor to be repeated.
        repeats: The repeat number of each axis.
"""

backend.expand.__doc__ = """
    Expand the given tensor along specified axes.
    
    Unlike `repeat`, only axis with size 1 can be expanded via this function.
    Also, the specified argument should be desired shape, rather than the
    repeat numbers.
    
    >>> from tensorkit import tensor as T
    >>> t = T.reshape(T.arange(3), [1, 3])
    >>> T.to_numpy(t)
    array([[0, 1, 2]], dtype=int32)
    >>> t2 = T.expand(t, [1, 2, -1])
    >>> tuple(T.shape(t2))
    (1, 2, 3)
    >>> T.to_numpy(t2)
    array([[[0, 1, 2],
            [0, 1, 2]]], dtype=int32)

    Args:
        x: The tensor to be expanded.
        repeats: The desired shape of the expanded tensor.  `-1` indicates
            not to change the original size of a certain axis.
"""

backend.squeeze.__doc__ = """
    Squeeze `1` s in the shape of a given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.zeros([1, 2, 1, 3, 4, 1])
    >>> tuple(T.shape(T.squeeze(t)))
    (2, 3, 4)
    >>> tuple(T.shape(T.squeeze(t, -1)))
    (1, 2, 1, 3, 4)
    >>> tuple(T.shape(T.squeeze(t, [0, -1])))
    (2, 1, 3, 4)
    
    Args:
        x: The tensor to be squeezed.
        axis: The axis(es) to be squeezed.  If not specified, squeeze all axes.
"""

backend.expand_dim.__doc__ = """
    Insert one dimension into a given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.reshape(T.arange(6), [2, 3])
    >>> T.to_numpy(t)
    array([[0, 1, 2],
           [3, 4, 5]], dtype=int32)
    >>> t2 = T.expand_dim(t, -2)
    >>> tuple(T.shape(t2))
    (2, 1, 3)
    >>> T.to_numpy(t2)
    array([[[0, 1, 2]],
    <BLANKLINE>
           [[3, 4, 5]]], dtype=int32)
           
    Args:
        x: The tensor into which the dimension should be inserted.
        axis: The index of dimension after insertion.
"""

backend.broadcast_shape.__doc__ = """
    Get the broadcasted shape of two tensor shapes.
    
    >>> from tensorkit import tensor as T
    >>> tuple(T.broadcast_shape([3, 4, 2, 1], [4, 1, 5]))
    (3, 4, 2, 5)
    
    Args:
        x: The first tensor shape.
        y: The second tensor shape.
"""

backend.broadcast_to.__doc__ = """
    Broadcast the shape of a given tensor to the specified shape.
    
    >>> from tensorkit import tensor as T
    >>> t = T.zeros([2, 1])
    >>> t2 = T.broadcast_to(t, [4, 2, 5])
    >>> tuple(T.shape(t2))
    (4, 2, 5)
    
    Args:
        x: The tensor to be broadcast.
        new_shape: The broadcasted new shape.
"""

backend.explicit_broadcast.__doc__ = """
    Broadcast two tensors into the same shape.
    
    >>> from tensorkit import tensor as T
    >>> t1 = T.zeros([2, 1])
    >>> t2 = T.zeros([3, 1, 5])
    >>> t3, t4 = T.explicit_broadcast(t1, t2)
    >>> tuple(T.shape(t3))
    (3, 2, 5)
    >>> tuple(T.shape(t4))
    (3, 2, 5)
    
    Args:
        x: The first tensor.
        y: The second tensor.
"""

backend.flatten_to_ndims.__doc__ = """
    Flatten multiple dimensions of `x` at the front into 1 dimension,
    such that the resulting tensor will have exactly `ndims` dimensions.
    
    >>> from tensorkit import tensor as T
    >>> t = T.arange(24).reshape([2, 3, 4])
    >>> tuple(T.shape(t))
    (2, 3, 4)
    >>> T.to_numpy(t)
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]], dtype=int32)

    >>> t2, s = T.flatten_to_ndims(t, 2)
    >>> tuple(T.shape(t2))
    (6, 4)
    >>> tuple(s)
    (2, 3)
    >>> T.to_numpy(t2)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]], dtype=int32)

    >>> t3 = t2[:, [0, 2]]
    >>> tuple(T.shape(t3))
    (6, 2)
    >>> T.to_numpy(t3)
    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10],
           [12, 14],
           [16, 18],
           [20, 22]], dtype=int32)

    >>> t4 = T.unflatten_from_ndims(t3, s)
    >>> tuple(T.shape(t4))
    (2, 3, 2)
    >>> T.to_numpy(t4)
    array([[[ 0,  2],
            [ 4,  6],
            [ 8, 10]],
    <BLANKLINE>
           [[12, 14],
            [16, 18],
            [20, 22]]], dtype=int32)

    Args:
        x: The tensor to be flatten.
        ndims: The number of dimensions of the resulting tensor.

    Returns:
        A tuple of ``(output_tensor, front_shape)``.  Passing this tuple to
        :func:`unflatten_from_ndims` will reshape `output_tensor` back to
        the input tensor `x`.  If `x` does not need to be flatten, then
        `output_tensor` will just be `x` itself, while `front_shape` will
        be :obj:`None`.
"""

backend.unflatten_from_ndims.__doc__ = """
    The inverse transformation of :func:`flatten_to_ndims`.

    If `front_shape` is :obj:`None`, `x` will be returned without any change.

    Args:
        x: The tensor to be unflatten.
        front_shape: The original front shape.
        
    See Also:
        :func:`flatten_to_ndims`
"""


# split / join / indexing / gathering
backend.index_select.__doc__ = """
    Select elements from `x` according to specified `indices`.
    
    The output tensor will have shape
    ``x.shape[: axis] + indices.shape + x.shape[axis+1:]``.
    
    >>> from tensorkit import tensor as T
    >>> t = T.arange(12).reshape([3, 4])
    >>> T.to_numpy(t)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=int32)
    
    >>> T.to_numpy(T.index_select(t, 1))
    array([4, 5, 6, 7], dtype=int32)

    >>> T.to_numpy(T.index_select(t, [0, 2, 1], axis=0))
    array([[ 0,  1,  2,  3],
           [ 8,  9, 10, 11],
           [ 4,  5,  6,  7]], dtype=int32)

    >>> T.to_numpy(T.index_select(t, [[0, 2, 1], [1, 2, 0]], axis=-1))
    array([[[ 0,  2,  1],
            [ 1,  2,  0]],
    <BLANKLINE>
           [[ 4,  6,  5],
            [ 5,  6,  4]],
    <BLANKLINE>
           [[ 8, 10,  9],
            [ 9, 10,  8]]], dtype=int32)
    
    Args:
        x: The tensor, where to select elements.
        indices: The element indices tensor.
            Some backend may not support negative indices.
        axis: Along which axis to select the elements.  Defaults to 0.
"""


# read / assign
backend.to_numpy.__doc__ = """
    Read the value of the given tensor from the device into a NumPy array.
    
    Args:
        x: The tensor to be read.
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


# sequential math element-wise operations
backend.add_n.__doc__ = """
    Add a sequence of tensors.
    
    Broadcast will be done automatically for adding these tensors.
    
    Args:
        tensors: The sequence of tensors.
"""

