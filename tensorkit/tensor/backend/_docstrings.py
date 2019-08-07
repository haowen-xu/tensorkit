import operator

import numpy as np

from . import backend

__all__ = []


################
# core package #
################

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

backend.to_numpy_bool.__doc__ = """
    Read the value of the given tensor from the device into a NumPy array.
    The tensor is regarded as a boolean tensor.
    
    Args:
        x: The tensor to be read.
"""


# univariate element-wise math operations
def _f(method, name=None, expr=None):
    if name is None:
        name = method.__name__
    if expr is None:
        expr = rf'\\{name}(x)'
    method.__doc__ = f"""
    Compute the output of element-wise :math:`{expr}`.
    
    Args:
        x: The input tensor.
    """


_f(backend.abs, expr='|x|')
_f(backend.neg, expr='-x')
_f(backend.exp)
_f(backend.log)
_f(backend.log1p, expr='\\log(1+x)')
_f(backend.sin)
_f(backend.cos)
_f(backend.square, expr='x ^ 2')


# bivariate element-wise math operations
def _f(method, name=None, expr=None):
    if name is None:
        name = method.__name__
    if expr is None:
        expr = rf'\\{name}(x,y)'
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
_f(backend.floordiv, expr='\\rfloor x / y \\rfloor')

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


# reduction operations
def _f(method, name=None, expr=None, desc=None):
    if name is None:
        name = method.__name__
    if expr is None:
        expr = rf'\\{name}(x)'
    if desc is not None:
        desc = f'\n{desc}'
    else:
        desc = ''
    method.__doc__ = f"""
    Compute :math:`{expr}` along specified dimension.{desc}

    Args:
        x: The input tensor.
        axis: The axis for computing :math:`{expr}`.  If not specified,
            all dimensions will be considered.
        keepdims: Whether or not to keep the reduced dimension?
            Defaults to :obj:`False`.
    """


_f(backend.reduce_sum, name='sum')
_f(backend.reduce_mean, name='mean')
_f(backend.reduce_max, name='max')
_f(backend.reduce_min, name='min')
_f(backend.log_sum_exp, expr='\\log \\sum_{k=1}^K \\exp(x_k)',
   desc="""
    .. math::

        \\begin{align*}
            \\log \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max})
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}
   """)
_f(backend.log_mean_exp, expr='\\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)',
   desc="""
    .. math::

        \\begin{align*}
            \\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)
                &= \\log \\left[\\exp(x_{max}) \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max})\\right] \\\\
                &= x_{max} + \\log \\frac{1}{K}
                    \\sum_{k=1}^K \\exp(x_k - x_{max}) \\\\
            x_{max} &= \\max x_k
        \\end{align*}
   """)


# logical operations
backend.to_boolean.__doc__ = """
    Convert a tensor into boolean tensor.
    
    The output tensor will have `T.boolean` dtype.  Some backend may not
    have a dedicated boolean dtype, and it may be an alias of another integer
    type, e.g., it might be `T.uint8`.

    >>> from tensorkit import tensor as T
    >>> t = T.to_boolean([True, False])
    >>> t.dtype == T.boolean
    True

    Args:
        x: The input tensor. 
"""

backend.logical_not.__doc__ = """
    Compute the element-wise logical not of a given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.to_boolean([True, False])
    >>> t2 = T.logical_not(t)
    >>> T.to_numpy(t2).astype(np.bool)
    array([False,  True])
    
    Args:
        x: The input tensor.
"""


def _f(method, name, op):
    x = np.asarray([True, True, False, False])
    y = np.asarray([True, False, False, True])
    t = op(x, y)

    method.__doc__ = f"""
    Compute element-wise logical {name} of two given tensors.
    
    >>> from tensorkit import tensor as T
    >>> x = T.to_boolean([True, True, False, False])
    >>> y = T.to_boolean([True, False, False, True])
    >>> t = T.logical_{name}(x, y)
    >>> T.to_numpy(t).astype(np.bool)
    {repr(t)}

    Args:
        x: The first input tensor.
        y: The second input tensor.
    """


_f(backend.logical_and, 'and', operator.and_)
_f(backend.logical_or, 'or', operator.or_)
_f(backend.logical_xor, 'xor', operator.xor)


# comparison operations
def _f(method, expr, op, out_processor='.astype(np.bool)'):
    x = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    y = np.asarray([0, 3, 1, 2], dtype=np.int32)
    t = op(x, y)
    t_repr = '\n    '.join(repr(t).split('\n'))

    method.__doc__ = f"""
    Compute element-wise `{expr}` of two given tensors.

    >>> from tensorkit import tensor as T
    >>> x = T.as_tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=T.int32)
    >>> y = T.as_tensor([0, 3, 1, 2], dtype=T.int32)
    >>> t = T.{method.__name__}(x, y)
    >>> T.to_numpy(t){out_processor}
    {t_repr}

    Args:
        x: The first input tensor.
        y: The second input tensor.
    """


_f(backend.equal, 'x == y', operator.eq)
_f(backend.not_equal, 'x != y', operator.ne)
_f(backend.less, 'x < y', operator.lt)
_f(backend.less_equal, 'x <= y', operator.le)
_f(backend.greater, 'x > y', operator.gt)
_f(backend.greater_equal, 'x >= y', operator.ge)
_f(backend.minimum, 'min(x, y)', np.minimum, '')
_f(backend.maximum, 'max(x, y)', np.maximum, '')

backend.clip.__doc__ = """
    Clip the values in the given tensor to a specified range.
    
    >>> from tensorkit import tensor as T
    >>> x = T.as_tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    >>> t = T.clip(x, 0.15, 0.7)
    >>> T.to_numpy(t)
    array([[0.15, 0.2 , 0.3 , 0.4 ],
           [0.5 , 0.6 , 0.7 , 0.7 ]], dtype=float32)
    
    Args:
        x: The input tensor.
        x_min: The minimum value.
        x_max: The maximum value.
"""


# gradient utilities
backend.requires_grad.__doc__ = """
    Set a given tensor to require gradient propagation.
    
    Args:
        x: The tensor.
        
    Returns:
        `x` itself.
"""

backend.clear_grad.__doc__ = """
    Clear the accumulated gradients at a given tensor.
    
    Args:
        x: The tensor.
        
    Returns:
        `x` itself.
"""

backend.back_prop.__doc__ = """
    Back-propagate gradients from a given tensor.
    
    Args:
        x: The tensor.
        
    Returns:
        `x` itself. 
"""

backend.grad.__doc__ = """
    Get the propagated gradients at a given tensor.
    
    Args:
        x: The tensor.
        
    Returns:
        The gradients, or :obj:`None` if no gradient has been propagated.
"""

backend.detach.__doc__ = """
    Detach a tensor from the auto-grad graph, such that no gradient
    will be propagated along the returned tensor.
    
    Args:
        x: The input tensor.
        
    Returns:
        The output detached tensor, different from `x`.
"""


# TensorWrapper
backend.TensorWrapper.__doc__ = """
    Tensor-like object that wraps a `Tensor` instance.

    This class is typically used to implement `super-tensor` classes,
    adding auxiliary methods to a :class:`Tensor`.
    `register_tensor_wrapper_class` should be called to register
    derived classes into TensorKit type system.

    Access to any attributes, properties and methods that does not belong 
    to the wrapper class itself will be transparently proxied to the wrapped
    tensor.
    Also, :class:`TensorWrapper` can be directly used in mathematical
    expressions and TensorKit arithmetic functions.
    For example, ``TensorWrapper(...) + T.exp(TensorWrapper(...))``.

    However, some backend may not support to do math operations with `Tensor`
    object as left-hand operand, and an arbitrary type as right-hand operand.
    For example, ``T.exp(TensorWrapper(...)) + TensorWrapper(...)`` may fail.
    A safe way to do this is to use TensorKit arithmetic functions instead,
    for example, ``T.add(T.exp(TensorWrapper(...)), TensorWrapper(...)``. 

    One thing to notice is that, :class:`TensorWrapper` are usually neither 
    :class:`Tensor` nor sub-classes of :class:`tf.Tensor`, i.e.,
    ``isinstance(TensorWrapper(...), tf.Tensor) == False``.

    All the attributes defined in sub-classes of :class:`TensorWrapper`
    must have names starting with ``_self_``, or defined as class attributes.
    The properties and methods are not restricted by this rule.

    An example of inheriting :class:`TensorWrapper` is shown as follows:

    .. code-block:: python
    
        from tensorkit import tensor as T


        class MyTensorWrapper(T.TensorWrapper):
        
            _flag = None

            def __init__(self, wrapped, flag):
                super(MyTensorWrapper, self).__init__()
                self._self_wrapped = wrapped
                self._flag = flag

            @property
            def tensor(self):
                return self._self_wrapped

            @property
            def flag(self):
                return self._flag

        T.register_tensor_wrapper_class(MyTensorWrapper)

        # tests
        t = MyTensorWrapper(T.as_tensor(0., dtype=T.float32), flag=123)
        assert(T.dtype(t) == T.float32)
        assert(t.flag == 123)
    """

backend.TensorWrapper.tensor.__doc__ = """
    Get the wrapped :class:`Tensor`.
    Derived classes must override this to return the actual wrapped tensor.
    """

backend.TensorWrapper.as_tensor.__doc__ = """
    Convert the :class:`TensorWrapper` object to a :class:`Tensor`.

    Args:
        dtype: The desired dtype of the target tensor.
    """

backend.register_tensor_wrapper_class.__doc__ = """
    Register a sub-class of :class:`TensorWrapper` into TensorKit type system.

    Args:
        cls: The subclass of :class:`TensorWrapper` to be registered.
    """


##############
# nn package #
##############

# activation functions
backend.nn.relu.__doc__ = r"""
    Compute element-wise :math:`\mathrm{ReLU}(x)`.

    .. math::

        \mathrm{ReLU}(x) = \begin{cases}
          x & (x \geq 0) \\
          0 & (x < 0)
        \end{cases}

    Args:
        x: The input tensor.
"""

backend.nn.leaky_relu.__doc__ = r"""
    Compute element-wise :math:`\mathrm{LeakyReLU}(x)`.

    .. math::

        \mathrm{LeakyReLU}(x) = \begin{cases}
          x & (x \geq 0) \\
          ax & (x < 0)
        \end{cases}

    Args:
        x: The input tensor.
        a: The negative slope.  Defaults to 0.01
"""

backend.nn.sigmoid.__doc__ = r"""
    Compute element-wise :math:`\mathrm{sigmoid}(x)`.

    .. math::

        \mathrm{sigmoid}(x) = \frac{\exp(x)}{1 + \exp(x)}

    Args:
        x: The input tensor.
"""

backend.nn.softmax.__doc__ = r"""
    Compute :math:`\mathrm{softmax}(x)` along the specified dimension.

    .. math::

        \mathrm{softmax}(\mathbf{x}) = \frac{\exp(\mathbf{x})}{\sum_i \exp(x_i)}

    Args:
        x: The input tensor.
        axis: Along which dimension to compute softmax.
"""

backend.nn.log_softmax.__doc__ = r"""
    Compute :math:`\log\mathrm{softmax}(x)` along the specified dimension.

    .. math::

        \log\mathrm{softmax}(\mathbf{x}) = 
            \log\frac{\exp(\mathbf{x})}{\sum_i \exp(x_i)}

    Args:
        x: The input tensor.
        axis: Along which dimension to compute softmax.
"""


# objective functions
backend.nn.binary_cross_entropy_with_logits.__doc__ = r"""
    Compute the binary cross entropy :math:`-(x \log p + (1-x) \log (1-p))`.
    
    Args:
        logits: The logits of `p`.
            :math:`p = \frac{\exp(logits)}{1 + \exp(logits)}`.
        labels: The integer or floating point label `x`, whose values should
            be within range `[0, 1]`.
        reduction: One of `{'none', 'sum', 'mean'}`.  If 'sum', the output
            will be summed to a scalar.  If 'mean', the output will be averaged.
        negative: Whether or not to take negative on the output?
            If :obj:`True`, will compute :math:`x \log p + (1-x) \log (1-p)`.
"""

backend.nn.cross_entropy_with_logits.__doc__ = r"""
    Compute the categorical cross entropy :math:`-\sum_{i=1}^k I(x=i) \log p_i`.

    Args:
        logits: The logits of `p_i`.
            :math:`p_i = \frac{\exp(logits_i)}{\sum_j^k \exp(logits_j)}`.
        labels: The integer label `x`, whose values should be integers
            from the set `{0, 1, ..., k-1}`.  `labels.shape` must be
            broadcastable against `logits.shape[:-1]`.
        reduction: One of `{'none', 'sum', 'mean'}`.  If 'sum', the output
            will be summed to a scalar.  If 'mean', the output will be averaged.
        negative: Whether or not to take negative on the output?
            If :obj:`True`, will compute :math:`\sum_{i=1}^k I(x=i) \log p_i`.
"""

backend.nn.sparse_cross_entropy_with_logits.__doc__ = r"""
    Compute the categorical cross entropy :math:`-\sum_{i=1}^k x_i \log p_i`.

    Args:
        logits: The logits of `p_i`.
            :math:`p_i = \frac{\exp(logits_i)}{\sum_j^k \exp(logits_j)}`.
        labels: The integer or floating-point label `x`.
            `labels.shape` must be broadcastable against `logits.shape`.
            If it is an integer tensor, its values must be one-hot encoded
            vectors.  If it is a floating-point tensor, its values must be
            non-negative vectors summed up to 1.
        reduction: One of `{'none', 'sum', 'mean'}`.  If 'sum', the output
            will be summed to a scalar.  If 'mean', the output will be averaged.
        negative: Whether or not to take negative on the output?
            If :obj:`True`, will compute :math:`\sum_{i=1}^k x_i \log p_i`.
"""


# tensor transformations
backend.nn.one_hot.__doc__ = """
    Construct a one-hot encoded tensor from the given indexing tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.as_tensor([[3, 1], [0, 2]])
    >>> t2 = T.nn.one_hot(t, 4)
    >>> T.to_numpy(t2)
    array([[[0, 0, 0, 1],
            [0, 1, 0, 0]],
    <BLANKLINE>
           [[1, 0, 0, 0],
            [0, 0, 1, 0]]])
    
    Args:
        x: The indexing tensor, whose values should be integers from the set
            `{0, 1, ..., n_classes-1}`.
        n_classes: The number of components of the one-hot vectors.
        dtype: The dtype of the returned tensor.
"""


##################
# random package #
##################
backend.random.seed.__doc__ = """
    Set the global random seed for the backend.
    
    The user may still need to set the seed for Python system libraries and
    other third-party libraries, for example:
    
    >>> import random
    >>> import numpy as np
    >>> from tensorkit import tensor as T

    >>> random.seed(1234)
    >>> np.random.seed(1234)
    >>> T.random.seed(1234)
    
    Args:
        seed: The random seed.
"""

backend.random.new_state.__doc__ = """
    Create a new `RandomState` object.
    
    Random states with the same initial seed is guaranteed to deduce same
    random outputs.  For example::
    
        from tensorkit import tensor as T

        rs1 = T.random.new_state(1234)
        output1 = T.random.randn([2, 3, 4], random_state=rs1)

        rs2 = T.random.new_state(1234)
        output2 = T.random.randn([2, 3, 4], random_state=rs2)
        
        # output1 and output2 are guaranteed to be identical
    
    However, it is not guaranteed to have same outputs with the global seed.
    For example::
    
        from tensorkit import tensor as T

        T.random.seed(1234)
        output1 = T.random.randn([2, 3, 4])

        rs = T.random.new_state(1234)
        output2 = T.random.randn([2, 3, 4], random_state=rs)
        
        # output1 and output2 are not guaranteed to be the same
    
    
    Args:
        seed: The initial seed for the random state.
"""

backend.random.normal.__doc__ = r"""
    Generate :math:`\mathcal{N}(\mu,\sigma^2)` distributed random numbers.
    
    .. math::
    
        p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(
                -\frac{(x-\mu)^2}{2 \sigma^2}
            \right)
            
    Args:
        mean: The mean (:math:`\mu`) of normal distribution.
        std: The standard deviation (:math:`\sigma`) of normal distribution.
        random_state: The optional random state object. 
"""

backend.random.randn.__doc__ = r"""
    Generate :math:`\mathcal{N}(0,1)` distributed random numbers.
    
    .. math::
    
        p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{x^2}{2}\right)

    Args:
        shape: The shape of the returned tensor.
        dtype: The dtype of the returned tensor.
        random_state: The optional random state object. 
"""

backend.random.bernoulli.__doc__ = r"""
    Generate bernoulli random numbers.
    
    .. math::
    
        p(x) = p^{I(x=1)} (1-p)^{I(x = 0)}
        
    Args:
        logits: The logits of `p`.
            :math:`p = \frac{\exp(logits)}{1 + \exp(logits)}`.
            Either `logits` or `probs` must be specified, but not both.
        probs: The `p`.
        n_samples: The number of samples to take for each `logits` or `probs`.
            If specified, the returned tensor will have shape
            ``(n_samples,) + (logits or probs).shape``.
        dtype: The dtype of the returned tensor.
        random_state: The optional random state object.
"""

backend.random.categorical.__doc__ = r"""
    Generate categorical random numbers.
    
    .. math::
    
        p(x) = \prod_{i=1}^k p_i^{I(x=i)}
        
    Args:
        logits: The logits of `p_i`.
            :math:`p_i = \frac{\exp(logits_i)}{\sum_j^k \exp(logits_j)}`.
            Either `logits` or `probs` must be specified, but not both.
        probs: The `p_i`.
        n_samples: The number of samples to take for each `logits` or `probs`.
            If specified, the returned tensor will have shape
            ``(n_samples,) + (logits or probs).shape[:-1]``.
        dtype: The dtype of the returned tensor.
        random_state: The optional random state object.
"""
