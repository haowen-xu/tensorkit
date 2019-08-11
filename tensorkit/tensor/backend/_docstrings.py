import operator

import numpy as np

from . import backend

__all__ = []


def _s(*args):
    doc = args[-1]
    for m in args[:-1]:
        qualname = getattr(m, '__qualname__', None)
        if qualname and not qualname.startswith('tensorkit'):
            continue
        if not m.__doc__ and callable(m):
            m.__doc__ = doc


################
# core package #
################

# jit
_s(backend.jit, """
    Compile the decorated function if the backend provides JIT engine, 
    and if `tensorkit.settings.disable_jit` is :obj:`False`.

    Args:
        fn: The function to be compiled.  It must be a function, not
            a class method.
""")

_s(backend.float_x, """
    Get the default dtype for floating numbers, as configured in
    `tensorkit.settings.float_x`.
""")

_s(backend.is_floating_point, """
    Query whether or not the specified DType is a float DType.

    >>> from tensorkit import tensor as T

    >>> T.is_floating_point(T.float32)
    True
    >>> T.is_floating_point(T.int32)
    False

    Args: 
        dtype: The queried DType.
""")


_s(backend.cast, """
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
""")

_s(backend.dtype, """
    Get the DType of the input type.

    >>> import numpy as np
    >>> from tensorkit import tensor as T

    >>> t = T.as_tensor(np.random.randn(2, 3).astype(np.float32)) 
    >>> T.dtype(t) is T.float32
    True

    Args:
        x: The input tensor.
""")


# tensor constructors
_s(backend.as_tensor, """
    Convert arbitrary data into a Tensor.
    
    >>> import numpy as np
    >>> from tensorkit import tensor as T
    
    >>> t = T.as_tensor([1, 2, 3], dtype=T.int32)
    >>> isinstance(t, T.Tensor)
    True
    >>> T.shape(t)
    [3]
    >>> t.dtype is T.int32
    True
    >>> T.to_numpy(t)
    array([1, 2, 3], dtype=int32)

    Args:
        data: The data to be converted.
        dtype: Cast the data into this DType.
""")

_s(backend.zeros, """
    Construct a tensor with all elements equal to zero.

    >>> from tensorkit import tensor as T
    >>> t = T.zeros([2, 3], dtype=T.float32)
    >>> T.to_numpy(t)
    array([[0., 0., 0.],
           [0., 0., 0.]], dtype=float32)

    Args:
        shape: The shape of the tensor.
        dtype: The dtype of the tensor.
""")

_s(backend.ones, """
    Construct a tensor with all elements equal to one.

    >>> from tensorkit import tensor as T
    >>> t = T.ones([2, 3], dtype=T.float32)
    >>> T.to_numpy(t)
    array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)

    Args:
        shape: The shape of the tensor.
        dtype: The dtype of the tensor.
""")

_s(backend.arange, """
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
""")


# shape utils
_s(backend.shape, """
    Get the shape of the given tensor.
    
    >>> from tensorkit import tensor as T
    >>> shape = T.shape(T.zeros([2, 3]))
    >>> shape
    [2, 3]
    
    Args:
        x: The tensor.
""")

_s(backend.rank, """
    Get the rank of the given tensor.
    
    >>> from tensorkit import tensor as T
    >>> T.rank(T.zeros([2, 3]))
    2
    
    Args:
        x: The tensor.
""")

_s(backend.reshape, """
    Reshape the given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.zeros([2, 3, 4])
    >>> t2 = T.reshape(t, [3, 8])
    >>> T.shape(t2)
    [3, 8]
    
    Args:
        x: The tensor to be reshaped.
        shape: The new shape for the tensor.
""")

_s(backend.repeat, """
    Repeat the given tensor along specified axes.
    
    >>> from tensorkit import tensor as T
    >>> t = T.reshape(T.arange(3), [1, 3])
    >>> T.to_numpy(t)
    array([[0, 1, 2]], dtype=int32)
    >>> t2 = T.repeat(t, [1, 3, 2])
    >>> T.shape(t2)
    [1, 3, 6]
    >>> T.to_numpy(t2)
    array([[[0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2],
            [0, 1, 2, 0, 1, 2]]], dtype=int32)

    Args:
        x: The tensor to be repeated.
        repeats: The repeat number of each axis.
""")

_s(backend.expand, """
    Expand the given tensor along specified axes.
    
    Unlike `repeat`, only axis with size 1 can be expanded via this function.
    Also, the specified argument should be desired shape, rather than the
    repeat numbers.
    
    >>> from tensorkit import tensor as T
    >>> t = T.reshape(T.arange(3), [1, 3])
    >>> T.to_numpy(t)
    array([[0, 1, 2]], dtype=int32)
    >>> t2 = T.expand(t, [1, 2, -1])
    >>> T.shape(t2)
    [1, 2, 3]
    >>> T.to_numpy(t2)
    array([[[0, 1, 2],
            [0, 1, 2]]], dtype=int32)

    Args:
        x: The tensor to be expanded.
        repeats: The desired shape of the expanded tensor.  `-1` indicates
            not to change the original size of a certain axis.
""")

_s(backend.squeeze, """
    Squeeze `1` s in the shape of a given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.zeros([1, 2, 1, 3, 4, 1])
    >>> T.shape(T.squeeze(t))
    [2, 3, 4]
    >>> T.shape(T.squeeze(t, -1))
    [1, 2, 1, 3, 4]
    >>> T.shape(T.squeeze(t, [0, -1]))
    [2, 1, 3, 4]
    
    Args:
        x: The tensor to be squeezed.
        axis: The axis(es) to be squeezed.  If not specified, squeeze all axes.
""")

_s(backend.expand_dim, """
    Insert one dimension into a given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.reshape(T.arange(6), [2, 3])
    >>> T.to_numpy(t)
    array([[0, 1, 2],
           [3, 4, 5]], dtype=int32)
    >>> t2 = T.expand_dim(t, -2)
    >>> T.shape(t2)
    [2, 1, 3]
    >>> T.to_numpy(t2)
    array([[[0, 1, 2]],
    <BLANKLINE>
           [[3, 4, 5]]], dtype=int32)
           
    Args:
        x: The tensor into which the dimension should be inserted.
        axis: The index of dimension after insertion.
""")

_s(backend.broadcast_shape, """
    Get the broadcasted shape of two tensor shapes.
    
    >>> from tensorkit import tensor as T
    >>> T.broadcast_shape([3, 4, 2, 1], [4, 1, 5])
    [3, 4, 2, 5]
    
    Args:
        x: The first tensor shape.
        y: The second tensor shape.
""")

_s(backend.broadcast_to, """
    Broadcast the shape of a given tensor to the specified shape.
    
    >>> from tensorkit import tensor as T
    >>> t = T.zeros([2, 1])
    >>> t2 = T.broadcast_to(t, [4, 2, 5])
    >>> T.shape(t2)
    [4, 2, 5]
    
    Args:
        x: The tensor to be broadcast.
        new_shape: The broadcasted new shape.
""")

_s(backend.explicit_broadcast, """
    Broadcast two tensors into the same shape.
    
    >>> from tensorkit import tensor as T
    >>> t1 = T.zeros([2, 1])
    >>> t2 = T.zeros([3, 1, 5])
    >>> t3, t4 = T.explicit_broadcast(t1, t2)
    >>> T.shape(t3)
    [3, 2, 5]
    >>> T.shape(t4)
    [3, 2, 5]
    
    Args:
        x: The first tensor.
        y: The second tensor.
""")

_s(backend.flatten_to_ndims, """
    Flatten multiple dimensions of `x` at the front into 1 dimension,
    such that the resulting tensor will have exactly `ndims` dimensions.
    
    >>> from tensorkit import tensor as T
    >>> t = T.arange(24).reshape([2, 3, 4])
    >>> T.shape(t)
    [2, 3, 4]
    >>> T.to_numpy(t)
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]], dtype=int32)

    >>> t2, s = T.flatten_to_ndims(t, 2)
    >>> T.shape(t2)
    [6, 4]
    >>> s
    [2, 3]
    >>> T.to_numpy(t2)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15],
           [16, 17, 18, 19],
           [20, 21, 22, 23]], dtype=int32)

    >>> t3 = t2[:, [0, 2]]
    >>> T.shape(t3)
    [6, 2]
    >>> T.to_numpy(t3)
    array([[ 0,  2],
           [ 4,  6],
           [ 8, 10],
           [12, 14],
           [16, 18],
           [20, 22]], dtype=int32)

    >>> t4 = T.unflatten_from_ndims(t3, s)
    >>> T.shape(t4)
    [2, 3, 2]
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
""")

_s(backend.unflatten_from_ndims, """
    The inverse transformation of :func:`flatten_to_ndims`.

    If `front_shape` is :obj:`None`, `x` will be returned without any change.

    Args:
        x: The tensor to be unflatten.
        front_shape: The original front shape.
        
    See Also:
        :func:`flatten_to_ndims`
""")


# split / join / indexing / gathering
_s(backend.index_select, """
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
""")


# read / assign
_s(backend.to_numpy, """
    Read the value of the given tensor from the device into a NumPy array.
    
    Args:
        x: The tensor to be read.
""")

_s(backend.to_numpy_bool, """
    Read the value of the given tensor from the device into a NumPy array.
    The tensor is regarded as a boolean tensor.
    
    Args:
        x: The tensor to be read.
""")


# univariate element-wise math operations
def _f(method, expr):
    _s(method, f"""
    Compute the output of element-wise :math:`{expr}`.
    
    Args:
        x: The input tensor.
    """)


_f(backend.abs, '|x|')
_f(backend.neg, '-x')
_f(backend.exp, r'\exp(x)')
_f(backend.log, r'\log(x)')
_f(backend.log1p, r'\log(1+x)')
_f(backend.sin, r'\sin(x)')
_f(backend.cos, r'\cos(x)')
_f(backend.square, 'x ^ 2')


# bivariate element-wise math operations
def _f(method,  expr):
    _s(method, f"""
    Compute the output of element-wise :math:`{expr}`.
    
    If `x` and `y` have different shapes, they will be broadcast first.

    Args:
        x: The 1st input tensor.
        y: The 2nd input tensor.
    """)


_f(backend.add, 'x + y')
_f(backend.sub, 'x - y')
_f(backend.mul, 'x * y')
_f(backend.mod, 'x % y')
_f(backend.pow, 'x ^ y')
_f(backend.floordiv, r'\rfloor x / y \rfloor')

_s(backend.div, backend.truediv, """
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
""")


# sequential math element-wise operations
_s(backend.add_n, """
    Add a sequence of tensors.
    
    Broadcast will be done automatically for adding these tensors.
    
    Args:
        tensors: The sequence of tensors.
""")


# reduction operations
def _f(method, expr, desc=None):
    if desc is not None:
        desc = f'\n{desc}'
    else:
        desc = ''
    _s(method, f"""
    Compute :math:`{expr}` along specified dimension.{desc}

    Args:
        x: The input tensor.
        axis: The axis for computing :math:`{expr}`.  If not specified,
            all dimensions will be considered.
        keepdims: Whether or not to keep the reduced dimension?
            Defaults to :obj:`False`.
    """)


_f(backend.reduce_sum, r'\mathrm{sum}(x)')
_f(backend.reduce_mean, r'\mathrm{mean}(x)')
_f(backend.reduce_max, r'\max(x)')
_f(backend.reduce_min, r'\min(x)')
_f(backend.log_sum_exp, '\\log \\sum_{k=1}^K \\exp(x_k)',
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
_f(backend.log_mean_exp, '\\log \\frac{1}{K} \\sum_{k=1}^K \\exp(x_k)',
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
_s(backend.as_boolean, """
    Convert a tensor into boolean tensor.
    
    The output tensor will have `T.boolean` dtype.  Some backend may not
    have a dedicated boolean dtype, and it may be an alias of another integer
    type, e.g., it might be `T.uint8`.

    >>> from tensorkit import tensor as T
    >>> t = T.as_boolean([True, False])
    >>> t.dtype == T.boolean
    True

    Args:
        x: The input tensor. 
""")

_s(backend.logical_not, """
    Compute the element-wise logical not of a given tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.as_boolean([True, False])
    >>> t2 = T.logical_not(t)
    >>> T.to_numpy(t2).astype(np.bool)
    array([False,  True])
    
    Args:
        x: The input tensor.
""")


def _f(method, name, op):
    x = np.asarray([True, True, False, False])
    y = np.asarray([True, False, False, True])
    t = op(x, y)

    _s(method, f"""
    Compute element-wise logical {name} of two given tensors.
    
    >>> from tensorkit import tensor as T
    >>> x = T.as_boolean([True, True, False, False])
    >>> y = T.as_boolean([True, False, False, True])
    >>> t = T.logical_{name}(x, y)
    >>> T.to_numpy(t).astype(np.bool)
    {repr(t)}

    Args:
        x: The first input tensor.
        y: The second input tensor.
    """)


_f(backend.logical_and, 'and', operator.and_)
_f(backend.logical_or, 'or', operator.or_)
_f(backend.logical_xor, 'xor', operator.xor)


# comparison operations
def _f(method, name, expr, op, out_processor='.astype(np.bool)'):
    x = np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.int32)
    y = np.asarray([0, 3, 1, 2], dtype=np.int32)
    t = op(x, y)
    t_repr = '\n    '.join(repr(t).split('\n'))

    _s(method, f"""
    Compute element-wise `{expr}` of two given tensors.

    >>> from tensorkit import tensor as T
    >>> x = T.as_tensor([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=T.int32)
    >>> y = T.as_tensor([0, 3, 1, 2], dtype=T.int32)
    >>> t = T.{name}(x, y)
    >>> T.to_numpy(t){out_processor}
    {t_repr}

    Args:
        x: The first input tensor.
        y: The second input tensor.
    """)


_f(backend.equal, 'equal', 'x == y', operator.eq)
_f(backend.not_equal, 'not_equal', 'x != y', operator.ne)
_f(backend.less, 'less', 'x < y', operator.lt)
_f(backend.less_equal, 'less_equal', 'x <= y', operator.le)
_f(backend.greater, 'greater', 'x > y', operator.gt)
_f(backend.greater_equal, 'greater_equal', 'x >= y', operator.ge)
_f(backend.minimum, 'minimum', 'min(x, y)', np.minimum, '')
_f(backend.maximum, 'maximum', 'max(x, y)', np.maximum, '')

_s(backend.clip, """
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
""")


# gradient utilities
_s(backend.requires_grad, """
    Set a given tensor to require gradient propagation.
    
    Args:
        x: The tensor.
        
    Returns:
        `x` itself.
""")

_s(backend.clear_grad, """
    Clear the accumulated gradients at a given tensor.
    
    Args:
        x: The tensor.
        
    Returns:
        `x` itself.
""")

_s(backend.back_prop, """
    Back-propagate gradients from a given tensor.
    
    Args:
        x: The tensor.
        
    Returns:
        `x` itself. 
""")

_s(backend.grad, """
    Get the propagated gradients at a given tensor.
    
    Args:
        x: The tensor.
        
    Returns:
        The gradients, or :obj:`None` if no gradient has been propagated.
""")

_s(backend.detach, """
    Detach a tensor from the auto-grad graph, such that no gradient
    will be propagated along the returned tensor.
    
    Args:
        x: The input tensor.
        
    Returns:
        The output detached tensor, different from `x`.
""")


# # TensorWrapper
# _s(backend.TensorWrapper, """
#     Tensor-like object that wraps a `Tensor` instance.
#
#     This class is typically used to implement `super-tensor` classes,
#     adding auxiliary methods to a :class:`Tensor`.
#     `register_tensor_wrapper_class` should be called to register
#     derived classes into TensorKit type system.
#
#     Access to any attributes, properties and methods that does not belong
#     to the wrapper class itself will be transparently proxied to the wrapped
#     tensor.
#     Also, :class:`TensorWrapper` can be directly used in mathematical
#     expressions and TensorKit arithmetic functions.
#     For example, ``TensorWrapper(...) + T.exp(TensorWrapper(...))``.
#
#     However, some backend may not support to do math operations with `Tensor`
#     object as left-hand operand, and an arbitrary type as right-hand operand.
#     For example, ``T.exp(TensorWrapper(...)) + TensorWrapper(...)`` may fail.
#     A safe way to do this is to use TensorKit arithmetic functions instead,
#     for example, ``T.add(T.exp(TensorWrapper(...)), TensorWrapper(...)``.
#
#     One thing to notice is that, :class:`TensorWrapper` are usually neither
#     :class:`Tensor` nor sub-classes of :class:`tf.Tensor`, i.e.,
#     ``isinstance(TensorWrapper(...), tf.Tensor) == False``.
#
#     All the attributes defined in sub-classes of :class:`TensorWrapper`
#     must have names starting with ``_self_``, or defined as class attributes.
#     The properties and methods are not restricted by this rule.
#
#     An example of inheriting :class:`TensorWrapper` is shown as follows:
#
#     .. code-block:: python
#
#         from tensorkit import tensor as T
#
#
#         class MyTensorWrapper(T.TensorWrapper):
#
#             _flag = None
#
#             def __init__(self, wrapped, flag):
#                 super(MyTensorWrapper, self).__init__()
#                 self._self_wrapped = wrapped
#                 self._flag = flag
#
#             @property
#             def tensor(self):
#                 return self._self_wrapped
#
#             @property
#             def flag(self):
#                 return self._flag
#
#         T.register_tensor_wrapper_class(MyTensorWrapper)
#
#         # tests
#         t = MyTensorWrapper(T.as_tensor(0., dtype=T.float32), flag=123)
#         assert(T.dtype(t) == T.float32)
#         assert(t.flag == 123)
# """)
#
# _s(backend.TensorWrapper.as_tensor, """
#     Convert the :class:`TensorWrapper` object to a :class:`Tensor`.
#
#     Args:
#         dtype: If specified, cast the tensor into this desired dtype.
# """)
#
# _s(backend.register_tensor_wrapper_class, """
#     Register a sub-class of :class:`TensorWrapper` into TensorKit type system.
#
#     Args:
#         cls: The subclass of :class:`TensorWrapper` to be registered.
# """)


# activation functions
_s(backend.relu, r"""
    Compute element-wise :math:`\mathrm{ReLU}(x)`.

    .. math::

        \mathrm{ReLU}(x) = \begin{cases}
          x & (x \geq 0) \\
          0 & (x < 0)
        \end{cases}

    Args:
        x: The input tensor.
""")

_s(backend.leaky_relu, r"""
    Compute element-wise :math:`\mathrm{LeakyReLU}(x)`.

    .. math::

        \mathrm{LeakyReLU}(x) = \begin{cases}
          x & (x \geq 0) \\
          ax & (x < 0)
        \end{cases}

    Args:
        x: The input tensor.
        a: The negative slope.  Defaults to 0.01
""")

_s(backend.sigmoid, r"""
    Compute element-wise :math:`\mathrm{sigmoid}(x)`.

    .. math::

        \mathrm{sigmoid}(x) = \frac{\exp(x)}{1 + \exp(x)}

    Args:
        x: The input tensor.
""")

_s(backend.softmax, r"""
    Compute :math:`\mathrm{softmax}(x)` along the specified dimension.

    .. math::

        \mathrm{softmax}(\mathbf{x}) = \frac{\exp(\mathbf{x})}{\sum_i \exp(x_i)}

    Args:
        x: The input tensor.
        axis: Along which dimension to compute softmax.
""")

_s(backend.log_softmax, r"""
    Compute :math:`\log\mathrm{softmax}(x)` along the specified dimension.

    .. math::

        \log\mathrm{softmax}(\mathbf{x}) = 
            \log\frac{\exp(\mathbf{x})}{\sum_i \exp(x_i)}

    Args:
        x: The input tensor.
        axis: Along which dimension to compute softmax.
""")


# objective functions
_s(backend.binary_cross_entropy_with_logits, r"""
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
""")

_s(backend.cross_entropy_with_logits, r"""
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
""")

_s(backend.sparse_cross_entropy_with_logits, r"""
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
""")


# tensor transformations
_s(backend.one_hot, """
    Construct a one-hot encoded tensor from the given indexing tensor.
    
    >>> from tensorkit import tensor as T
    >>> t = T.as_tensor([[3, 1], [0, 2]])
    >>> t2 = T.one_hot(t, 4)
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
""")


# random utilities
_s(backend.random_seed, """
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
""")

_s(backend.random_normal, r"""
    Generate :math:`\mathcal{N}(\mu,\sigma^2)` distributed random numbers.
    
    .. math::
    
        p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(
                -\frac{(x-\mu)^2}{2 \sigma^2}
            \right)
            
    Args:
        mean: The mean (:math:`\mu`) of normal distribution.
        std: The standard deviation (:math:`\sigma`) of normal distribution.
        random_state: The optional random state object. 
""")

_s(backend.randn, r"""
    Generate :math:`\mathcal{N}(0,1)` distributed random numbers.
    
    .. math::
    
        p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{x^2}{2}\right)

    Args:
        shape: The shape of the returned tensor.
        dtype: The dtype of the returned tensor.
        random_state: The optional random state object. 
""")

_s(backend.bernoulli, r"""
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
""")

_s(backend.categorical, r"""
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
""")
