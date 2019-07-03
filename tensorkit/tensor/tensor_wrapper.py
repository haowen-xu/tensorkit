from typing import *

__all__ = ['TensorWrapper', 'register_tensor_wrapper_class']


class TensorWrapper(object):

    @property
    def tensor(self) -> 'B.Tensor':
        raise NotImplementedError()

    def as_tensor(self, dtype: 'B.DType' = None) -> 'B.Tensor':
        t = self.tensor
        if dtype is not None:
            t = B.cast(t, dtype=dtype)
        return t

    # mimic `tf.Tensor` interface
    def __dir__(self):
        ret = list(set(dir(self.tensor) + list(object.__dir__(self))))
        return ret

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __setattr__(self, name, value):
        if name.startswith('_self_'):
            object.__setattr__(self, name, value)
        elif hasattr(type(self), name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.tensor, name, value)

    def __delattr__(self, name):
        if name.startswith('_self_'):
            object.__delattr__(self, name)
        elif hasattr(type(self), name):
            object.__delattr__(self, name)
        else:
            delattr(self.tensor, name)

    def __iter__(self):
        return iter(self.tensor)

    def __bool__(self):
        return bool(self.tensor)

    # overloading arithmetic operations
    def __abs__(self):
        return B.abs(self.tensor)

    def __neg__(self):
        return B.neg(self.tensor)

    def __add__(self, other):
        return B.add(self.tensor, other)

    def __radd__(self, other):
        return B.add(other, self.tensor)

    def __sub__(self, other):
        return B.sub(self.tensor, other)

    def __rsub__(self, other):
        return B.sub(other, self.tensor)

    def __mul__(self, other):
        return B.mul(self.tensor, other)

    def __rmul__(self, other):
        return B.mul(other, self.tensor)

    def __div__(self, other):
        return B.div(self.tensor, other)

    def __rdiv__(self, other):
        return B.div(other, self.tensor)

    def __truediv__(self, other):
        return B.truediv(self.tensor, other)

    def __rtruediv__(self, other):
        return B.truediv(other, self.tensor)

    def __floordiv__(self, other):
        return B.floordiv(self.tensor, other)

    def __rfloordiv__(self, other):
        return B.floordiv(other, self.tensor)

    def __mod__(self, other):
        return B.mod(self.tensor, other)

    def __rmod__(self, other):
        return B.mod(other, self.tensor)

    def __pow__(self, other):
        return B.pow(self.tensor, other)

    def __rpow__(self, other):
        return B.pow(other, self.tensor)

    # logical operations
    def __invert__(self):
        return B.invert(self.tensor)

    def __and__(self, other):
        return B.and_(self.tensor, other)

    def __rand__(self, other):
        return B.and_(other, self.tensor)

    def __or__(self, other):
        return B.or_(self.tensor, other)

    def __ror__(self, other):
        return B.or_(other, self.tensor)

    def __xor__(self, other):
        return B.xor(self.tensor, other)

    def __rxor__(self, other):
        return B.xor(other, self.tensor)

    # boolean operations
    def __lt__(self, other):
        return B.less(self.tensor, other)

    def __le__(self, other):
        return B.less_equal(self.tensor, other)

    def __gt__(self, other):
        return B.greater(self.tensor, other)

    def __ge__(self, other):
        return B.greater_equal(self.tensor, other)

    # slicing and indexing
    def __getitem__(self, item):
        return (B.as_tensor(self.tensor))[item]


def register_tensor_wrapper_class(cls: Type[TensorWrapper]):
    # nothing should be done, all is okay
    pass


from . import backend as B
