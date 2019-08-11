from typing import Optional, Type, TypeVar

from torch import Tensor, dtype as DType

from ....utils import extend_object, is_extended_by

__all__ = [
    'extended_tensor_supports_property',
    'ExtendedTensor', 'extend_tensor',
    'is_extended_tensor', 'register_extended_tensor_class'
]

extended_tensor_supports_property = False


class ExtendedTensor(Tensor):

    def as_tensor(self, dtype: Optional[DType] = None) -> 'Tensor':
        t = self
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t


TExtendedTensor = TypeVar('TExtendedTensor')


def extend_tensor(tensor_: Tensor,
                  cls_: Type[TExtendedTensor],
                  **kwargs) -> TExtendedTensor:
    if not isinstance(cls_, type) or not issubclass(cls_, ExtendedTensor):
        raise TypeError(f'`cls_` is not a class, or not a subclass of '
                        f'`ExtendedTensor`: got {cls_!r}')
    return extend_object(
        obj_=tensor_,
        cls_=cls_,
        trivial_parents_=(Tensor,),
        **kwargs
    )


def is_extended_tensor(tensor_: Tensor,
                       cls_: Type[ExtendedTensor] = ExtendedTensor) -> bool:
    return is_extended_by(tensor_, cls_)


def register_extended_tensor_class(cls: Type[ExtendedTensor]):
    if not isinstance(cls, type) or not issubclass(cls, ExtendedTensor):
        raise TypeError(f'`cls` is not a class, or not a subclass of '
                        f'`ExtendedTensor`: got {cls!r}')
