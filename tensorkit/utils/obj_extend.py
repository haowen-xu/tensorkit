import inspect
import types
from typing import *

__all__ = ['extend_object', 'is_extended_by']


TClass = TypeVar('TClass')
_extend_object_base_attrs_cache = {}
extension_classes_magic_key = '__ExtendedObject_template_classes__'


def extend_object(obj_: Any,
                  cls_: Type[TClass],
                  trivial_parents_: Tuple[type, ...] = (object,),
                  **kwargs) -> TClass:
    """
    Extend an object `obj` with attributes and methods defined in the given
    template class `cls`.  Only instance methods and static methods are
    supported.  Class methods and properties are not supported.

    An example of using simple template class without `init_extension()` method:

    >>> class A(object):
    ...     pass

    >>> class B(object):
    ...     _value: int = None
    ...
    ...     def get_value(self):
    ...         return self._value

    >>> obj = A()
    >>> is_extended_by(obj, B)
    False
    >>> obj2 = extend_object(obj, B, value=2)
    >>> obj2 is obj
    True
    >>> is_extended_by(obj, B)
    True
    >>> obj2.get_value()
    2

    An example of using template class with `init_extension()` method:

    >>> class A(object):
    ...     pass

    >>> class B(object):
    ...     _value: int = None
    ...
    ...     def init_extension(self, value: int, power: int = 1):
    ...         self._value = value ** power
    ...
    ...     def get_value(self):
    ...         return self._value

    >>> obj = A()
    >>> is_extended_by(obj, B)
    False
    >>> obj2 = extend_object(obj, B, value=2, power=8)
    >>> obj2 is obj
    True
    >>> is_extended_by(obj, B)
    True
    >>> obj2.get_value()
    256

    Args:
        cls_: The template class.
        obj_: The object.
        trivial_parents_: The parent classes, whose attributes and methods
            are to be ignored when extending the target `obj_`.
        **kwargs: If `cls_` has defined `init_extension` method, then these
            named arguments will be passed to this method in order to initialize
            the extension state.  Otherwise the attributes of `obj_` will be
            set according to these named arguments as follows: for each
            pair of named argument ``(key, value)``, if `obj_` or `cls_` has
            defined attribute ``"_" + key``, then this attribute will be
            assigned the specified value; otherwise the attribute ``key`` will
            be assigned the value.  If no such an attribute is found, then
            an :class:`AttributeError` will be raised.

    Returns:
        The object with extended attributes and methods.
    """
    # get a list of trivial attributes from the parents
    trivial_parents_ = tuple(trivial_parents_)
    if trivial_parents_ not in _extend_object_base_attrs_cache:
        _extend_object_base_attrs_cache[trivial_parents_] = \
            dir(type('dummy', tuple(trivial_parents_), {}))
    base_attrs = _extend_object_base_attrs_cache[trivial_parents_]

    # extend the target object
    for key in dir(cls_):
        # skip magic functions
        if key.startswith('__') and key.endswith('__'):
            continue

        # skip the attributes, methods and properties defined in super-classes
        if key in base_attrs:
            continue
        val = getattr(cls_, key)

        # for methods, bind to the object
        if inspect.isfunction(val):
            arg_spec = inspect.getfullargspec(val)
            if len(arg_spec.args) > 0 and arg_spec.args[0] == 'self':
                # looks like an unbound method, bind it to object
                setattr(obj_, key, types.MethodType(val, obj_))
            else:
                # looks like a static method, bind it to object
                setattr(obj_, key, val)
        elif inspect.ismethod(val):
            raise TypeError(f'Class method is not supported: {key!r}')
        elif isinstance(val, property):
            raise TypeError(f'Property is not supported: {key!r}')
        else:
            # looks like other things, just copy it
            setattr(obj_, key, val)

    # now call the extension constructor, or simply assign the attributes
    if hasattr(obj_, 'init_extension'):
        getattr(obj_, 'init_extension')(**kwargs)
    else:
        def try_set(attr_name, val):
            if not hasattr(cls_, attr_name):
                return False
            if attr_name in base_attrs:
                raise AttributeError(
                    f'Attribute {attr_name!r} is defined in trivial parents of '
                    f'the template class {cls_}, thus cannot be set.'
                )
            setattr(obj_, attr_name, val)
            return True

        for key, val in kwargs.items():
            flag = try_set(f'_{key}', val)
            if not flag:
                try_set(key, val)
            if not flag:
                raise AttributeError(f'Attribute {key!r} is not defined in '
                                     f'the template class {cls_}.')

    # finally, set an additional attribute, to store a reference to the
    # template class
    if not hasattr(obj_, extension_classes_magic_key):
        setattr(obj_, extension_classes_magic_key, [])
    getattr(obj_, extension_classes_magic_key).append(cls_)

    return obj_


def is_extended_by(obj_: Any, cls_: type) -> bool:
    magic_val = getattr(obj_, extension_classes_magic_key, None)
    return bool(magic_val and (cls_ in magic_val))
