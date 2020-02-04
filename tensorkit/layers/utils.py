from typing import *

from ..tensor import Module
from .activation import *
from .core import *
from .flow_layer import *

__all__ = [
    'flatten_nested_layers', 'get_activation_class',
]


def flatten_nested_layers(nested_layers: Sequence[
    Union[
        Module,
        Sequence[
            Union[
                Module,
                Sequence[Module]
            ]
        ]
    ]
]) -> List[Module]:
    """
    Flatten a nested list of layers into a list of layers.

    Args:
        nested_layers: Nested list of layers.

    Returns:
        The flatten layer list.
    """
    def do_flatten(target, layer_or_layers):
        if isinstance(layer_or_layers, Module):
            target.append(layer_or_layers)
        elif hasattr(layer_or_layers, '__iter__') and not \
                isinstance(layer_or_layers, (str, bytes, dict)):
            for layer in layer_or_layers:
                do_flatten(target, layer)
        else:
            raise TypeError('`nested_layers` is not a nested list of layers.')

    ret = []
    do_flatten(ret, nested_layers)
    return ret


_activation_classes: Dict[str, Optional[Type[Module]]] = {
    'linear': None,
    'relu': ReLU,
    'leakyrelu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}


def get_activation_class(activation: Optional[str]) -> Optional[Type[Module]]:
    """
    Get the activation module class according to `name`.

    Args:
        activation: The activation name, or None (indicating no activation).

    Returns:
        The module class, or None (indicating no activation).
    """
    if activation is not None:
        canonical_name = activation.lower().replace('_', '')
        if canonical_name not in _activation_classes:
            raise ValueError(f'Unsupported activation: {activation}')
        return _activation_classes[canonical_name]
