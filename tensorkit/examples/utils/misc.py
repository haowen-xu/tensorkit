from typing import *

import tensorkit as tk
from tensorkit import tensor as T

__all__ = ['get_weight_parameters', 'get_parameters_and_names']


def get_weight_parameters(layer: T.Module) -> List[T.Variable]:
    params = []
    for name, param in tk.layers.iter_named_parameters(layer):
        if not name.endswith('.bias_store.value') and not name.endswith('.bias'):
            params.append(param)
    return params


def get_parameters_and_names(layer: T.Module
                             ) -> Tuple[List[T.Variable], List[str]]:
    params = []
    names = []
    for name, param in tk.layers.iter_named_parameters(layer):
        params.append(param)
        names.append(name)
    return params, names
