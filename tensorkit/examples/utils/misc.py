from typing import *

import mltk
import numpy as np

import tensorkit as tk
from tensorkit import tensor as T

__all__ = [
    'get_weights_and_names', 'get_params_and_names',
    'print_experiment_summary', 'print_parameters_summary',
]


def get_weights_and_names(layer: T.Module) -> Tuple[List[T.Variable], List[str]]:
    params = []
    names = []
    for name, param in tk.layers.iter_named_parameters(layer):
        if not name.endswith('.bias_store.value') and not name.endswith('.bias'):
            params.append(param)
            names.append(name)
    return params, names


def get_params_and_names(layer: T.Module
                         ) -> Tuple[List[T.Variable], List[str]]:
    params = []
    names = []
    for name, param in tk.layers.iter_named_parameters(layer):
        params.append(param)
        names.append(name)
    return params, names


def print_experiment_summary(exp: mltk.Experiment,
                             train_stream: mltk.DataStream,
                             val_stream: Optional[mltk.DataStream] = None,
                             test_stream: Optional[mltk.DataStream] = None):
    # the config
    mltk.print_config(exp.config)
    print('')

    # the dataset info
    data_info = []
    for name, stream in [('Train', train_stream), ('Validation', val_stream),
                         ('Test', test_stream)]:
        if stream is not None:
            data_info.append((name, len(stream)))
    if data_info:
        print(mltk.format_key_values(data_info, 'Number of Data'))
        print('')

    # the device info
    device_info = [
        ('Current', T.current_device())
    ]
    gpu_devices = T.gpu_device_list()
    if gpu_devices:
        device_info.append(('Available', gpu_devices))
    print(mltk.format_key_values(device_info, 'Device Info'))
    print('')


def print_parameters_summary(params: List[T.Variable], names: List[str]):
    shapes = []
    sizes = []
    total_size = 0
    max_shape_len = 0
    max_size_len = 0
    right_pad = ' ' * 3

    for param in params:
        shape = T.shape(param)
        size = np.prod(shape)
        total_size += size
        shapes.append(str(shape))
        sizes.append(f'{size:,d}')
        max_shape_len = max(max_shape_len, len(shapes[-1]))
        max_size_len = max(max_size_len, len(sizes[-1]))

    total_size = f'{total_size:,d}'
    right_len = max(max_shape_len + len(right_pad) + max_size_len, len(total_size))

    param_info = []
    max_name_len = 0
    for param, name, shape, size in zip(params, names, shapes, sizes):
        max_name_len = max(max_name_len, len(name))
        right = f'{shape:<{max_shape_len}s}{right_pad}{size:>{max_size_len}s}'
        right = f'{right:>{right_len}s}'
        param_info.append((name, right))

    if param_info:
        param_info.append(('Total', f'{total_size:>{right_len}s}'))
        lines = mltk.format_key_values(
            param_info, title='Parameters', formatter=str).strip().split('\n')
        k = len(lines[-1])
        lines.insert(-1, '-' * k)

        print('\n'.join(lines))
