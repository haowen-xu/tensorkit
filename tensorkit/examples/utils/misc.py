from typing import *

import mltk

import tensorkit as tk
from tensorkit import tensor as T

__all__ = [
    'get_weight_parameters', 'get_parameters_and_names',
    'print_experiment_summary',
]


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
