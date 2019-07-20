import numpy as np
import torch

__all__ = [
    'int8', 'uint8', 'int16', 'int32', 'int64', 'float16', 'float32',
    'float64',
]

int8 = torch.int8
uint8 = torch.uint8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64

_DTYPES = {
    'int8': torch.int8,
    'uint8': torch.uint8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}

_dtype_info_fn = {
    True: (torch.iinfo, np.iinfo),
    False: (torch.finfo, np.finfo),
}


def _dtype(np_dtype, *candidates, is_int=False):
    torch_info_fn, np_info_fn = _dtype_info_fn[is_int]
    np_info = np_info_fn(np_dtype)

    for c in candidates:
        torch_info = torch_info_fn(c)
        if torch_info.bits == np_info.bits:
            return c

    raise TypeError(f'Cannot find a corresponding backend dtype for {np_dtype}')


_NUMPY_DTYPES = {
    int: _dtype(int, torch.int, torch.int32, torch.int64, is_int=True),
    np.int: _dtype(np.int, torch.int, torch.int32, torch.int64, is_int=True),
    np.int8: torch.int8,
    np.uint8: torch.uint8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    float: _dtype(np.float, torch.float, torch.float32, torch.float64),
    np.float: _dtype(np.float, torch.float, torch.float32, torch.float64),
    np.double: _dtype(np.double, torch.double, torch.float64),
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}
