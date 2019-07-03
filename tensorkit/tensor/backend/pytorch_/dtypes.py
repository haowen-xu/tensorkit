import numpy as np
import torch

__all__ = [
    'int', 'int8', 'int16', 'int32', 'int64', 'float', 'float16', 'float32',
    'float64',
]

int = torch.int
int8 = torch.int8
int16 = torch.int16
int32 = torch.int32
int64 = torch.int64
float = torch.float
float16 = torch.float16
float32 = torch.float32
float64 = torch.float64

_DTYPES = {
    'int': torch.int,
    'int8': torch.int8,
    'int16': torch.int16,
    'int32': torch.int32,
    'int64': torch.int64,
    'float': torch.float,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}
_NUMPY_DTYPES = {
    np.int: (
        torch.int
        if np.iinfo(np.int).bits == torch.iinfo(torch.int)
        else (torch.int32 if np.iinfo(np.int).bits == 32 else torch.int64)
    ),
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.int16: torch.int16,
    np.uint8: torch.uint8,
    np.float: torch.float,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
}
