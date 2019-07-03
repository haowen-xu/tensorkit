import numpy as np
import torch

{%- set DTYPES = [
    'int', 'int8', 'int16', 'int32', 'int64',
    'float', 'float16', 'float32', 'float64',
]%}

__all__ = [
    {{format_all_list(DTYPES)}},
]

{% for name in DTYPES -%}
{{ name }} = torch.{{ name }}
{% endfor %}
_DTYPES = {
{%- for name in DTYPES %}
    '{{ name }}': torch.{{ name }},
{%- endfor %}
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
