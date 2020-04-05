"""
Utilities to use `graph-tool` package with `tensorkit`.
All utilities in this module are prefixed with `gt_`.
"""

from typing import *

import graph_tool as gt
from graph_tool import spectral
from mltk import NOT_SET

from ... import tensor as T

__all__ = [
    'gt_graph_to_adj_matrix',
]


@T.jit_ignore
def gt_graph_to_adj_matrix(g: gt.Graph,
                           edge_weights: Optional[gt.EdgePropertyMap] = NOT_SET,
                           dtype: str = T.float_x(),
                           device: Optional[str] = None,
                           ) -> T.SparseTensor:
    if edge_weights is NOT_SET:
        if 'weights' in g.ep:
            edge_weights = g.ep['weights']
        else:
            edge_weights = None
    adj_matrix = spectral.adjacency(g, weight=edge_weights)
    return T.sparse.from_spmatrix(
        adj_matrix, dtype=dtype, device=device, force_coalesced=True)
