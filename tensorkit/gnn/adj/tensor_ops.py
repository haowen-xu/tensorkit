from typing import *

from ... import tensor as T

__all__ = [
    'add_self_loop_to_adj', 'transpose_adj', 'normalize_adj',
    'normalize_partitioned_adj', 'merge_adj', 'split_adj',
]


# ---- adjacency matrix ops ----
def _check_adj_matrix(adj_matrix: T.SparseTensor,
                      size: Optional[int] = None
                      ) -> int:
    s = T.sparse.shape(adj_matrix)
    if len(s) != 2 or s[0] != s[1] or (size is not None and s[0] != size):
        raise ValueError(
            '`adj_matrix` is required to be a matrix with equal sizes '
            'in its dimensions: got shape {}.'.format(s))
    return s[0]


@T.jit_ignore
def add_self_loop_to_adj(adj_matrix: T.SparseTensor,
                         self_loop_weight: float = 0.) -> T.SparseTensor:
    _check_adj_matrix(adj_matrix)
    adj_matrix = T.sparse.coalesce(adj_matrix)
    return adj_matrix + self_loop_weight * T.sparse.eye(
        adj_matrix.shape[0],
        dtype=T.sparse.get_dtype(adj_matrix),
        device=T.sparse.get_device(adj_matrix)
    )


@T.jit_ignore
def transpose_adj(adj_matrix: T.SparseTensor) -> T.SparseTensor:
    _check_adj_matrix(adj_matrix)
    adj_matrix = T.sparse.coalesce(adj_matrix)
    indices = adj_matrix.indices()
    values = adj_matrix.values()
    indices = T.stack([indices[1], indices[0]], axis=0)
    return T.sparse.make_sparse(indices, values, T.sparse.shape(adj_matrix))


def _row_div(m: T.SparseTensor, v: T.Tensor) -> T.SparseTensor:
    indices = m.indices()
    v = m.values() / T.index_select(v, indices[0], axis=0)
    return T.sparse.make_sparse(indices, v, T.sparse.shape(m))


def _col_div(m: T.SparseTensor, v: T.Tensor) -> T.SparseTensor:
    indices = m.indices()
    v = m.values() / T.index_select(v, indices[1], axis=0)
    return T.sparse.make_sparse(indices, v, T.sparse.shape(m))


def _compute_degree(adj_matrix: T.SparseTensor) -> T.Tensor:
    if T.sparse.value_count(adj_matrix) == 0:
        size = T.sparse.length(adj_matrix)
        degree = T.zeros([size], dtype=T.sparse.get_dtype(adj_matrix),
                         device=T.sparse.get_device(adj_matrix))
    else:
        degree = T.sparse.to_dense(T.sparse.reduce_sum(adj_matrix, axis=-1))
    return degree


@T.jit_ignore
def normalize_adj(adj_matrix: T.SparseTensor,
                  undirected: bool = False,
                  epsilon: float = T.EPSILON) -> T.SparseTensor:
    _check_adj_matrix(adj_matrix)
    adj_matrix = T.sparse.coalesce(adj_matrix)

    # normalize the adj matrix
    # TODO: do we need to check whether or not all degrees are non-negative?
    degree = _compute_degree(adj_matrix)
    if epsilon != 0.:
        degree = T.clip_left(degree, epsilon)

    if undirected:
        degree_sqrt = T.sqrt(degree)
        adj_matrix = _row_div(adj_matrix, degree_sqrt)
        adj_matrix = _col_div(adj_matrix, degree_sqrt)
    else:
        adj_matrix = _row_div(adj_matrix, degree)

    return adj_matrix


@T.jit_ignore
def normalize_partitioned_adj(adj_matrices: List[T.SparseTensor],
                              undirected: bool = False,
                              epsilon: float = T.EPSILON
                              ) -> List[T.SparseTensor]:
    # check the size
    if not adj_matrices:
        raise ValueError(f'`adj_matrices` must not be empty.')
    size = _check_adj_matrix(adj_matrices[0])
    for adj_matrix in adj_matrices[1:]:
        _check_adj_matrix(adj_matrix, size)

    # do coalesce
    adj_matrices = [T.sparse.coalesce(adj) for adj in adj_matrices]

    # normalize the adj matrices
    degree = T.add_n([
        _compute_degree(adj)
        for adj in adj_matrices
    ])
    if epsilon != 0.:
        degree = T.clip_left(degree, epsilon)

    if undirected:
        degree_sqrt = T.sqrt(degree)
        adj_matrices = [_row_div(a, degree_sqrt) for a in adj_matrices]
        adj_matrices = [_col_div(a, degree_sqrt) for a in adj_matrices]
    else:
        adj_matrices = [_row_div(a, degree) for a in adj_matrices]

    return adj_matrices


@T.jit_ignore
def merge_adj(adj: List[T.SparseTensor],
              node_counts: List[int]
              ) -> Tuple[T.SparseTensor, List[int]]:
    """
    Merge the adjacency matrices of multiple graphs into a single, large graph.

    Args:
        adj: The adjacency matrices of the component graphs.
        node_counts: The node count of each component graph.

    Returns:
        The output adjacency matrix, as well the edge count of each
        component graph.
    """
    if not adj or not node_counts:
        raise ValueError(f'`adj` and `node_counts` must not be empty.')
    if len(adj) != len(node_counts):
        raise ValueError(f'`len(adj)` != `len(node_counts)`: '
                         f'{len(adj)} vs {len(node_counts)}.')

    n = 0
    indices = []
    values = []
    edge_counts = []
    edge_axis = 1 if T.sparse.SPARSE_INDICES_DEFAULT_IS_COORD_FIRST else 0

    for i, a in enumerate(adj):
        a = T.sparse.coalesce(a)

        # extract edge values
        values.append(T.sparse.get_values(a))

        # extract edge indices
        idx = T.sparse.get_indices(a) + n
        indices.append(idx)
        edge_counts.append(T.shape(idx)[edge_axis])
        n += node_counts[i]

    indices = T.concat(indices, axis=edge_axis)
    values = T.concat(values, axis=0)
    out_adj = T.sparse.make_sparse(
        indices, values, shape=[n, n], force_coalesced=True)

    return out_adj, edge_counts


@T.jit_ignore
def split_adj(adj: T.SparseTensor,
              node_counts: List[int],
              edge_counts: List[int]
              ) -> List[T.SparseTensor]:
    """
    The inverse operation of :func:`merge_adj`.

    Args:
        adj: The adjacency matrix of the whole graph.
        node_counts: The node counts of each component graph.
        edge_counts: The edge counts of each component graph.

    Returns:
        The adjacency matrices of each component graph.
    """
    if len(node_counts) != len(edge_counts):
        raise ValueError(f'`len(node_counts)` != `len(edge_counts)`: '
                         f'{len(node_counts)} vs {len(edge_counts)}.')

    out_adj = []
    adj = T.sparse.coalesce(adj)
    edge_axis = 1 if T.sparse.SPARSE_INDICES_DEFAULT_IS_COORD_FIRST else 0
    indices = T.split(T.sparse.get_indices(adj), edge_counts, axis=edge_axis)
    values = T.split(T.sparse.get_values(adj), edge_counts, axis=0)

    n = 0
    for idx, val, k in zip(indices, values, node_counts):
        out_adj.append(T.sparse.make_sparse(
            idx - n, val, shape=[k, k], force_coalesced=True))
        n += k
    return out_adj
