import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit.gnn import adj as gnn
from tests.helper import *


def assert_adj_matrix_shape(fn, **kwargs):
    with pytest.raises(Exception, match='`adj_matrix` is required to be a '
                                        'matrix with equal sizes'):
        _ = fn(T.sparse.from_numpy(np.random.randn(2, 3)))
    with pytest.raises(Exception, match='`adj_matrix` is required to be a '
                                        'matrix with equal sizes'):
        _ = fn(T.sparse.from_numpy(np.random.randn(3, 3, 3)))


class AdjMatrixTestCase(TestCase):

    def test_add_self_loop_to_adj(self):
        node_count = 50
        x = make_random_adj_matrix(node_count)
        y = T.sparse.to_numpy(x)
        for w in [0., 1., 2.5]:
            assert_allclose(
                gnn.add_self_loop_to_adj(x, w),
                y + w * np.eye(node_count)
            )
        assert_adj_matrix_shape(gnn.add_self_loop_to_adj)

    def test_transpose_adj(self):
        node_count = 50
        x = make_random_adj_matrix(node_count)
        y = T.sparse.to_numpy(x)
        assert_allclose(gnn.transpose_adj(x), np.transpose(y))
        assert_adj_matrix_shape(gnn.transpose_adj)

    def test_normalize_adj(self):
        node_count = 50
        eps = 1e-6

        # directed
        x = make_random_adj_matrix(node_count)
        y = T.sparse.to_numpy(x)
        d = np.diag(1. / np.maximum(np.sum(y, axis=-1), eps))

        assert_allclose(
            gnn.normalize_adj(x, epsilon=eps),
            np.dot(d, y),
            atol=1e-4, rtol=1e-6
        )

        # undirected
        x = make_random_adj_matrix(node_count, directed=False)
        y = T.sparse.to_numpy(x)
        d = np.diag(1. / np.sqrt(np.maximum(np.sum(y, axis=-1), eps)))

        assert_allclose(
            gnn.normalize_adj(x, undirected=True, epsilon=eps),
            np.dot(np.dot(d, y), d),
            atol=1e-4, rtol=1e-6
        )

    def test_merge_split_adj(self):
        node_counts = [20, 30, 40]
        n = sum(node_counts)
        in_adj = [make_random_adj_matrix(s) for s in node_counts]

        # merge adj
        adj, edge_counts = gnn.merge_adj(in_adj, node_counts)
        expected = np.zeros([n, n])
        m = 0
        for i, a in enumerate(in_adj):
            k = node_counts[i]
            expected[m: (m + k), m: (m + k)] = T.sparse.to_numpy(a)
            m += k
        assert_allclose(adj, expected, rtol=1e-4, atol=1e-6)

        # split adj
        out_adj = gnn.split_adj(adj, node_counts, edge_counts)
        self.assertEqual(len(out_adj), len(in_adj))
        for x, y in zip(out_adj, in_adj):
            assert_allclose(x, y, atol=1e-4, rtol=1e-6)

        # test errors
        with pytest.raises(ValueError,
                           match='`adj` and `node_counts` must not be empty'):
            _ = gnn.merge_adj([], node_counts)

        with pytest.raises(ValueError,
                           match='`adj` and `node_counts` must not be empty'):
            _ = gnn.merge_adj(in_adj, [])

        with pytest.raises(ValueError,
                           match=r'`len\(adj\)` != `len\(node_counts\)`'):
            _ = gnn.merge_adj(in_adj, node_counts[:-1])

        with pytest.raises(ValueError,
                           match=r'`len\(node_counts\)` != `len\(edge_counts\)`'):
            _ = gnn.split_adj(adj, node_counts, edge_counts[:-1])
