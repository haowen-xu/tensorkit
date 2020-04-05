import numpy as np

from tensorkit import tensor as T
from tensorkit.gnn import adj as gnn
from tests.helper import *

try:
    import graph_tool as gt
except ImportError:
    gt = None


if gt is not None:
    class GraphToolUtilsTestCase(TestCase):

        def test_gt_graph_to_adj_matrix(self):
            # build the graph
            node_count = 100
            adj = make_random_adj_matrix(node_count)
            adj_arr = T.sparse.to_numpy(adj)
            row, col = list(T.sparse.get_indices(adj, coord_first=True))
            values = T.sparse.get_values(adj)

            g = gt.Graph()
            weights = g.new_edge_property('float')
            nodes = list(g.add_vertex(node_count))

            for i, j, v in zip(row, col, values):
                edge = g.add_edge(nodes[j], nodes[i])
                weights[edge] = v

            # unweighted
            out_adj = gnn.gt_graph_to_adj_matrix(g)
            assert_allclose(
                out_adj,
                (adj_arr != 0.).astype(np.float32),
                rtol=1e-4, atol=1e-6
            )

            # weighted with arg
            out_adj = gnn.gt_graph_to_adj_matrix(g, edge_weights=weights)
            assert_allclose(out_adj, adj, rtol=1e-4, atol=1e-6)

            # weighted with property map assigned to graph
            g.ep['weights'] = weights
            out_adj = gnn.gt_graph_to_adj_matrix(g)
            assert_allclose(out_adj, adj, rtol=1e-4, atol=1e-6)

            # unweighted with arg
            out_adj = gnn.gt_graph_to_adj_matrix(g, edge_weights=None)
            assert_allclose(
                out_adj,
                (adj_arr != 0.).astype(np.float32),
                rtol=1e-4, atol=1e-6
            )
