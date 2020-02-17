import unittest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class BranchTestCase(unittest.TestCase):

    def test_branch(self):
        shared = tk.layers.Linear(5, 5)
        branches = [tk.layers.Linear(5, 4),
                    tk.layers.Linear(5, 3),
                    tk.layers.Linear(5, 2)]

        x = T.random.randn([7, 5])
        for k in range(len(branches) + 1):
            # without shared module
            layer = tk.layers.Branch(branches[:k])
            layer = tk.layers.jit_compile(layer)

            out = layer(x)
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), k)

            for i in range(k):
                assert_allclose(out[i], branches[i](x))

            # with shared module
            layer = tk.layers.Branch(branches[:k], shared=shared)
            layer = tk.layers.jit_compile(layer)

            out = layer(x)
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), k)

            for i in range(k):
                assert_allclose(out[i], branches[i](shared(x)))
