import unittest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class ContextualTestCase(unittest.TestCase):

    def test_IgnoreContext(self):
        x = T.random.randn([2, 3, 4])
        context = [T.random.randn([2, 3, 4]),
                   T.random.randn([2, 3, 4])]
        layer = T.jit_compile(tk.layers.IgnoreContext())
        assert_equal(layer(x), x)
        assert_equal(layer(x, context), x)

    def test_AddContext(self):
        x = T.random.randn([2, 3, 4])
        context = [T.random.randn([2, 3, 4]),
                   T.random.randn([2, 3, 4])]
        layer = T.jit_compile(tk.layers.AddContext())
        assert_equal(layer(x), x)
        assert_equal(layer(x, context), x + context[0] + context[1])

    def test_MultiplyContext(self):
        x = T.random.randn([2, 3, 4])
        context = [T.random.randn([2, 3, 4]),
                   T.random.randn([2, 3, 4])]
        layer = T.jit_compile(tk.layers.MultiplyContext())
        assert_equal(layer(x), x)
        assert_equal(layer(x, context), x * context[0] * context[1])
