import unittest
from typing import List

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.backend import Tensor
from tests.helper import *


class _MyContextualLayer(tk.layers.BaseContextualLayer):

    def _forward(self, input: Tensor, context: List[Tensor]) -> Tensor:
        output = input
        base = -1.
        for t in context:
            output = output + t * base
            base = base * 10.
        return output


class _MyMultiVariateContextualLayer(tk.layers.BaseMultiVariateContextualLayer):

    def _forward(self, inputs: List[Tensor], context: List[Tensor]) -> List[Tensor]:
        outputs: List[Tensor] = []
        input_base = 1.
        for input in inputs:
            output = input * input_base
            base = -1.
            for t in context:
                output = output + t * base
                base = base * 10.
            outputs.append(output)
            input_base *= 10.
        return outputs


class ContextualTestCase(unittest.TestCase):

    def test_BaseContextualLayer(self):
        x = T.random.randn([2, 3, 4])
        context = [T.random.randn([2, 3, 4]),
                   T.random.randn([2, 3, 4])]
        layer = T.jit_compile(_MyContextualLayer())
        assert_allclose(layer(x), x)
        assert_allclose(layer(x, context), x - context[0] - 10. * context[1])

    def test_BaseMultiVariateContextualLayer(self):
        inputs = [T.random.randn([2, 3, 4]),
                  T.random.randn([2, 3, 4])]
        context = [T.random.randn([2, 3, 4]),
                   T.random.randn([2, 3, 4])]
        layer = T.jit_compile(_MyMultiVariateContextualLayer())

        for k in range(len(inputs)):
            outputs = layer(inputs[:k])
            self.assertEqual(len(outputs), k)
            for j, (input, output) in enumerate(zip(inputs, outputs)):
                assert_allclose(output, input * (10 ** j))

            outputs = layer(inputs[:k], context)
            self.assertEqual(len(outputs), k)
            for j, (input, output) in enumerate(zip(inputs, outputs)):
                assert_allclose(
                    output,
                    input * (10 ** j) - context[0] - 10. * context[1]
                )

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
