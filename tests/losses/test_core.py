import unittest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.tensor import Tensor

from tests.helper import *


class _MySupervisedLoss1(tk.losses.BaseSupervisedLossLayer):

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        return output + target


class _MySupervisedLoss2(tk.losses.BaseSupervisedLossLayer):

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        return (output + target).mean()


class BaseLossesTestCase(unittest.TestCase):

    def test_supervised(self):
        output = T.random.randn([2, 3, 4])
        target = T.random.randn([3, 4])

        l = T.jit_compile(_MySupervisedLoss1())
        assert_allclose(l(output, target), (output + target).mean())
        l = T.jit_compile(_MySupervisedLoss2())
        assert_allclose(l(output, target), (output + target).mean())
