from itertools import product

import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit.arg_check import *
from tests import ops
from tests.helper import *
from tests.ops import *


class TensorLossesTestCase(TestCase):

    def test_negative_sampling(self):
        def sigmoid(x):
            return np.where(
                x >= 0.,
                1. / (1. + np.exp(-x)),
                np.exp(x) / (1. + np.exp(x)),
            )

        def log_sigmoid(x):
            return np.log(sigmoid(x))

        def f(pos_logits, neg_logits, reduction='none', negative=False):
            o = log_sigmoid(pos_logits) + np.sum(log_sigmoid(-neg_logits), axis=-1)
            if reduction == 'sum':
                o = np.sum(o)
            elif reduction == 'mean':
                o = np.mean(o)
            if negative:
                o = -o
            return o

        for k in [1, 5]:
            for reduction, negative in product(
                        (None, 'none', 'sum', 'mean'),
                        (None, True, False),
                    ):
                pos_logits = T.random.randn([10])
                neg_logits = T.random.randn([10, k])
                kwargs = {'negative': negative} if negative is not None else {}
                if reduction is not None:
                    kwargs['reduction'] = reduction
                x = T.losses.negative_sampling(pos_logits, neg_logits, **kwargs)
                y = f(T.to_numpy(pos_logits), T.to_numpy(neg_logits), **kwargs)
                assert_allclose(x, y, atol=1e-4, rtol=1e-6)

                # test errors
                with pytest.raises(Exception, match='`pos_logits` must be 1d, '
                                                    '`neg_logits` must be 2d'):
                    _ = T.losses.negative_sampling(neg_logits, neg_logits)

                with pytest.raises(Exception, match='`pos_logits` must be 1d, '
                                                    '`neg_logits` must be 2d'):
                    _ = T.losses.negative_sampling(pos_logits, pos_logits)

                with pytest.raises(Exception, match='`pos_logits` must be 1d, '
                                                    '`neg_logits` must be 2d'):
                    _ = T.losses.negative_sampling(pos_logits[:-1], neg_logits)
