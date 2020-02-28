import numpy as np

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class IdentityFlowTestCase(TestCase):

    def test_IdentityFlow(self):
        x = T.random.randn([2, 3, 4, 5])

        for event_ndims in (0, 1, 2):
            flow = tk.layers.jit_compile(tk.flows.IdentityFlow(event_ndims))
            log_det_shape = T.shape(x)[:4 - event_ndims]
            expected_log_det = T.zeros(log_det_shape)
            flow_standard_check(self, flow, x, x, expected_log_det,
                                T.random.randn(log_det_shape))
