import unittest

import numpy as np

from tensorkit import tensor as T
from tests.helper import assert_allclose


class LinalgTestCase(unittest.TestCase):

    def test_qr(self):
        np.random.seed(1234)
        for k in [1, 5]:
            m = np.random.randn(k, k)
            q, r = T.linalg.qr(T.as_tensor(m))
            expected_q, expected_r = np.linalg.qr(m)
            assert_allclose(q, expected_q)
            assert_allclose(r, expected_r)

    def test_slogdet(self):
        np.random.seed(1234)
        for k in [1, 5]:
            m = np.random.randn(k, k)
            sign, logdet = T.linalg.slogdet(T.as_tensor(m))
            expected_sign, expected_logdet = np.linalg.slogdet(m)
            assert_allclose(sign, expected_sign)
            assert_allclose(logdet, expected_logdet)
