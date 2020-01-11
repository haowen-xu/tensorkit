import unittest

import mock
import numpy as np

from tensorkit import tensor as T
from tensorkit import *


class VariationalChainTestCase(unittest.TestCase):

    def prepare_model(self):
        def p_log_probs(names):
            log_probs = {'c': 3., 'd': 4.}
            return [T.float_scalar(log_probs[n]) for n in names]

        def q_log_probs(names):
            log_probs = {'a': 1., 'b': 2.}
            return [T.float_scalar(log_probs[n]) for n in names]

        p_log_probs = mock.Mock(wraps=p_log_probs)
        q_log_probs = mock.Mock(wraps=q_log_probs)

        class MyNet(object):
            def __init__(self, log_probs, names):
                self._log_probs = log_probs
                self.names = names

            def log_probs(self, names):
                return self._log_probs(list(names))

            def __iter__(self):
                return iter(self.names)

        p = MyNet(p_log_probs, ['c', 'd'])
        q = MyNet(q_log_probs, ['a', 'b'])
        return p_log_probs, p, q_log_probs, q

    def test_default_args(self):
        p_log_probs, p, q_log_probs, q = self.prepare_model()

        chain = VariationalChain(p, q)
        self.assertIs(chain.p, p)
        self.assertIs(chain.q, q)
        self.assertEqual(chain.latent_names, ['a', 'b'])
        self.assertIsNone(chain.latent_axes)
        self.assertIsInstance(chain.vi, VariationalInference)
        np.testing.assert_allclose(T.to_numpy(chain.log_joint), 7.)
        np.testing.assert_allclose(T.to_numpy(chain.latent_log_joint), 3.)
        np.testing.assert_allclose(T.to_numpy(chain.vi.log_joint), 7.)
        np.testing.assert_allclose(T.to_numpy(chain.vi.latent_log_joint), 3.)

        self.assertEqual(q_log_probs.call_args, ((['a', 'b'],),))
        self.assertEqual(p_log_probs.call_args, ((['c', 'd'],),))

    def test_log_joint_arg(self):
        p_log_probs, p, q_log_probs, q = self.prepare_model()

        chain = VariationalChain(p, q, log_joint=T.float_scalar(-1.),
                                 latent_log_joint=T.float_scalar(-2.))
        np.testing.assert_allclose(T.to_numpy(chain.log_joint), -1.)
        np.testing.assert_allclose(T.to_numpy(chain.latent_log_joint), -2.)
        np.testing.assert_allclose(T.to_numpy(chain.vi.log_joint), -1.)
        np.testing.assert_allclose(T.to_numpy(chain.vi.latent_log_joint), -2.)

        self.assertFalse(p_log_probs.called)
        self.assertFalse(q_log_probs.called)

    def test_latent_names_arg(self):
        p_log_probs, p, q_log_probs, q = self.prepare_model()

        chain = VariationalChain(p, q, latent_names=['a'])
        self.assertEqual(chain.latent_names, ['a'])
        np.testing.assert_allclose(T.to_numpy(chain.log_joint), 7.)
        np.testing.assert_allclose(T.to_numpy(chain.vi.log_joint), 7.)
        np.testing.assert_allclose(T.to_numpy(chain.vi.latent_log_joint), 1.)

        self.assertEqual(p_log_probs.call_args, ((['c', 'd'],),))
        self.assertEqual(q_log_probs.call_args, ((['a'],),))

    def test_latent_axis_arg(self):
        p_log_probs, p, q_log_probs, q = self.prepare_model()
        chain = VariationalChain(p, q, latent_axes=[1])
        self.assertEqual(chain.latent_axes, [1])
        self.assertEqual(chain.vi.axes, [1])
