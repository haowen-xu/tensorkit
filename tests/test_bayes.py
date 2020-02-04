import unittest

import mock
import numpy as np
import pytest

from tensorkit import tensor as T
from tensorkit import *
from tensorkit.distributions import *
from tests.helper import *


class BayesianNetTestCase(unittest.TestCase):

    def test_construct(self):
        # no observation
        net = BayesianNet()
        self.assertEqual(len(net), 0)
        self.assertEqual(list(net), [])
        self.assertEqual(dict(net.observed), {})
        self.assertEqual(net._original_observed, {})
        self.assertEqual(net._stochastic_tensors, {})
        with pytest.raises(Exception):
            # `net.observed` should be read-only
            net.observed['x'] = T.zeros([])

        # with observation
        normal = UnitNormal([2, 3, 4])
        x = T.as_tensor(np.random.randn(3, 4))
        y = normal.sample()

        net = BayesianNet({'x': x, 'y': y})
        self.assertEqual(len(net), 0)
        self.assertEqual(list(net), [])
        self.assertEqual(list(net.observed), ['x', 'y'])
        self.assertIs(net.observed['x'], x)
        self.assertIs(net.observed['y'], y.tensor)
        self.assertIs(net._original_observed['x'], x)
        self.assertIs(net._original_observed['y'], y)

    def test_add(self):
        x_observed = T.as_tensor(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))
        net = BayesianNet({'x': x_observed})
        d = UnitNormal([3, 4])
        self.assertNotIn('x', net)
        self.assertNotIn('y', net)

        # add an observed node
        x = net.add('x', d, n_samples=2, group_ndims=1)
        self.assertIs(net.get('x'), x)
        self.assertIs(net['x'], x)
        self.assertIn('x', net)
        self.assertListEqual(list(net), ['x'])

        self.assertIsInstance(x, StochasticTensor)
        self.assertIs(x.distribution, d)
        self.assertEqual(x.n_samples, 2)
        self.assertEqual(x.group_ndims, 1)
        self.assertEqual(x.reparameterized, True)
        self.assertIs(x.tensor, x_observed)
        self.assertEqual(T.shape(x.tensor), [2, 3, 4])

        # add an unobserved node
        y = net.add('y', d, group_ndims=1, reparameterized=False)
        self.assertIs(net.get('y'), y)
        self.assertIs(net['y'], y)
        self.assertIn('y', net)
        self.assertListEqual(list(net), ['x', 'y'])

        self.assertIsInstance(y, StochasticTensor)
        self.assertIs(y.distribution, d)
        self.assertEqual(y.n_samples, None)
        self.assertEqual(y.group_ndims, 1)
        self.assertEqual(y.reparameterized, False)
        self.assertEqual(T.shape(y.tensor), [3, 4])

        # error adding the same variable
        with pytest.raises(
                ValueError,
                match="Stochastic tensor 'x' already exists."):
            _ = net.add('x', d)

    def test_add_reparameterized_arg(self):
        normal = UnitNormal(shape=[2, 3])

        # test reparameterized: False
        with mock.patch('tensorkit.tensor.stop_grad',
                        mock.Mock(wraps=T.stop_grad)) as m:
            # TODO: switch to some other namespace when refractored
            x = normal.sample(5, reparameterized=True)
            self.assertTrue(x.reparameterized)
            net = BayesianNet({'x': x.tensor})
            t = net.add('x', normal, n_samples=5, reparameterized=False)
            self.assertFalse(t.reparameterized)
        self.assertTrue(m.call_count, 1)
        self.assertIs(m.call_args[0][0], x.tensor)

        # test inherit reparameterized from `x`
        x = normal.sample(5, reparameterized=True)
        self.assertTrue(x.reparameterized)
        net = BayesianNet({'x': x})
        t = net.add('x', normal, n_samples=5)
        self.assertEqual(t.reparameterized, x.reparameterized)
        assert_allclose(x.tensor, t.tensor)

        x = normal.sample(5, reparameterized=False)
        self.assertFalse(x.reparameterized)
        net = BayesianNet({'x': x})
        t = net.add('x', normal, n_samples=5)
        self.assertEqual(t.reparameterized, x.reparameterized)
        assert_allclose(x.tensor, t.tensor)

        # test override reparameterized: True -> False
        with mock.patch('tensorkit.tensor.stop_grad',
                        mock.Mock(wraps=T.stop_grad)) as m:
            x = normal.sample(5, reparameterized=True)
            self.assertTrue(x.reparameterized)
            net = BayesianNet({'x': x})
            t = net.add('x', normal, n_samples=5, reparameterized=False)
            self.assertFalse(t.reparameterized)
        self.assertTrue(m.call_count, 1)
        self.assertIs(m.call_args[0][0], x.tensor)

        # test cannot override reparameterized: False -> True
        x = normal.sample(5, reparameterized=False)
        self.assertFalse(x.reparameterized)
        net = BayesianNet({'x': x})
        with pytest.raises(ValueError,
                           match="`reparameterized` is True, but the "
                                 "observation for stochastic tensor 'x' is "
                                 "not re-parameterized"):
            _ = net.add('x', normal, n_samples=5, reparameterized=True)

    def test_outputs(self):
        x_observed = T.as_tensor(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))
        net = BayesianNet({'x': x_observed})
        normal = UnitNormal([3, 4])
        x = net.add('x', normal)
        y = net.add('y', normal)

        # test single query
        x_out = net.output('x')
        self.assertIs(x_out, x.tensor)
        self.assertIsInstance(x_out, T.Tensor)
        assert_equal(x_out, x_observed)

        # test multiple query
        x_out, y_out = net.outputs(iter(['x', 'y']))
        self.assertIs(x_out, x.tensor)
        self.assertIs(y_out, y.tensor)
        self.assertIsInstance(x_out, T.Tensor)
        self.assertIsInstance(y_out, T.Tensor)
        assert_equal(x_out, x_observed)

    def test_log_prob(self):
        x_observed = T.as_tensor(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))
        net = BayesianNet({'x': x_observed})
        normal = UnitNormal([3, 4])
        x = net.add('x', normal)
        y = net.add('y', normal)

        # test single query
        x_log_prob = net.log_prob('x')
        self.assertIsInstance(x_log_prob, T.Tensor)
        assert_allclose(x_log_prob, normal.log_prob(x_observed))

        # test multiple query
        x_log_prob, y_log_prob = net.log_probs(iter(['x', 'y']))
        self.assertIsInstance(x_log_prob, T.Tensor)
        self.assertIsInstance(y_log_prob, T.Tensor)
        assert_allclose(x_log_prob, normal.log_prob(x_observed))
        assert_allclose(x_log_prob, normal.log_prob(x.tensor))
        assert_allclose(y_log_prob, normal.log_prob(y.tensor))

    def test_query_pair(self):
        x_observed = T.as_tensor(
            np.arange(24, dtype=np.float32).reshape([2, 3, 4]))
        net = BayesianNet({'x': x_observed})
        normal = UnitNormal([3, 4])
        x = net.add('x', normal)
        y = net.add('y', normal)

        # test single query
        x_out, x_log_prob = net.query_pair('x')
        self.assertIsInstance(x_out, T.Tensor)
        self.assertIsInstance(x_log_prob, T.Tensor)
        self.assertIs(x_out, x.tensor)
        assert_allclose(x_log_prob, normal.log_prob(x_observed))

        # test multiple query
        [(x_out, x_log_prob), (y_out, y_log_prob)] = \
            net.query_pairs(iter(['x', 'y']))
        for o in [x_out, x_log_prob, y_out, y_log_prob]:
            self.assertIsInstance(o, T.Tensor)
        self.assertIs(x_out, x.tensor)
        self.assertIs(y_out, y.tensor)
        assert_allclose(x_log_prob, normal.log_prob(x_observed))
        assert_allclose(x_log_prob, normal.log_prob(x.tensor))
        assert_allclose(y_log_prob, normal.log_prob(y.tensor))

    def test_chain(self):
        q_net = BayesianNet({'x': T.ones([1])})
        q_net.add('z', Normal(q_net.observed['x'], T.float_scalar(1.)))
        q_net.add('y', Normal(q_net.observed['x'] * 2, T.float_scalar(2.)))

        def net_builder(observed):
            net = BayesianNet(observed)
            z = net.add('z', UnitNormal([1]))
            y = net.add('y', Normal(T.zeros([1]), T.full([1], 2.)))
            x = net.add('x', Normal(z.tensor + y.tensor, T.ones([1])))
            return net

        net_builder = mock.Mock(wraps=net_builder)

        # test chain with default parameters
        chain = q_net.chain(net_builder)
        self.assertEqual(
            net_builder.call_args,
            (({'y': q_net['y'], 'z': q_net['z']},),)
        )
        self.assertEqual(chain.latent_names, ['z', 'y'])
        self.assertIsNone(chain.latent_axis)

        # test chain with latent_names
        chain = q_net.chain(net_builder, latent_names=['y'])
        self.assertEqual(
            net_builder.call_args,
            (({'y': q_net['y']},),)
        )
        self.assertEqual(chain.latent_names, ['y'])

        # test chain with latent_axis
        chain = q_net.chain(net_builder, latent_axis=-1)
        self.assertEqual(chain.latent_axis, [-1])

        chain = q_net.chain(net_builder, latent_axis=[-1, 2])
        self.assertEqual(chain.latent_axis, [-1, 2])

        # test chain with observed
        chain = q_net.chain(net_builder, observed=q_net.observed)
        self.assertEqual(
            net_builder.call_args,
            (({'x': q_net.observed['x'], 'y': q_net['y'], 'z': q_net['z']},),)
        )
        self.assertEqual(chain.latent_names, ['z', 'y'])
