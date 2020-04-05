from functools import partial

import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.arg_check import *
from tensorkit.gnn import adj as gnn
from tensorkit.layers import (jit_compile, SimpleParamStore, Dense, Conv1d,
                              Conv2d, Conv3d, Linear)
from tests.helper import *
from tests.ops import *

assert_allclose_ = partial(assert_allclose, atol=1e-4, rtol=1e-6)


class GCNIdentityTestCase(TestCase):

    def test_identity(self):
        x = T.random.randn([50, 3])
        adj = make_random_adj_matrix(50)
        layer = jit_compile(gnn.GCNIdentity())
        assert_allclose_(layer(x), x)
        assert_allclose_(layer(x, adj), x)

    def test_sequential(self):
        layers = [
            gnn.GCNDense(3, 4, activation=tk.layers.LeakyReLU),
            gnn.GCNDense(4, 5, activation=tk.layers.LeakyReLU),
        ]
        adj = make_random_adj_matrix(50)
        x = T.random.randn([50, 7, 3])

        for i in range(1, len(layers) + 1):
            m = jit_compile(gnn.GCNSequential(layers[:i]))
            out = m(x, adj)
            expected = x
            for layer in layers[:i]:
                expected = layer(expected, adj)
            assert_allclose_(out, expected)

    def test_partitioned_sequential(self):
        def make_partitioned_gcn(in_featuers, out_features):
            return gnn.PartitionedGCNLayer(
                modules=[Linear(in_featuers, out_features),
                         Linear(in_featuers, out_features)],
                self_module=Linear(in_featuers, out_features),
                bias_store=SimpleParamStore([out_features], tk.init.uniform),
                activation=tk.layers.LeakyReLU,
            )

        layers = [
            make_partitioned_gcn(3, 4),
            make_partitioned_gcn(4, 5),
        ]
        adjs = [make_random_adj_matrix(50) for _ in range(2)]
        x = T.random.randn([50, 7, 3])

        for i in range(1, len(layers) + 1):
            m = jit_compile(gnn.PartitionedGCNSequential(layers[:i]))
            out = m(x, adjs)
            expected = x
            for layer in layers[:i]:
                expected = layer(expected, adjs)
            assert_allclose_(out, expected)


class PartitionedGCNLayerTestCase(TestCase):

    def test_partitioned_gcn(self):
        def h(cls, value_ndims, feature_axis, m_factory, n_partitions):
            in_shape = make_conv_shape([], 3, [6, 7, 8][:value_ndims - 1])
            inputs = [
                T.random.randn(in_shape)
                for in_shape in [[50] + in_shape, [50, 7] + in_shape]]
            all_modules = [
                jit_compile(m_factory(3, 4, activation=tk.layers.Tanh))
                for _ in range(n_partitions + 1)]
            in_adj = [make_random_adj_matrix(50) for _ in all_modules]

            # compute the out shapes for add merge mode
            add_out_shape = list(in_shape)
            add_out_shape[feature_axis] = 4

            def f(modules, self_module, self_weight, use_bias, normalizer,
                  activation, merge_mode):
                if n_partitions == 1:
                    adj = in_adj[0]
                else:
                    adj = in_adj[:len(modules)]

                out_shape = list(add_out_shape)
                if merge_mode == 'concat':
                    out_shape[feature_axis] *= len(modules) + int(self_module is not None)

                bias_store = (
                    SimpleParamStore(out_shape, initializer=tk.init.normal)
                    if use_bias else None
                )
                layer_kwargs = dict(self_module=self_module,
                                    self_weight=self_weight,
                                    bias_store=bias_store,
                                    normalizer=normalizer,
                                    activation=activation,
                                    merge_mode=merge_mode)

                layer = jit_compile(
                    cls(module=modules[0], **layer_kwargs) if n_partitions == 1
                    else cls(modules=modules, **layer_kwargs)
                )
                if isinstance(activation, type):
                    activation_layer = activation()
                else:
                    activation_layer = activation

                for x in inputs:
                    # test errors
                    if len(modules) > 1:
                        with pytest.raises(
                                Exception,
                                match=r'`adj` is expected to have .* element'
                                      r'\(s\), but got .*'):
                            _ = layer(x, in_adj[: len(modules) - 1])

                    if T.rank(x) == value_ndims + 1:
                        with pytest.raises(
                                Exception,
                                match='`input` is expected to be at least .*d'):
                            _ = layer(x[0], adj)

                    # obtain valid output
                    y = layer(x, adj)
                    self.assertEqual(
                        T.shape(y),
                        T.shape(x)[:-value_ndims] + out_shape
                    )

                    # compute the expected output
                    def g(m, x):
                        m_out, m_front = T.flatten_to_ndims(x, value_ndims + 1)
                        m_out = m(m_out)
                        m_out = T.unflatten_from_ndims(m_out, m_front)
                        return m_out

                    outs = []
                    for m, a in zip(modules, in_adj):
                        m_out = T.as_tensor(
                            np.reshape(
                                np.dot(T.sparse.to_numpy(a),
                                       T.to_numpy(x).reshape([50, -1])),
                                x.shape
                            )
                        )
                        outs.append(g(m, m_out))

                    if self_module is not None:
                        outs.append(g(self_module, x))

                    if merge_mode == 'add':
                        out = T.add_n(outs)
                    elif merge_mode == 'concat':
                        out = T.concat(outs, axis=feature_axis)

                    if bias_store is not None:
                        out = out + bias_store()
                    if normalizer is not None:
                        out = normalizer(out)
                    if activation is not None:
                        out = activation_layer(out)

                    # assert the output is expected
                    assert_allclose_(y, out)

            # test modules
            normalizer_ = tk.layers.Sigmoid()
            activation_ = tk.layers.LeakyReLU()
            for n_modules in range(1, len(all_modules)):
                f(all_modules[:n_modules], all_modules[-1], 1.0, True,
                  normalizer_, tk.layers.LeakyReLU, merge_mode='add')
                f(all_modules[:n_modules], all_modules[-1], 1.5, True,
                  normalizer_, tk.layers.LeakyReLU, merge_mode='concat')
                f(all_modules[:n_modules], None, 2.0, True,
                  normalizer_, activation_, merge_mode='add')
                f(all_modules[:n_modules], None, 2.5, True,
                  normalizer_, activation_, merge_mode='concat')

            # test no bias_store, no normalizer, no activation
            f(all_modules[:-1], all_modules[-1], 1.0, False, None, None, merge_mode='add')
            f(all_modules[:-1], all_modules[-1], 1.0, False, None, None, merge_mode='concat')

        fa = lambda k: -1 if T.IS_CHANNEL_LAST else -(k + 1)
        mf = lambda cls: partial(cls, kernel_size=1)

        h(gnn.PartitionedGCNLayer, value_ndims=1, feature_axis=-1,
          n_partitions=3, m_factory=Dense)
        h(gnn.PartitionedGCNLayer1d, value_ndims=2, feature_axis=fa(1),
          n_partitions=3, m_factory=mf(Conv1d))
        h(gnn.PartitionedGCNLayer2d, value_ndims=3, feature_axis=fa(2),
          n_partitions=3, m_factory=mf(Conv2d))
        h(gnn.PartitionedGCNLayer3d, value_ndims=4, feature_axis=fa(3),
          n_partitions=3, m_factory=mf(Conv3d))

        h(gnn.GCNLayer, value_ndims=1, feature_axis=-1, n_partitions=1, m_factory=Dense)
        h(gnn.GCNLayer1d, value_ndims=2, feature_axis=fa(1), n_partitions=1, m_factory=mf(Conv1d))
        h(gnn.GCNLayer2d, value_ndims=3, feature_axis=fa(2), n_partitions=1, m_factory=mf(Conv2d))
        h(gnn.GCNLayer3d, value_ndims=4, feature_axis=fa(3), n_partitions=1, m_factory=mf(Conv3d))

        # test errors
        with pytest.raises(ValueError,
                           match='`feature_matrix_ndims` must be at least 2'):
            _ = gnn.PartitionedGCNLayer(
                modules=[Linear(3, 4)], feature_matrix_ndims=1)

        with pytest.raises(ValueError,
                           match='`feature_axis` out of range'):
            _ = gnn.PartitionedGCNLayer(
                modules=[Linear(3, 4)], feature_axis=-2)
        with pytest.raises(ValueError,
                           match='`feature_axis` out of range'):
            _ = gnn.PartitionedGCNLayer(
                modules=[Linear(3, 4)], feature_axis=0)

        with pytest.raises(ValueError,
                           match='`modules` is required not to be empty'):
            _ = gnn.PartitionedGCNLayer(modules=[])

    def test_self_loop(self):
        def h(cls, value_ndims, feature_axis, m_factory):
            in_shape = make_conv_shape([], 3, [6, 7, 8][:value_ndims - 1])
            inputs = [
                T.random.randn(in_shape)
                for in_shape in [[50] + in_shape, [50, 7] + in_shape]]

            out_shape = list(in_shape)
            out_shape[feature_axis] = 4
            module = jit_compile(m_factory(3, 4, activation=tk.layers.Tanh))
            layer = jit_compile(cls(module))

            for adj in [None, make_random_adj_matrix(50)]:
                for x in inputs:
                    y = layer(x, adj)
                    self.assertEqual(
                        T.shape(y),
                        T.shape(x)[:-value_ndims] + out_shape
                    )

                    # compute the expected output
                    expected, m_front = T.flatten_to_ndims(x, value_ndims + 1)
                    expected = module(expected)
                    expected = T.unflatten_from_ndims(expected, m_front)

                    # assert the output is expected
                    assert_allclose_(y, expected)

        fa = lambda k: -1 if T.IS_CHANNEL_LAST else -(k + 1)
        mf = lambda cls: partial(cls, kernel_size=1)
        h(gnn.GCNSelfLoop, value_ndims=1, feature_axis=-1, m_factory=Dense)
        h(gnn.GCNSelfLoop1d, value_ndims=2, feature_axis=fa(1), m_factory=mf(Conv1d))
        h(gnn.GCNSelfLoop2d, value_ndims=3, feature_axis=fa(2), m_factory=mf(Conv2d))
        h(gnn.GCNSelfLoop3d, value_ndims=4, feature_axis=fa(3), m_factory=mf(Conv3d))

        # test errors
        with pytest.raises(ValueError,
                           match='`feature_matrix_ndims` must be at least 2'):
            _ = gnn.GCNSelfLoop(module=Linear(3, 4), feature_matrix_ndims=1)

    def test_gcn_dense(self):
        def h(x, adj, in_features, out_features, use_self_loop, self_weight,
              merge_mode, use_bias, normalizer, activation, bias_init,
              weight_norm):
            T.random.seed(1234)
            m = gnn.GCNDense(
                in_features=in_features, out_features=out_features,
                use_self_loop=use_self_loop, self_weight=self_weight,
                merge_mode=merge_mode,
                normalizer=normalizer, activation=activation,
                weight_norm=weight_norm, bias_init=bias_init,
                **({'use_bias': use_bias} if use_bias is not None else {})
            )
            m(x, adj)
            tk.layers.set_eval_mode(m)
            return jit_compile(m)(x, adj)

        def g(x, adj, in_features, out_features, use_self_loop, self_weight,
              merge_mode, use_bias, normalizer, activation, bias_init,
              weight_norm):
            T.random.seed(1234)
            linear_kwargs = dict(use_bias=False, weight_norm=weight_norm)
            if use_bias is None:
                use_bias = normalizer is None
            m = gnn.GCNLayer(
                module=Linear(in_features, out_features, **linear_kwargs),
                self_module=(
                    Linear(in_features, out_features, **linear_kwargs)
                    if use_self_loop else None),
                self_weight=self_weight,
                merge_mode=merge_mode,
                normalizer=(
                    get_layer_from_layer_or_factory(
                        'normalizer', normalizer, args=(out_features,))
                    if normalizer is not None else None
                ),
                activation=activation,
                bias_store=(
                    SimpleParamStore(
                        [out_features * (1 + int(use_self_loop and merge_mode == 'concat'))],
                        initializer=bias_init
                    )
                    if use_bias else None),
            )
            m(x, adj)
            tk.layers.set_eval_mode(m)
            return m(x, adj)

        def f(*args, **kwargs):
            assert_allclose_(h(*args, **kwargs), g(*args, **kwargs))

        inputs = [T.random.randn(in_shape)
                  for in_shape in [[50, 3], [50, 7, 3]]]
        adj = make_random_adj_matrix(50)

        for x in inputs:
            f(x, adj, in_features=3, out_features=4,
              use_self_loop=False, self_weight=1.0, merge_mode='add',
              use_bias=False, normalizer=None, activation=None,
              bias_init=tk.init.zeros, weight_norm=False)
            f(x, adj, in_features=3, out_features=4,
              use_self_loop=False, self_weight=1.0, merge_mode='concat',
              use_bias=None, normalizer=None, activation=None,
              bias_init=tk.init.zeros, weight_norm=False)

            f(x, adj, in_features=3, out_features=4,
              use_self_loop=True, self_weight=1.0, merge_mode='add',
              use_bias=None, normalizer=tk.layers.BatchNorm,
              activation=tk.layers.LeakyReLU, bias_init=tk.init.uniform,
              weight_norm=True)
            f(x, adj, in_features=3, out_features=4,
              use_self_loop=True, self_weight=2.0, merge_mode='concat',
              use_bias=True, normalizer=tk.layers.Sigmoid(),
              activation=tk.layers.LeakyReLU(), bias_init=tk.init.uniform,
              weight_norm=True)
