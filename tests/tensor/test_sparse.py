import numpy as np
import pytest
from scipy import sparse as sp

from tensorkit import tensor as T
from tests.helper import *


class SparseTestCase(TestCase):

    def test_defaults(self):
        self.assertTrue(T.sparse.MAKE_SPARSE_DEFAULT_FORCE_COALESCED)

    def test_ctor_and_type_convert(self):
        def f(row, col, values, shape, dtype, force_coalesced):
            values = values.astype(np.float64)
            if dtype == T.int32 or dtype == T.int64:
                values = np.asarray(values * 1000, dtype=np.int64)

            x = make_ndarray_by_coo(row, col, values, shape)
            self.assertFalse(T.sparse.is_sparse_tensor(x))
            self.assertFalse(T.sparse.is_sparse_tensor(T.as_tensor(x)))

            the_force_coalesced = (
                force_coalesced if force_coalesced is not None
                else T.sparse.MAKE_SPARSE_DEFAULT_FORCE_COALESCED)
            the_dtype = dtype if dtype is not None else T.float64

            def g(x, y):
                self.assertTrue(T.sparse.is_sparse_tensor(y))
                self.assertEqual(T.sparse.is_coalesced(y), the_force_coalesced)
                self.assertEqual(T.sparse.get_dtype(y), the_dtype)
                self.assertEqual(T.sparse.rank(y), len(shape))
                self.assertEqual(T.sparse.length(y), shape[0])
                self.assertEqual(T.sparse.shape(y), shape)
                self.assertEqual(T.sparse.get_device(y), T.current_device())
                assert_allclose(x, y, rtol=1e-4, atol=1e-6)

            def h(x, y, t_):
                self.assertIsInstance(y, t_)
                assert_allclose(x, y)

            # make_sparse
            for coord_first in (None, True, False):
                kwargs = {}
                if coord_first is not None:
                    kwargs['coord_first'] = coord_first
                if force_coalesced is not None:
                    kwargs['force_coalesced'] = force_coalesced

                the_coord_first = (coord_first if coord_first is not None
                                   else T.sparse.SPARSE_INDICES_DEFAULT_IS_COORD_FIRST)
                y = T.sparse.make_sparse(
                    T.as_tensor(np.stack([row, col], axis=0 if the_coord_first else 1),
                                dtype=T.int64),
                    T.as_tensor(values),
                    dtype=dtype,
                    shape=shape,
                    **kwargs
                )
                g(x, y)
                self.assertEqual(T.sparse.value_count(y), len(row))

            # from_dense
            y = T.sparse.from_dense(
                T.as_tensor(x, dtype=dtype),
                **({'force_coalesced': force_coalesced}
                   if force_coalesced is not None else {})
            )
            self.assertTrue(T.sparse.is_sparse_tensor(y))
            g(x, y)
            t = T.sparse.to_dense(y)
            self.assertFalse(T.sparse.is_sparse_tensor(t))
            h(x, t, T.Tensor)

            # from_numpy
            y = T.sparse.from_numpy(
                x, dtype=dtype,
                **({'force_coalesced': force_coalesced}
                   if force_coalesced is not None else {})
            )
            g(x, y)
            h(x, T.sparse.to_numpy(y), np.ndarray)

            # from_coomatrix
            if len(shape) == 2:
                spmat = sp.coo_matrix((values, (row, col)), shape=shape)
                for m in (spmat, spmat.tocsr()):
                    y = T.sparse.from_spmatrix(
                        m,
                        dtype=dtype,
                        **({'force_coalesced': force_coalesced}
                           if force_coalesced is not None else {})
                    )
                    g(x, y)
                    h(m, T.sparse.to_spmatrix(y), sp.spmatrix)

            # dtype from another sparse tensor
            z = T.sparse.make_sparse(
                T.as_tensor(np.stack([row, col], axis=0)),
                T.as_tensor(values),
                dtype=y.dtype,
                shape=shape,
            )
            self.assertEqual(T.sparse.get_dtype(z), T.sparse.get_dtype(y))

            # test to_dtype
            z = T.sparse.make_sparse(
                T.as_tensor(np.stack([row, col], axis=0)),
                T.as_tensor(values),
                shape=shape,
            )
            z = T.sparse.to_dtype(z, T.sparse.get_dtype(y))
            self.assertEqual(T.sparse.get_dtype(z), T.sparse.get_dtype(y))

        # test ordinary
        for force_coalesced in [None, True, False]:
            for dtype in [None, 'float32', T.float64, T.int32, 'int64']:
                f(row=np.array([0, 0, 1, 3, 4]), col=np.array([1, 4, 5, 3, 2]),
                  values=np.random.randn(5), shape=[5, 6], dtype=dtype,
                  force_coalesced=force_coalesced)
                f(row=np.array([0, 0, 1, 3, 4]), col=np.array([1, 4, 5, 3, 2]),
                  values=np.random.randn(10).reshape([5, 2]), shape=[5, 6, 2],
                  dtype=dtype, force_coalesced=force_coalesced)

        # test with_device and to_device
        f = lambda: T.sparse.make_sparse(
            T.as_tensor([[0, 1], [1, 0]]),
            T.random.randn([2]),
            shape=[3, 3]
        )
        with T.use_device(T.CPU_DEVICE):
            t = f()
            self.assertEqual(T.sparse.get_device(t), T.CPU_DEVICE)

        t = T.sparse.to_device(f(), T.CPU_DEVICE)
        self.assertEqual(T.sparse.get_device(t), T.CPU_DEVICE)

        # test errors
        with pytest.raises(ValueError, match='`indices` must be a 2d tensor'):
            _ = T.sparse.make_sparse(
                T.zeros([2, 3, 4]), T.random.randn([5]), shape=[5, 5])

        with pytest.raises(ValueError, match='`dtype` not supported'):
            _ = T.sparse.make_sparse(
                T.as_tensor([[0, 1], [1, 0]]),
                T.random.randn([2]),
                shape=[3, 3], dtype=T.boolean,
            )

        with pytest.raises(ValueError, match='`indices` must be a int32 or '
                                             'int64 tensor'):
            _ = T.sparse.make_sparse(
                T.as_tensor([[0, 1], [1, 0]], dtype=T.int16),
                T.random.randn([2]),
                shape=[3, 3]
            )

    def test_coalesce_and_get_indices_values(self):
        row, col = np.array([0, 1, 2]), np.array([2, 0, 1])
        values = np.array([1., 2., 3.])
        shape = [3, 3]
        x = make_ndarray_by_coo(row, col, values, shape)
        for force_coalesced in (False, True):
            t = T.sparse.make_sparse(
                T.as_tensor([[0, 0, 2, 1], [2, 2, 1, 0]], dtype=T.int32),
                T.as_tensor([.5, .5, 3., 2.]),
                shape=shape, coord_first=True, force_coalesced=force_coalesced,
            )

            self.assertEqual(T.sparse.is_coalesced(t), force_coalesced)
            if force_coalesced:
                self.assertIs(T.sparse.coalesce(t), t)
            else:
                t2 = T.sparse.coalesce(t)
                self.assertIsNot(t2, t)
                t = t2

            assert_allclose(t, x)
            assert_equal(
                T.sparse.get_indices(t),
                np.stack(
                    [row, col],
                    axis=(0 if T.sparse.SPARSE_INDICES_DEFAULT_IS_COORD_FIRST
                          else 1)
                )
            )
            assert_equal(
                T.sparse.get_indices(t, coord_first=True),
                np.stack([row, col], axis=0)
            )
            assert_equal(
                T.sparse.get_indices(t, coord_first=False),
                np.stack([row, col], axis=1)
            )
            assert_equal(T.sparse.get_values(t), values)

    def test_eye(self):
        for dtype in [None, T.float32, T.float64, T.int32, T.int64]:
            if dtype is not None:
                the_dtype = dtype
                kwargs = {'dtype': dtype}
            else:
                the_dtype = T.float32
                kwargs = {}

            for n, m in zip([2, 3, 4], [2, 3, 4]):
                y = T.sparse.eye(n, m, **kwargs)
                assert_equal(np.eye(n, m), y)
                self.assertEqual(T.sparse.get_dtype(y), the_dtype)
                self.assertEqual(T.sparse.get_device(y), T.current_device())

        with T.use_device(T.CPU_DEVICE):
            y = T.sparse.eye(5)
            assert_equal(np.eye(5), y)
            self.assertEqual(T.sparse.get_device(y), T.CPU_DEVICE)

    def test_reduce_sum(self):
        indices = T.as_tensor(np.random.randint(0, 50, size=[2, 200]))
        values = T.random.randn([200])
        shape = [60, 50]

        for force_coalesced in [False, True]:
            x = T.sparse.make_sparse(
                indices, values, shape=shape, force_coalesced=force_coalesced)
            y = T.sparse.to_numpy(x)
            for axis in (None, 0, 1, -1, -2):
                assert_allclose(
                    T.sparse.reduce_sum(x, axis=axis),
                    np.sum(y, axis=axis),
                    rtol=1e-4, atol=1e-6,
                )

    def test_no_element_sparse_tensor(self):
        for coord_first in [True, False]:
            # construct the no-element sparse tensor
            indices_shape = [2, 0] if coord_first else [0, 2]
            x = T.sparse.make_sparse(
                T.zeros(indices_shape, dtype=T.index_dtype),
                T.zeros([0], dtype=T.float32),
                coord_first=coord_first,
                shape=[3, 4],
            )
            assert_allclose(x, np.zeros([3, 4]))
            self.assertEqual(T.sparse.value_count(x), 0)

    def test_matmul(self):
        indices = T.as_tensor(np.random.randint(0, 50, size=[2, 200]))
        values = T.random.randn([200])
        shape = [60, 50]
        y = T.random.randn([50, 30])

        for force_coalesced in [False, True]:
            x = T.sparse.make_sparse(
                indices, values, shape=shape, force_coalesced=force_coalesced)
            assert_allclose(
                T.sparse.matmul(x, y),
                np.dot(T.sparse.to_numpy(x), T.to_numpy(y)),
                rtol=1e-4, atol=1e-6,
            )

    def test_grad(self):
        def f():
            indices = T.as_tensor([[0, 1], [1, 0]])
            values = T.requires_grad(T.as_tensor([0.1, 0.2]))
            shape = [2, 2]
            x = T.sparse.make_sparse(indices, values, shape=shape)
            return values, x

        values, x = f()
        y = T.sparse.reduce_sum(x * x)
        [grad] = T.grad([y], [values])
        assert_allclose(grad, 2 * values, atol=1e-4, rtol=1e-6)

        values, x = f()
        y = T.sparse.reduce_sum(T.sparse.stop_grad(x) * x)
        [grad] = T.grad([y], [values])
        assert_allclose(grad, values, atol=1e-4, rtol=1e-6)
