import os
from functools import partial
from tempfile import TemporaryDirectory

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


def optimizer_standard_check(ctx, optimizer_factory, lr):
    a = T.variable([], initializer=123.)
    b = T.variable([], initializer=456.)

    def calculate_loss(a, b):
        return (a + b) ** 2

    optimizer = optimizer_factory(iter([a]), lr)
    ctx.assertEqual(optimizer.lr, lr)
    ctx.assertEqual(list(optimizer.iter_params()), [a])

    with pytest.raises(ValueError, match='Duplicated parameter'):
        optimizer.add_params([a])

    with pytest.raises(ValueError, match='Duplicated parameter'):
        _ = optimizer_factory([a, a], lr)

    # test optimize a
    optimizer.clear_grad()
    with optimizer.capture_grad():
        loss = calculate_loss(a, b)
        optimizer.add_loss(loss)
    optimizer.step()
    ctx.assertLessEqual(calculate_loss(a, b), loss)
    assert_not_equal(a, 123.)
    assert_equal(b, 456.)

    # test optimize a and b,
    # and also using 'set_param_grad' to optimize a0 and b0
    T.random.seed(1234)
    optimizer = optimizer_factory(iter([a]), lr)
    optimizer.add_params(iter([b]))
    ctx.assertEqual(list(optimizer.iter_params()), [a, b])

    T.random.seed(1234)
    a0 = T.variable([], initializer=a)
    b0 = T.variable([], initializer=b)
    optimizer0 = optimizer_factory([a0], lr)
    optimizer0.add_params([b0])

    with optimizer.capture_grad():
        loss = calculate_loss(a, b)
        optimizer.add_loss(loss)

    # copy grads to optimizer0
    params_and_grads = list(optimizer.iter_params_and_grads())
    ctx.assertEqual(len(params_and_grads), 2)
    ctx.assertIs(params_and_grads[0][0], a)
    ctx.assertIs(params_and_grads[1][0], b)
    optimizer0.set_param_grad(
        a0, T.as_tensor(params_and_grads[0][1], force_copy=True))
    optimizer0.set_param_grad(
        b0, T.as_tensor(params_and_grads[1][1], force_copy=True))

    optimizer.step()
    ctx.assertLessEqual(calculate_loss(a, b), loss)
    assert_not_equal(a, 123.)
    assert_not_equal(b, 456.)

    optimizer0.step()
    assert_allclose(calculate_loss(a0, b0), calculate_loss(a, b))
    assert_allclose(a0, a)
    assert_allclose(b0, b)

    # save checkpoint
    with TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, 'ckpt')
        checkpoint = tk.train.Checkpoint(optimizer=optimizer)
        checkpoint.save(ckpt_path)

        # test backup and restore the status
        a2 = T.variable([], initializer=a)
        b2 = T.variable([], initializer=b)
        optimizer2 = optimizer_factory([a2], lr)
        optimizer2.add_params([b2])
        checkpoint2 = tk.train.Checkpoint(optimizer=optimizer2)
        checkpoint2.restore(ckpt_path)

        with optimizer2.capture_grad():
            loss = calculate_loss(a2, b2)
            optimizer2.add_loss(loss)
        optimizer2.step()
        ctx.assertLessEqual(calculate_loss(a2, b2), loss)
        assert_not_equal(a2, a)
        assert_not_equal(b2, b)

        # test backup and restore the status, and use maximize instead of minimize
        a3 = T.variable([], initializer=a)
        b3 = T.variable([], initializer=b)
        optimizer3 = optimizer_factory([a3], lr)
        optimizer3.add_params([b3])
        checkpoint3 = tk.train.Checkpoint(optimizer=optimizer3)
        checkpoint3.restore(ckpt_path)

        with optimizer3.capture_grad():
            loss = calculate_loss(a3, b3)
            optimizer3.add_loss(-loss, maximize=True)
        optimizer3.step()
        ctx.assertLessEqual(calculate_loss(a3, b3), loss)
        assert_allclose(a3, a2)
        assert_allclose(b3, b2)
        assert_allclose(calculate_loss(a3, b3), calculate_loss(a2, b2))

        # backup and restore the status, change the learning rate and get
        # the third output, and compare to the result with optimizer2
        a4 = T.variable([], initializer=a)
        b4 = T.variable([], initializer=b)
        optimizer4 = optimizer_factory([a4], lr)
        optimizer4.add_params([b4])
        checkpoint4 = tk.train.Checkpoint(optimizer=optimizer4)
        checkpoint4.restore(ckpt_path)

        optimizer4.set_lr(lr * 0.5)
        ctx.assertEqual(optimizer4.lr, lr * 0.5)
        with optimizer4.capture_grad():
            loss = calculate_loss(a4, b4)
            optimizer4.add_loss(loss)
        optimizer4.step()
        assert_not_allclose(a4, a2)
        assert_not_allclose(b4, b2)
        assert_not_allclose(calculate_loss(a4, b4), calculate_loss(a2, b2))

    # now proceed the optimization from the first optimizer, and compare
    # the result with optimizer2
    optimizer.clear_grad()
    with optimizer.capture_grad():
        loss = calculate_loss(a, b)
        optimizer.add_loss(loss)
    optimizer.step()
    ctx.assertLessEqual(calculate_loss(a, b), loss)
    assert_allclose(a, a2)
    assert_allclose(b, b2)
    assert_allclose(calculate_loss(a, b), calculate_loss(a2, b2))

    # test context
    optimizer.clear_grad()
    with pytest.raises(RuntimeError,
                       match=r'`add_loss\(\)` must be called inside the '
                             r'`capture_grad\(\)` context'):
        optimizer.add_loss(calculate_loss(a, b))

    optimizer.clear_grad()
    with optimizer.capture_grad():
        optimizer.add_loss(calculate_loss(a, b))
        with pytest.raises(RuntimeError,
                           match=r'`step\(\)` must be called outside the '
                                 r'`capture_grad\(\)` context'):
            optimizer.step()

    # test clip grads
    def check_clip_grad(optimizer_fn, naive_fn):
        def f(g):
            a = T.variable([], initializer=123.)
            b = T.variable([], initializer=456.)
            c = T.variable([], initializer=789.)
            T.random.seed(1234)
            optimizer = optimizer_factory([a, b, c], lr)

            with optimizer.capture_grad():
                optimizer.add_loss((a + b) ** 2)
            g(optimizer)
            optimizer.step()

            return [T.to_numpy(t) for t in (a, b, c, (a + b) ** 2)]

        def h(optimizer):
            params = []
            grads = []
            for param, grad in optimizer.iter_params_and_grads():
                if grad is not None:
                    params.append(param)
                    grads.append(grad)
            grads = naive_fn(grads)
            for param, grad in zip(params, grads):
                optimizer.set_param_grad(param, grad)

        a, b, c, loss = f(lambda optimizer: optimizer_fn(optimizer))
        a0, b0, c0, loss0 = f(h)

        for t, t0 in zip((a, b, c, loss), (a0, b0, c0, loss0)):
            assert_allclose(t, t0, rtol=1e-4, atol=1e-6)

    def naive_clip_by_value(grads, clip_min, clip_max):
        return [T.clip(g, clip_min, clip_max) for g in grads]

    def naive_clip_by_norm(grads, clip_norm):
        return [T.clip_by_norm(g, clip_norm) for g in grads]

    def naive_clip_by_global_norm(grads, clip_norm):
        return T.clip_by_global_norm(grads, clip_norm)

    for v in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        check_clip_grad(
            lambda optimizer: optimizer.clip_grad_by_value(-v, v),
            lambda grads: naive_clip_by_value(grads, -v, v),
        )
        check_clip_grad(
            lambda optimizer: optimizer.clip_grad_by_norm(v),
            lambda grads: naive_clip_by_norm(grads, v),
        )
        check_clip_grad(
            lambda optimizer: optimizer.clip_grad_by_global_norm(v),
            lambda grads: naive_clip_by_global_norm(grads, v),
        )


class OptimizerTestCase(TestCase):

    def test_sgd(self):
        optimizer_standard_check(self, partial(tk.optim.SGD), 0.001)
        optimizer_standard_check(self, partial(tk.optim.SGD, momentum=0.9), 0.001)
        optimizer_standard_check(self, partial(tk.optim.SGD, momentum=0.9, nesterov=True), 0.001)

    def test_adam(self):
        optimizer_standard_check(self, partial(tk.optim.Adam), 0.1)
        optimizer_standard_check(self, partial(tk.optim.Adam, amsgrad=True), 0.1)
