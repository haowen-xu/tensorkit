import os
from functools import partial
from tempfile import TemporaryDirectory

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

    # test optimize a
    optimizer.clear_grad()
    with optimizer.capture_grad():
        loss = calculate_loss(a, b)
        optimizer.minimize(loss)
        ctx.assertLessEqual(calculate_loss(a, b), loss)
        assert_not_equal(a, 123.)
        assert_equal(b, 456.)

    # test optimize a and b
    optimizer.clear_grad()
    optimizer.add_param_group(iter([b]))
    with optimizer.capture_grad():
        loss = calculate_loss(a, b)
        optimizer.minimize(loss)
        ctx.assertLessEqual(calculate_loss(a, b), loss)
        assert_not_equal(a, 123.)
        assert_not_equal(b, 456.)

    # save checkpoint
    with TemporaryDirectory() as temp_dir:
        ckpt_path = os.path.join(temp_dir, 'ckpt')
        checkpoint = tk.train.Checkpoint(optimizer=optimizer)
        checkpoint.save(ckpt_path)

        # test backup and restore the status
        a2 = T.variable([], initializer=a)
        b2 = T.variable([], initializer=b)
        optimizer2 = optimizer_factory([a2], lr)
        optimizer2.add_param_group([b2])
        checkpoint2 = tk.train.Checkpoint(optimizer=optimizer2)
        checkpoint2.restore(ckpt_path)

        with optimizer2.capture_grad():
            loss = calculate_loss(a2, b2)
            optimizer2.minimize(loss)
            ctx.assertLessEqual(calculate_loss(a2, b2), loss)
            assert_not_equal(a2, a)
            assert_not_equal(b2, b)

        # test backup and restore the status, and use maximize instead of minimize
        a3 = T.variable([], initializer=a)
        b3 = T.variable([], initializer=b)
        optimizer3 = optimizer_factory([a3], lr)
        optimizer3.add_param_group([b3])
        checkpoint3 = tk.train.Checkpoint(optimizer=optimizer3)
        checkpoint3.restore(ckpt_path)

        with optimizer3.capture_grad():
            loss = calculate_loss(a3, b3)
            optimizer3.maximize(-loss)
            ctx.assertLessEqual(calculate_loss(a3, b3), loss)
            assert_allclose(a3, a2)
            assert_allclose(b3, b2)
            assert_allclose(calculate_loss(a3, b3), calculate_loss(a2, b2))

        # backup and restore the status, change the learning rate and get
        # the third output, and compare to the result with optimizer2
        a4 = T.variable([], initializer=a)
        b4 = T.variable([], initializer=b)
        optimizer4 = optimizer_factory([a4], lr)
        optimizer4.add_param_group([b4])
        checkpoint4 = tk.train.Checkpoint(optimizer=optimizer4)
        checkpoint4.restore(ckpt_path)

        optimizer4.set_lr(lr * 0.5)
        ctx.assertEqual(optimizer4.lr, lr * 0.5)
        with optimizer4.capture_grad():
            loss = calculate_loss(a4, b4)
            optimizer4.minimize(loss)
            assert_not_allclose(a4, a2)
            assert_not_allclose(b4, b2)
            assert_not_allclose(calculate_loss(a4, b4), calculate_loss(a2, b2))

    # now proceed the optimization from the first optimizer, and compare
    # the result with optimizer2
    optimizer.clear_grad()
    with optimizer.capture_grad():
        loss = calculate_loss(a, b)
        optimizer.minimize(loss)
        ctx.assertLessEqual(calculate_loss(a, b), loss)
        assert_allclose(a, a2)
        assert_allclose(b, b2)
        assert_allclose(calculate_loss(a, b), calculate_loss(a2, b2))


class OptimizerTestCase(TestCase):

    def test_sgd(self):
        optimizer_standard_check(self, partial(tk.optim.SGD), 0.001)
        optimizer_standard_check(self, partial(tk.optim.SGD, momentum=0.9), 0.001)
        optimizer_standard_check(self, partial(tk.optim.SGD, momentum=0.9, nesterov=True), 0.001)

    def test_adam(self):
        optimizer_standard_check(self, partial(tk.optim.Adam), 0.1)
        optimizer_standard_check(self, partial(tk.optim.Adam, amsgrad=True), 0.1)
