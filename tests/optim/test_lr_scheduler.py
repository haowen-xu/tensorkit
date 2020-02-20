import mltk
from mock import Mock

import tensorkit as tk
from tests.helper import *


class _MyFakeOptimizer(object):
    def __init__(self, lr):
        self.lr = lr

    def set_lr(self, lr):
        self.lr = lr


def standard_lr_scheduler_check(ctx, scheduler_factory, lr_func):
    # test start with eopch = 0
    optimizer = _MyFakeOptimizer(0.1)
    assert_allclose(optimizer.lr, 0.1)

    ev_hosts = mltk.EventHost()
    loop = Mock(epoch=0, on_epoch_end=ev_hosts['on_epoch_end'])
    scheduler = scheduler_factory(loop, optimizer)
    assert_allclose(optimizer.lr, lr_func(loop, optimizer))

    for epoch in range(1, 29):
        loop.epoch = epoch
        if epoch < 15:
            scheduler.update_lr()
        else:
            ev_hosts.fire('on_epoch_end')
        assert_allclose(optimizer.lr, lr_func(loop, optimizer))

    final_lr = optimizer.lr
    scheduler.unbind_events()
    for epoch in range(29, 39):
        loop.epoch = epoch
        ev_hosts.fire('on_epoch_end')
        assert_allclose(optimizer.lr, final_lr)

    for epoch in range(29, 39):
        loop.epoch = epoch
        scheduler.update_lr()  # still can update the lr if manually called
        assert_allclose(optimizer.lr, lr_func(loop, optimizer))

    # test start with epoch = some value
    optimizer = _MyFakeOptimizer(0.1)
    assert_allclose(optimizer.lr, 0.1)

    ev_hosts = mltk.EventHost()
    loop = Mock(epoch=40, on_epoch_end=ev_hosts['on_epoch_end'])
    scheduler = scheduler_factory(loop, optimizer)
    assert_allclose(optimizer.lr, lr_func(loop, optimizer))


class LRSchedulerTestCaes(TestCase):

    def test_annealing_lr(self):
        standard_lr_scheduler_check(
            self,
            lambda loop, optimizer: tk.optim.lr_scheduler.AnnealingLR(
                loop, optimizer, initial_lr=0.01, ratio=0.5, epochs=2
            ),
            lambda loop, optimizer: 0.01 * 0.5 ** int(loop.epoch // 2)
        )
