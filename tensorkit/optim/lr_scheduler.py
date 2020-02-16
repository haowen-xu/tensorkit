from typing import *

import mltk

from .core import *

__all__ = [
    'LRScheduler', 'AnnealingLR',
]


class LRScheduler(object):
    """
    Base class that schedules the learning rate of an optimizer
    during a :class:`mltk.TrainLoop`.
    """

    loop: mltk.TrainLoop
    optimizer: Optimizer

    def __init__(self,
                 loop: mltk.TrainLoop,
                 optimizer: Optimizer):
        self.loop = loop
        self.optimizer = optimizer
        self._bind_events(loop)
        self.update_lr()

    def update_lr(self):
        """Update the learning rate of the optimizer according to the loop."""
        raise NotImplementedError()

    def close(self):
        """Close this scheduler, such that it will no longer affect the optimizer."""
        self._unbind_events(self.loop)

    def _bind_events(self, loop: mltk.TrainLoop):
        raise NotImplementedError()

    def _unbind_events(self, loop: mltk.TrainLoop):
        raise NotImplementedError()


class AnnealingLR(LRScheduler):

    initial_lr: float
    ratio: float
    epochs: int

    def __init__(self,
                 loop: mltk.TrainLoop,
                 optimizer: Optimizer,
                 initial_lr: float,
                 ratio: float,
                 epochs: int
                 ):
        self.initial_lr = float(initial_lr)
        self.ratio = float(ratio)
        self.epochs = int(epochs)
        super().__init__(loop, optimizer)

    def _bind_events(self, loop: mltk.TrainLoop):
        loop.on_epoch_end.do(self.update_lr)

    def _unbind_events(self, loop: mltk.TrainLoop):
        loop.on_epoch_end.cancel_do(self.update_lr)

    def update_lr(self):
        n_cycles = int(self.loop.epoch // self.epochs)
        lr_discount = self.ratio ** n_cycles
        self.optimizer.set_lr(self.initial_lr * lr_discount)
