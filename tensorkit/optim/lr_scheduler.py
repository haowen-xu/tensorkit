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

    loop: Optional[mltk.TrainLoop]
    optimizer: Optimizer

    def __init__(self, optimizer: Optimizer):
        self.loop = None
        self.optimizer = optimizer

    def bind(self, loop: mltk.TrainLoop):
        if self.loop is not None:
            if loop is self.loop:
                return
            raise RuntimeError('Already bind to a train loop.')
        self.loop = loop
        self._bind_events(loop)
        self.update_lr()

    def update_lr(self):
        """Update the learning rate of the optimizer according to the loop."""
        raise NotImplementedError()

    def unbind(self):
        if self.loop is not None:
            self._unbind_events(self.loop)
            self.loop = None

    def _bind_events(self, loop: mltk.TrainLoop):
        raise NotImplementedError()

    def _unbind_events(self, loop: mltk.TrainLoop):
        raise NotImplementedError()


class AnnealingLR(LRScheduler):
    """
    Learning rate scheduler that anneals the learning rate after every few
    `epochs`, by a specified `ratio`.
    """

    initial_lr: float
    ratio: float
    epochs: int

    def __init__(self,
                 optimizer: Optimizer,
                 initial_lr: float,
                 ratio: float,
                 epochs: int
                 ):
        self.initial_lr = float(initial_lr)
        self.ratio = float(ratio)
        self.epochs = int(epochs)
        super().__init__(optimizer)

    def _bind_events(self, loop: mltk.TrainLoop):
        loop.on_epoch_end.do(self.update_lr)

    def _unbind_events(self, loop: mltk.TrainLoop):
        loop.on_epoch_end.cancel_do(self.update_lr)

    def update_lr(self):
        n_cycles = int(self.loop.epoch // self.epochs)
        lr_discount = self.ratio ** n_cycles
        self.optimizer.set_lr(self.initial_lr * lr_discount)
