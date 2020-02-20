from typing import *

import mltk

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.typing_ import TensorOrData

__all__ = ['fit_model']


def fit_model(loop: mltk.TrainLoop,
              optimizer: tk.optim.Optimizer,
              fn: Callable[..., Dict[str, TensorOrData]],
              stream: mltk.DataStream,
              loss_metric: str = 'loss',
              minimize_loss: bool = True):
    def step(*train_data):
        optimizer.clear_grad()
        with optimizer.capture_grad():
            metrics = fn(*train_data)
            try:
                loss = metrics[loss_metric]
                if not isinstance(loss, T.Tensor):
                    raise TypeError()
            except Exception:
                raise ValueError(
                    f'`train_fn` is expected to return a dict, carrying '
                    f'the train loss in the "{loss_metric}" entry: got '
                    f'{metrics!r}.'
                )
            else:
                if minimize_loss:
                    optimizer.minimize(loss)
                else:
                    optimizer.maximize(loss)
            return metrics

    loop.run(step, stream)
