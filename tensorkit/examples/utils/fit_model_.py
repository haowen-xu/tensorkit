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
              maximize_loss: bool = False,
              clip_norm: Optional[float] = None,
              global_clip_norm: Optional[float] = None,
              grad_processor: Optional[Callable[[tk.optim.Optimizer], None]] = None,
              param_names: Optional[Sequence[str]] = None):
    def step(*train_data):
        # clear the captured gradient from the last step
        optimizer.clear_grad()

        # compute the loss and capture the gradients
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
                optimizer.add_loss(loss, maximize=maximize_loss)

        # post-process the gradients
        if clip_norm is not None:
            optimizer.clip_grad_by_norm(clip_norm)
        if global_clip_norm is not None:
            optimizer.clip_grad_by_global_norm(global_clip_norm)
        if grad_processor is not None:
            grad_processor(optimizer)

        if tk.settings.validate_tensors:
            for i, (param, grad) in enumerate(optimizer.iter_params_and_grads()):
                if grad is not None:
                    message = (
                        f'grad for {param_names[i]}'
                        if param_names else f'grad for params[{i}]'
                    )
                    T.assert_finite(grad, message=message)

        # run the optimization step
        optimizer.step()

        return metrics

    param_names = list(param_names) if param_names else None
    loop.run(step, stream)
