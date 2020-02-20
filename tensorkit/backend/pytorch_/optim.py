from contextlib import contextmanager
from typing import *

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from .core import *

__all__ = [
    'Optimizer', 'SGD', 'Adam',
]


class Optimizer(object):

    @property
    def lr(self) -> float:
        raise NotImplementedError()

    def set_lr(self, lr: float):
        raise NotImplementedError()

    def add_param_group(self, params: Iterator[Variable]):
        raise NotImplementedError()

    def clear_grad(self):
        raise NotImplementedError()

    @contextmanager
    def capture_grad(self) -> Generator[None, None, None]:
        raise NotImplementedError()

    def minimize(self, loss: Tensor):
        raise NotImplementedError()

    def maximize(self, loss: Tensor):
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError()


class BackendOptimizer(Optimizer):

    _lr: float = None
    torch_optimizer: TorchOptimizer

    def __init__(self,
                 lr: float,
                 torch_optimizer: TorchOptimizer):
        self.torch_optimizer = torch_optimizer
        self.set_lr(lr)

    @property
    def lr(self) -> float:
        return self._lr

    def set_lr(self, lr: float):
        if self._lr != lr:
            for group in self.torch_optimizer.param_groups:
                group['lr'] = lr
        self._lr = lr

    def add_param_group(self, params: Iterator[Variable]):
        self.torch_optimizer.add_param_group({
            'params': list(params),
            'lr': self._lr,
        })

    def clear_grad(self):
        self.torch_optimizer.zero_grad()

    @contextmanager
    def capture_grad(self) -> Generator[None, None, None]:
        yield

    def minimize(self, loss: Tensor):
        loss.backward()
        self.torch_optimizer.step()

    def maximize(self, loss: Tensor):
        self.minimize(-loss)

    def state_dict(self) -> Dict[str, Any]:
        return self.torch_optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.torch_optimizer.load_state_dict(state_dict)

        # ensure that we've got all state on the same device as the parameters.
        device = self.torch_optimizer.param_groups[0]['params'][0].device
        for state in self.torch_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device)


class SGD(BackendOptimizer):

    def __init__(self,
                 params: Iterable[Variable],
                 lr: float,
                 momentum: float = 0.,
                 nesterov: bool = False):
        """
        Construct a new :class:`SGD` optimizer.

        Args:
            params: The parameters to be optimized.
            lr: The learning rate.
            momentum: The momentum.  Typically 0.9 for momentum SGD optimization.
            nesterov: Whether or not to use Nesterov momentum optimizer?
        """
        super().__init__(
            lr=lr,
            torch_optimizer=torch.optim.SGD(
                params=params,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
            )
        )


class Adam(BackendOptimizer):

    def __init__(self,
                 params: Iterable[Variable],
                 lr: float = 1e-3,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 amsgrad: bool = False):
        super().__init__(
            lr=lr,
            torch_optimizer=torch.optim.Adam(
                params=params,
                lr=lr,
                betas=(beta_1, beta_2),
                eps=epsilon,
                amsgrad=amsgrad,
            )
        )