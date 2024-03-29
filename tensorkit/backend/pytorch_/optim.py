from contextlib import contextmanager
from typing import *

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer

from .core import *

__all__ = [
    'Optimizer', 'BackendOptimizer',
    'SGD', 'Adadelta', 'Adagrad', 'Adam', 'Adamax', 'RMSprop',
]


class Optimizer(object):

    @property
    def lr(self) -> float:
        raise NotImplementedError()

    def set_lr(self, lr: float):
        raise NotImplementedError()

    def add_params(self, params: Iterator[Variable]):
        raise NotImplementedError()

    def iter_params(self) -> Iterator[Variable]:
        raise NotImplementedError()

    def iter_params_and_grads(self) -> Iterator[Tuple[Variable, Optional[Tensor]]]:
        raise NotImplementedError()

    def set_param_grad(self, param: Variable, grad: Optional[Tensor]):
        raise NotImplementedError()

    def clear_grad(self):
        raise NotImplementedError()

    @contextmanager
    def capture_grad(self) -> Generator[None, None, None]:
        raise NotImplementedError()

    def clip_grad_by_value(self, clip_min: float, clip_max: float):
        for param, grad in self.iter_params_and_grads():
            if grad is not None:
                self.set_param_grad(param, clip(grad, clip_min, clip_max))

    def clip_grad_by_norm(self, clip_norm: float):
        for param, grad in self.iter_params_and_grads():
            if grad is not None:
                self.set_param_grad(param, clip_by_norm(grad, clip_norm))

    def clip_grad_by_global_norm(self, clip_norm: float):
        params = []
        grads = []
        for param, grad in self.iter_params_and_grads():
            if grad is not None:
                params.append(param)
                grads.append(grad)
        if grads:
            grads = clip_by_global_norm(grads, clip_norm)
            for param, grad in zip(params, grads):
                self.set_param_grad(param, grad)

    def add_loss(self, loss: Tensor, maximize: bool = False):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        raise NotImplementedError()


class BackendOptimizer(Optimizer):

    _lr: float = None
    _in_capture_context: bool = False
    torch_optimizer: TorchOptimizer
    params: List[Variable]  # all parameters, without partitioned into groups

    def __init__(self,
                 params: Iterable[Variable],
                 lr: float,
                 torch_optimizer_factory: Callable[[], TorchOptimizer]):
        self.params = []
        for p in params:
            if any(id(p) == id(pp) for pp in self.params):
                raise ValueError(f'Duplicated parameter: {p!r}')
            self.params.append(p)

        self.torch_optimizer = torch_optimizer_factory()
        self.set_lr(lr)

    @property
    def lr(self) -> float:
        return self._lr

    def set_lr(self, lr: float):
        if self._lr != lr:
            for group in self.torch_optimizer.param_groups:
                group['lr'] = lr
        self._lr = lr

    def add_params(self, params: Iterator[Variable]):
        params = list(params)
        for p in params:
            if any(id(p) == id(pp) for pp in self.params):
                raise ValueError(f'Duplicated parameter: {p!r}')
        self.params.extend(params)
        self.torch_optimizer.add_param_group({
            'params': params,
            'lr': self._lr,
        })

    def iter_params(self) -> Iterator[Variable]:
        return iter(self.params)

    def iter_params_and_grads(self) -> Iterator[Tuple[Variable, Optional[Tensor]]]:
        for p in self.params:
            yield p, p.grad

    def set_param_grad(self, param: Variable, grad: Optional[Tensor]):
        param.grad = grad

    def clear_grad(self):
        self.torch_optimizer.zero_grad()

    @contextmanager
    def capture_grad(self) -> Generator[None, None, None]:
        self._in_capture_context = True
        try:
            yield
        finally:
            self._in_capture_context = False

    def add_loss(self, loss: Tensor, maximize: bool = False):
        if not self._in_capture_context:
            raise RuntimeError(
                '`add_loss()` must be called inside the `capture_grad()` context.')

        if maximize:
            loss = -loss
        loss.backward()

    def step(self):
        if self._in_capture_context:
            raise RuntimeError(
                '`step()` must be called outside the `capture_grad()` context.')
        self.torch_optimizer.step()

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
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.SGD(
                params=params,
                lr=lr,
                momentum=momentum,
                nesterov=nesterov,
            )
        )


class Adadelta(BackendOptimizer):

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.Adadelta(
                params=params,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
            )
        )


class Adagrad(BackendOptimizer):

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0,
                 initial_accumulator_value=0, eps=1e-10):
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.Adagrad(
                params=params,
                lr=lr,
                lr_decay=lr_decay,
                weight_decay=weight_decay,
                initial_accumulator_value=initial_accumulator_value,
                eps=eps,
            )
        )


class Adam(BackendOptimizer):
    """
    The Adam algorithm from
    D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization”.
    """

    def __init__(self,
                 params: Iterable[Variable],
                 lr: float = 1e-3,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 amsgrad: bool = False):
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.Adam(
                params=params,
                lr=lr,
                betas=(beta_1, beta_2),
                eps=epsilon,
                amsgrad=amsgrad,
            )
        )


class Adamax(BackendOptimizer):
    """
    The AdaMax algorithm from
    D. P. Kingma and J. Ba, “Adam: A Method for Stochastic Optimization”.
    """

    def __init__(self,
                 params: Iterable[Variable],
                 lr: float = 0.002,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8):
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.Adamax(
                params=params,
                lr=lr,
                betas=(beta_1, beta_2),
                eps=epsilon,
            )
        )


class RMSprop(BackendOptimizer):

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False):
        params = list(params)
        super().__init__(
            params=params,
            lr=lr,
            torch_optimizer_factory=lambda: torch.optim.RMSprop(
                params=params,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
            )
        )
