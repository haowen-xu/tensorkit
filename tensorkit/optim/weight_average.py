from contextlib import contextmanager
from typing import *

from mltk import StatefulObject

from .. import tensor as T

__all__ = [
    'WeightAveraging',
    'WeightMeanAveraging', 'StochasticWeightAveraging',
    'WeightMovingAveraging',
]


class WeightAveraging(StatefulObject):

    enabled: bool
    weights: List[T.Variable]
    averages: List[T.Variable]
    num_updates: int

    def __init__(self,
                 weights: Iterable[T.Variable],
                 enabled: bool = True):
        self.enabled = bool(enabled)
        self.weights = list(weights)
        self.averages = [
            T.variable(
                shape=T.shape(weight),
                dtype=T.get_dtype(weight),
                device=T.get_device(weight),
                requires_grad=False,
            )
            for weight in self.weights
        ]
        self.num_updates = 0

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'averages': self.averages,
            'num_updates': self.num_updates,
        }

    def set_state_dict(self, state: Dict[str, Any]):
        enabled = bool(state['enabled'])
        averages = state['averages']
        num_updates = int(state['num_updates'])
        if len(averages) != len(self.weights):
            raise ValueError('Bad state: `averages` does not match the size '
                             'of `self.weights`.')
        for avg, new_avg in zip(self.averages, averages):
            T.assign(avg, new_avg)
        self.enabled = enabled
        self.num_updates = num_updates

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def _get_debiased_average(self, average: T.Variable):
        raise NotImplementedError()

    def _update_single(self, average: T.Variable, weight: Union[T.Tensor, T.Variable]):
        raise NotImplementedError()

    def update(self):
        """
        Update the average values according to the weight values at the moment.
        This method will take effect only if `self.enabled` is True.
        """
        if self.enabled:
            for avg, weight in zip(self.averages, self.weights):
                self._update_single(avg, weight)
            self.num_updates += 1

    def commit(self,
               backup: bool = False,
               device: Optional[str] = None) -> List[T.Tensor]:
        """
        Commit the averaged weight to the weight variables.
        This method will update the weights even if `self.enabled` is False.

        Args:
            backup: If True, will return a copy of the original weight values.
            device: Where to put the copied tensors?

        Returns:
            The original weight values before commit, if `backup` is True.
        """
        ret = []

        for avg, weight in zip(self.averages, self.weights):
            avg_val = self._get_debiased_average(avg)
            if backup:
                ret.append(T.copy(
                    weight, device=device or T.get_device(weight),
                    requires_grad=False
                ))
            T.assign(weight, avg_val)

        if backup:
            return ret

    @contextmanager
    def temporarily_commit(self):
        """
        Temporarily commit the averaged weight values within a context,
        and restore on exit.
        """
        # backup the original weight values
        backup_weights = self.commit(backup=True)
        try:
            yield
        finally:
            for weight, val in zip(self.weights, backup_weights):
                T.assign(weight, val)


class WeightMeanAveraging(WeightAveraging):
    """
    Averaging the weight by statistical mean.

    Also known as the stochastic Weight Average method.
    See: P. Izmailov, D. Podoprikhin, T. Garipov, D. Vetrov, and A. G. Wilson,
    “Averaging weights leads to wider optima and better generalization,”, 2018.
    """

    def _get_debiased_average(self, average: T.Variable):
        return average

    def _update_single(self, average: T.Variable, weight: Union[T.Tensor, T.Variable]):
        diff = (weight - average) / (1. + self.num_updates)
        T.assign_add(average, diff)


StochasticWeightAveraging = WeightMeanAveraging


class WeightMovingAveraging(WeightAveraging):
    """
    Weight averaging by exponential moving average method.

    This is usually known as a variant of Polyak weight averaging method.
    """

    decay: float
    zero_debias: bool

    def __init__(self,
                 weights: Iterable[T.Variable],
                 decay: float,
                 zero_debias: bool = True,
                 enabled: bool = True):
        super().__init__(weights=weights, enabled=enabled)
        self.decay = decay
        self.zero_debias = zero_debias

    def get_state_dict(self) -> Dict[str, Any]:
        ret = super().get_state_dict()
        ret['decay'] = self.decay
        ret['zero_debias'] = self.zero_debias
        return ret

    def set_state_dict(self, state: Dict[str, Any]):
        self.decay = float(state['decay'])
        self.zero_debias = bool(state['zero_debias'])
        super().set_state_dict(state)

    def _get_debiased_average(self, average: T.Variable):
        if self.zero_debias and self.num_updates > 0:
            return average / (1. - self.decay ** self.num_updates)
        else:
            return average

    def _update_single(self, average: T.Variable, weight: Union[T.Tensor, T.Variable]):
        T.assign_add(average, (weight - average) * (1. - self.decay))
