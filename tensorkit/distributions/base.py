from typing import *

from .. import tensor as T
from ..settings_ import settings

__all__ = ['Distribution']


class Distribution(object):

    def __init__(self,
                 dtype: T.DTypeLike,
                 is_continuous: bool,
                 is_reparameterized: bool,
                 value_shape: T.ShapeArgType,
                 event_ndims: int,
                 min_event_ndims: int,
                 check_numerics: Optional[bool],
                 random_state: Optional[T.random.RandomState]):
        # validate the arguments
        value_shape = T.as_shape(value_shape)
        event_ndims = int(event_ndims)
        min_event_ndims = int(min_event_ndims)

        if event_ndims < min_event_ndims:
            raise ValueError(f'`event_ndims < {min_event_ndims}` is invalid: '
                             f'got {event_ndims}')
        if event_ndims > len(value_shape):
            raise ValueError(f'`event_ndims > {len(value_shape)}` is invalid: '
                             f'got {event_ndims}, value shape is {value_shape}')

        if event_ndims > 0:
            batch_shape = value_shape[:-event_ndims]
            event_shape = value_shape[-event_ndims:]
        else:
            batch_shape = value_shape
            event_shape = ()

        if check_numerics is not None:
            check_numerics = bool(check_numerics)
        else:
            check_numerics = settings.check_numerics

        # construct the object
        self._dtype = T.as_dtype(dtype)
        self._is_continuous = bool(is_continuous)
        self._is_reparamaterized = bool(is_reparameterized)
        self._batch_shape = batch_shape
        self._event_shape = event_shape
        self._min_event_ndims = min_event_ndims
        self._check_numerics = check_numerics
        self._random_state = random_state

    @property
    def dtype(self) -> T.DType:
        return self._dtype

    @property
    def is_continuous(self) -> bool:
        return self._is_continuous

    @property
    def is_reparamterized(self) -> bool:
        return self._is_reparamaterized

    @property
    def batch_shape(self) -> T.ShapeTuple:
        return self._batch_shape

    @property
    def event_shape(self) -> T.ShapeTuple:
        return self._event_shape

    @property
    def event_ndims(self) -> int:
        return len(self._event_shape)

    @property
    def min_event_ndims(self) -> int:
        return self._min_event_ndims

    @property
    def random_state(self) -> Optional[T.random.RandomState]:
        return self._random_state

    def sample(self,
               n_samples: Optional[int] = None,
               group_ndims: int = 0,
               is_reparameterized: Optional[bool] = None,
               compute_prob: Optional[bool] = None
               ) -> 'StochasticTensor':
        raise NotImplementedError()

    def log_prob(self, given: T.TensorLike, group_ndims: int = 0) -> T.Tensor:
        raise NotImplementedError()

    def prob(self, given: T.TensorLike, group_ndims: int = 0) -> T.Tensor:
        return T.exp(self.log_prob(given=given, group_ndims=group_ndims))

    def _maybe_check_numerics(self, name: str, tensor: T.Tensor) -> T.Tensor:
        if self._check_numerics:
            pass  # TODO: check numerics
        return tensor

    def _add_to_event_ndims(self, group_ndims: int) -> int:
        event_ndims = self.event_ndims + group_ndims
        if event_ndims < self.min_event_ndims:
            raise ValueError('Invalid `group_ndims`: group_ndims + '
                             'self.event_ndims < self.min_event_ndims')
        return event_ndims

    def _check_reparameterized(self, is_reparamaterized: Optional[bool]):
        if is_reparamaterized and not self.is_reparamterized:
            raise ValueError(
                f'Distribution {self!r} is not re-parameterized, thus cannot '
                f'generate re-paramterized samples.')

        if is_reparamaterized is None:
            is_reparamaterized = self.is_reparamterized
        return is_reparamaterized


# back reference to the StochasticTensor
from ..stochastic import StochasticTensor
