from typing import *

import numpy as np

__all__ = [
    'SamplerMapper',
    'BaseSampler', 'BernoulliSampler', 'UniformNoiseSampler'
]


class SamplerMapper(object):
    """Wrap a :class:`BaseSampler` as a data stream mapper."""

    sampler: 'BaseSampler'

    def __init__(self, sampler: 'BaseSampler'):
        self.sampler = sampler

    def __call__(self, x: np.ndarray, *args):
        return (self.sampler.sample(x),) + args


class BaseSampler(object):
    """Base class for samplers."""

    def sample(self, x: np.ndarray) -> np.ndarray:
        """
        Sample array according to `x`.

        Args:
            x: The input `x` array.

        Returns:
            The sampled array.
        """
        raise NotImplementedError()

    def as_mapper(self) -> SamplerMapper:
        return SamplerMapper(self)


class BernoulliSampler(BaseSampler):
    """
    A :class:`DataMapper` which can sample 0/1 integers according to the
    input probability.  The input is assumed to be float numbers range within
    [0, 1) or [0, 1].
    """

    dtype: np.dtype
    random_state: np.random.RandomState

    def __init__(self,
                 dtype: Union[np.dtype, str] = np.int32,
                 random_state: Optional[np.random.RandomState] = None):
        """
        Construct a new :class:`BernoulliSampler`.

        Args:
            dtype: The data type of the sampled array.  Default `np.int32`.
            random_state: Optional numpy RandomState for sampling.
                (default :obj:`None`, construct a new :class:`RandomState`).
        """
        self.dtype = np.dtype(dtype)
        self.random_state = \
            random_state or np.random.RandomState(np.random.randint(0x7fffffff))

    def sample(self, x):
        sampled = np.asarray(
            self.random_state.uniform(0., 1., size=x.shape) < x,
            dtype=self.dtype
        )
        return sampled


class UniformNoiseSampler(BaseSampler):
    """
    A :class:`DataMapper` which can add uniform noise onto the input array.
    The data type of the returned array will be the same as the input array,
    unless `dtype` is specified at construction.
    """

    min_val: float
    max_val: float
    dtype: Optional[np.dtype]
    random_state: np.random.RandomState

    def __init__(self,
                 min_val: float = 0.,
                 max_val: float = 1.,
                 dtype: Optional[Union[np.dtype, str]] = None,
                 random_state: Optional[np.random.RandomState] = None):
        """
        Construct a new :class:`UniformNoiseSampler`.

        Args:
            min_val: The lower bound of the uniform noise (included).
            max_val: The upper bound of the uniform noise (excluded).
            dtype: The data type of the sampled array.  Default `np.int32`.
            random_state: Optional numpy RandomState for sampling.
                (default :obj:`None`, construct a new :class:`RandomState`).
        """
        self.min_val = min_val
        self.max_val = max_val
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.random_state = \
            random_state or np.random.RandomState(np.random.randint(0x7fffffff))

    def sample(self, x):
        dtype = self.dtype or x.dtype
        noise = self.random_state.uniform(self.min_val, self.max_val, size=x.shape)
        return np.asarray(x + noise, dtype=dtype)
