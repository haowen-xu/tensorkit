from typing import *

import mltk
import numpy as np
from mltk import ArrayTuple

from .. import tensor as T

__all__ = [
    'TensorStream',
    'as_tensor_stream',
]


class TensorStream(mltk.DataStream):
    """
    A subclass of :class:`mltk.DataStream` that transforms the underlying
    NumPy array data stream into tensor data stream.
    """

    source: mltk.DataStream
    device: str

    def __init__(self, source: mltk.DataStream, device: Optional[str] = None):
        """
        Construct a new :class:`TensorStream`.

        Args:
            source: The source data stream.
            device: The device where to place new tensors.
        """
        device = device or T.current_device()
        super().__init__(
            batch_size=source.batch_size,
            array_count=source.array_count,
            data_shapes=source.data_shapes,
            data_length=source.data_length,
            random_state=source.random_state,
        )
        self.source = source
        self.device = device

    def copy(self, **kwargs):
        kwargs.setdefault('device', self.device)
        return TensorStream(source=self.source, **kwargs)

    def _minibatch_iterator(self) -> Generator[ArrayTuple, None, None]:
        g = iter(self.source)
        try:
            for batch_data in g:
                with T.no_grad():
                    batch_data = tuple(
                        T.as_tensor(np.copy(arr), device=self.device)
                        for arr in batch_data
                    )
                yield batch_data
        finally:
            g.close()


def as_tensor_stream(source: mltk.DataStream,
                     device: Optional[str] = None,
                     prefetch: Optional[int] = None
                     ) -> Union[TensorStream, mltk.data.ThreadingDataStream]:
    """
    Construct a tensor data stream.

    Args:
        source: The source NumPy array stream.
        device: The device where to place new tensors.
        prefetch: Number of batches to prefetch in background.
            If specified, will wrap the constructed :class:`TensorStream`
            with a :class:`mltk.data.ThreadingDataStream`.

    Returns:
        The tensor data stream.
    """
    stream = TensorStream(source, device=device)
    if prefetch is not None:
        stream = stream.threaded(prefetch)
    return stream
