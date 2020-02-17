from typing import *

import mltk
from mltk import ArrayTuple

from .. import tensor as T

__all__ = [
    'TensorStream',
    'as_tensor_stream',
]


class TensorStream(mltk.DataStream):

    source: mltk.DataStream
    device: str

    def __init__(self, source: mltk.DataStream, device: Optional[str] = None):
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
                        T.from_numpy(arr, device=self.device)
                        for arr in batch_data
                    )
                    yield batch_data
        finally:
            g.close()

    def _concat_arrays(self, arrays: Sequence[T.Tensor]) -> T.Tensor:
        return T.concat(list(arrays), axis=0)


def as_tensor_stream(source: mltk.DataStream,
                     device: Optional[str] = None,
                     prefetch: Optional[int] = None
                     ) -> mltk.DataStream:
    stream = TensorStream(source, device=device)
    if prefetch is not None:
        stream = stream.threaded(prefetch)
    return stream
