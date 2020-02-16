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

    def __init__(self, source: mltk.DataStream):
        super().__init__(
            batch_size=source.batch_size,
            array_count=source.array_count,
            data_shapes=source.data_shapes,
            data_length=source.data_length,
            random_state=source.random_state,
        )
        self.source = source

    def copy(self, **kwargs):
        return TensorStream(source=self.source, **kwargs)

    def _minibatch_iterator(self) -> Generator[ArrayTuple, None, None]:
        g = iter(self.source)
        try:
            for batch_data in g:
                with T.no_grad():
                    batch_data = tuple(T.from_numpy(arr) for arr in batch_data)
                    yield batch_data
        finally:
            g.close()


def as_tensor_stream(source: mltk.DataStream,
                     prefetch: Optional[int] = None
                     ) -> mltk.DataStream:
    stream = TensorStream(source)
    if prefetch is not None:
        stream = stream.threaded(prefetch)
    return stream
