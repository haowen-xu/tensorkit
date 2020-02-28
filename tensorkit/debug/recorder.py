from contextlib import contextmanager
from typing import *

import mltk
import numpy as np

from .. import tensor as T
from ..flows import *
from ..layers import *

__all__ = [
    'Recorder', 'ConcatRecorder', 'StatsRecorder',
    'RecorderFactory',
    'RecorderManager', 'has_recorder_manager', 'record_tensor',
    'LayerRecorder', 'FlowRecorder', 'with_recorder',
]


class Recorder(object):
    """Base class for tensor recorder."""

    def clear(self):
        raise NotImplementedError()

    def record(self, tensor: T.Tensor):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()


class NullRecorder(Recorder):
    """
    Tensor recorder that does nothing with the collected tensors.
    Typically used as the `exclude` rules on tensors.
    """

    def clear(self):
        pass

    def record(self, tensor: T.Tensor):
        pass

    def get(self):
        return None


class ConcatRecorder(Recorder):
    """Tensor recorder that concatenates the collected tensors."""

    def __init__(self):
        self.tensors = []

    def clear(self):
        self.tensors.clear()

    def record(self, tensor: T.Tensor):
        self.tensors.append(T.to_numpy(tensor))

    def get(self):
        if self.tensors:
            return np.concatenate(self.tensors, axis=0)


class StatsRecorder(Recorder):
    """Tensor recorder that computes statistics of the collected tensors."""

    metrics_Recorder: Union[mltk.MetricCollector, mltk.GeneralMetricCollector]

    def __init__(self, shape: Sequence[int] = ()):
        shape = list(shape)
        if not shape:
            self.metrics_Recorder = mltk.ScalarMetricCollector()
        else:
            self.metrics_Recorder = mltk.GeneralMetricCollector(shape)

    def clear(self):
        self.metrics_Recorder.reset()

    def record(self, tensor: T.Tensor):
        self.metrics_Recorder.update(T.to_numpy(tensor))

    def get(self):
        return self.metrics_Recorder.stats


RecorderFactory = Callable[[], Recorder]


class RecorderManager(object):
    """The manager of tensor recorders, indexed by names."""

    _factories: List[Tuple[Union[str, mltk.PatternType], RecorderFactory]]
    _default_factory: RecorderFactory
    _recorders: Dict[str, Recorder]

    def __init__(self, factories: Union[
                     Sequence[Tuple[Union[str, mltk.PatternType], RecorderFactory]],
                     Dict[str, RecorderFactory],
                 ] = (),
                 default_factory: RecorderFactory = NullRecorder):
        """
        Construct a new :class:`RecorderManager` instance.

        Args:
            factories: List of `(matcher, factory)`.  `matcher` should be
                a str or a regex pattern, which matches the tensor names.
                And if one matcher matches a given tensor name, then `factory`
                is used to construct the tensor recorder.
        """
        if hasattr(factories, 'items'):
            factories = list(factories.items())
        else:
            factories = list(factories)
        self._factories = factories
        self._default_factory = default_factory
        self._recorders = {}

    def add_factory(self,
                    name: Union[str, mltk.PatternType],
                    factory: RecorderFactory):
        """
        Add a new factory.

        Args:
            name: A str or a regex pattern, which matches the tensor names.
            factory: A factory, that constructs the tensor recorder.
        """
        self._factories.append((name, factory))

    def get_recorder(self, name: str) -> Recorder:
        """
        Get a tensor recorder according to the name.

        Args:
            name: Name of the tensor to be recorded.

        Returns:
            The tensor recorder.
        """
        if name not in self._recorders:
            factory = self._default_factory
            for n, f in self._factories:
                if (isinstance(n, str) and n == name) or \
                        (hasattr(n, 'match') and n.match(name)):
                    factory = f
                    break
            self._recorders[name] = factory()
        return self._recorders[name]

    def clear(self):
        """Clear all the recorded tensor information."""
        for c in self._recorders.values():
            c.clear()

    def record(self, name: str, tensor: T.Tensor):
        """
        Record the information of a tensor.

        Args:
            name: Name of the tensor.
            tensor: The tensor.
        """
        self.get_recorder(name).record(tensor)

    def get(self, name: str):
        """
        Get the recorded information of a tensor.

        Args:
            name: Name of the tensor.

        Returns:
            The information, or None if no information has been collected.
        """
        if name in self._recorders:
            return self.get_recorder(name).get()

    def iter_all(self, filter: Optional[Callable[[str], bool]] = None
                 ) -> Iterator[Tuple[str, Any]]:
        """
        Iterate the information of all recorded tensors.

        Args:
            filter: A callable function, which filters the name
                of the tensors.  A tensor and its recorded information
                will be yielded only when `filter` does not return False
                for its name.

        Yields:
            `(name, info)` of the recorded tensors.
        """
        for name, recorder in self._recorders.items():
            val = recorder.get()
            if val is not None and (filter is None or filter(name)):
                yield name, val

    def get_all(self, filter: Optional[Callable[[str], bool]] = None
                ) -> Dict[str, Any]:
        """
        Get the recorded objects.

        Args:
            filter: A callable function, which filters the name
                of the tensors.  A tensor and its recorded information
                will be yielded only when `filter` does not return False
                for its name.
        """
        return {key: val for key, val in self.iter_all(filter)}

    @contextmanager
    def push_to_stack(self):
        """
        Push this object to the global recorders stack within a context.

        When recording a tensor via :func:`record_tensor()`, the tensor
        will be recorded by all :class:`RecorderManager` on the stack.
        """
        _recorders_stack.push(self)
        try:
            yield self
        finally:
            _recorders_stack.pop()


_recorders_stack: mltk.utils.ContextStack[RecorderManager] = mltk.utils.ContextStack()


def has_recorder_manager() -> bool:
    """
    Whether or not there is at least one :class:`RecorderManager` on the stack.
    """
    return _recorders_stack.top() is not None


def record_tensor(name: str, tensor: T.Tensor):
    """
    Record a tensor by all :class:`RecorderManager` instances on the stack.

    Args:
        name: Name of the tensor.
        tensor: The tensor to be recorded.
    """
    for r in _recorders_stack.items:
        r.record(name, tensor)


class LayerRecorder(BaseLayer):
    """
    Class that wraps a layer and records its output.

    If there is only one output, it will be recorded as `name + ".output"`.
    Otherwise the outputs will be recorded as `name + ".output" + index`.
    """

    name: str
    wrapped: T.Module

    def __init__(self, wrapped: T.Module, name: str):
        """
        Construct a new :class:`LayerRecorder`.

        Args:
            wrapped: The wrapped layer.
            name: The name prefix of the recorded tensor.
        """
        super().__init__()
        self.name = name
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        outputs = self.wrapped(*args, **kwargs)

        if has_recorder_manager():
            if isinstance(outputs, (list, tuple)):
                for i, t in enumerate(outputs):
                    if isinstance(t, T.Tensor):
                        record_tensor(f'{self.name}.output.{i}', t)
            elif isinstance(outputs, T.Tensor):
                record_tensor(f'{self.name}.output', outputs)

        return outputs


class FlowRecorder(Flow):
    """
    Class that wraps a flow and records its output.

    If `inverse = False`, the output will be recorded as `name + ".output"`,
    while the output log-det will be recorded as `name + ".log_det"`.
    Otherwise if `inverse = True`,  the output will be recorded as
    `name + ".inv_output"`, while the output log-det will be recorded as
    `name + ".inv_log_det"`.
    """

    name: str
    wrapped: Flow

    def __init__(self, wrapped: Flow, name: str):
        """
        Construct a new :class:`LayerRecorder`.

        Args:
            wrapped: The wrapped flow.
            name: The name prefix of the recorded tensor.
        """

        super().__init__(
            x_event_ndims=wrapped.get_x_event_ndims(),
            y_event_ndims=wrapped.get_y_event_ndims(),
            explicitly_invertible=wrapped.is_explicitly_invertible(),
        )
        self.name = name
        self.wrapped = wrapped

    def _transform(self,
                   input: T.Tensor,
                   input_log_det: Optional[T.Tensor],
                   inverse: bool,
                   compute_log_det: bool
                   ) -> Tuple[T.Tensor, Optional[T.Tensor]]:
        output, output_log_det = self.wrapped(
            input, input_log_det, inverse, compute_log_det)

        if has_recorder_manager():
            pfx = 'inv_' if inverse else ''
            record_tensor(f'{self.name}.{pfx}output', output)
            if output_log_det is not None:
                record_tensor(f'{self.name}.{pfx}output_log_det', output_log_det)
                log_det = output_log_det
                if input_log_det is not None:
                    log_det = log_det - input_log_det
                record_tensor(f'{self.name}.{pfx}log_det', log_det)

        return output, output_log_det


def with_recorder(module: T.Module, name: str):
    """
    Wrap `module` with a suitable recorder.

    Args:
        module: The module to be recorded.
        name: The name prefix of the recorded tensors.

    Returns:
        The wrapped module.
    """
    if isinstance(module, Flow):
        return FlowRecorder(module, name)
    else:
        return LayerRecorder(module, name)
