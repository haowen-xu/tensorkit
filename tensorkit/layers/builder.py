from contextlib import contextmanager
from typing import *

from mltk.utils import NOT_SET, ContextStack

from .activation import *
from .composed import *
from .core import *
from .pool import *
from .resnet import *
from .shape_ import *
from .. import tensor as T
from ..arg_check import *
from ..typing_ import *

__all__ = ['LayerArgs', 'get_default_layer_args', 'SequentialBuilder']


def _get_layer_class(name: str, support_wildcard: bool = False) -> Optional[type]:
    if support_wildcard and name == '*':
        return None

    if not _cached_layer_class_names_map:
        # map the standard names of the layers to the layer classes
        import tensorkit as tk
        for attr in dir(tk.layers):
            val = getattr(tk.layers, attr)
            if isinstance(val, type) and issubclass(val, T.Module):
                _cached_layer_class_names_map[attr.lower()] = val

        # aliases to XXXTransposeNd
        for spatial_ndims in (1, 2, 3):
            for prefix in ('LinearConv', 'Conv'):
                # the original name and the layer class
                orig_name = f'{prefix}Transpose{spatial_ndims}d'
                layer_cls = getattr(tk.layers, orig_name)

                # the new name
                alias_name = orig_name
                alias_name = alias_name.replace('ConvTranspose', 'DeConv')
                _cached_layer_class_names_map[alias_name.lower()] = layer_cls

    canonical_name = name.lower().replace('_', '')
    if canonical_name not in _cached_layer_class_names_map:
        raise ValueError(f'Unsupported layer class: {name!r}.')
    return _cached_layer_class_names_map[canonical_name]


_cached_layer_class_names_map = {}


def _calculate_conv_output_size(in_size, kernel_size, stride, padding, dilation):
    out_size = []
    for i, k, s, p, d in zip(in_size, kernel_size, stride, padding, dilation):
        if i is None:
            out_size.append(None)
        else:
            l = T.utils.calculate_conv_output_size([i], [k], [s], [p], [d])[0]
            out_size.append(l)
    return out_size


def _calculate_deconv_output_size(in_size, kernel_size, stride, padding, output_padding, dilation):
    out_size = []
    for i, k, s, p, op, d in zip(in_size, kernel_size, stride, padding, output_padding, dilation):
        if i is None:
            out_size.append(None)
        else:
            l = T.utils.calculate_deconv_output_size([i], [k], [s], [p], [op], [d])[0]
            out_size.append(l)
    return out_size


if T.IS_CHANNEL_LAST:
    def _split_channel_spatial(shape):
        return shape[-1], shape[:-1]


    def _unsplit_channel_spatial(channel, spatial):
        return list(spatial) + [channel]

else:
    def _split_channel_spatial(shape):
        return shape[0], shape[1:]


    def _unsplit_channel_spatial(channel, spatial):
        return [channel] + list(spatial)


class LayerArgs(object):
    """A class that manages the default arguments for constructing layers."""

    # type? => {arg_name: arg_val}.
    # None type indicates arguments for all types.
    args: Dict[Optional[type], Dict[str, Any]]

    def __init__(self, layer_args: Optional['LayerArgs'] = NOT_SET):
        """
        Construct a new :class:`LayerArgs` instance.

        Args:
            layer_args: Clone from this :class:`LayerArgs` instance.
        """
        if layer_args is NOT_SET:
            layer_args = get_default_layer_args()

        if layer_args is None:
            self.args = {}
        else:
            self.args = {type_: {key: val for key, val in type_args.items()}
                         for type_, type_args in layer_args.args.items()}

    @contextmanager
    def as_default(self) -> ContextManager['LayerArgs']:
        """Push this `LayerArgs` instance as the default."""
        try:
            _layer_args_stack.push(self)
            yield self
        finally:
            _layer_args_stack.pop()

    def copy(self) -> 'LayerArgs':
        """
        Copy a new `LayerArgs` instance.

        Returns:
            A new :class:`LayerArgs` instance.
        """
        return LayerArgs(self)

    def set_args(self,
                 type_or_types_: Union[
                     str, Type[T.Module], Sequence[Union[str, Type[T.Module]]]],
                 layer_args_: Optional[Sequence[str]] = NOT_SET,
                 **kwargs) -> 'LayerArgs':
        """
        Set default arguments for the specified layer types.

        Args:
            type_or_types_: The layer type or types.
            layer_args_: If specified, override the `__layer_args__` of the layer.
            **kwargs: The default arguments to be set.

        Returns:
            This :class:`LayerArgs` instance.
        """
        if isinstance(type_or_types_, (str, type)):
            type_or_types_ = [type_or_types_]

        for type_ in type_or_types_:
            if isinstance(type_, str):
                type_ = _get_layer_class(type_, support_wildcard=True)
            if type_ not in self.args:
                self.args[type_] = {}

            # validate the arguments
            if type_ is not None:
                if layer_args_ is NOT_SET:
                    layer_args_ = getattr(type_, '__layer_args__', None)
                layer_has_kwargs = getattr(type_, '__layer_has_kwargs__', False)
                if layer_args_ is not None and not layer_has_kwargs:
                    for k in kwargs:
                        if k not in layer_args_:
                            raise ValueError(
                                f'The constructor of {type_!r} does not have '
                                f'the specified keyword argument: {k}'
                            )

            # update the arguments
            self.args[type_].update(kwargs)

        return self

    def get_kwargs(self, type_: Union[str, type], **kwargs) -> Dict[str, Any]:
        """
        Get the merged keyword arguments for the specified layer type.

        Args:
            type_: The layer type.
            **kwargs: The overrided keyword arguments.

        Returns:
            The merged keyword arguments.
        """
        if isinstance(type_, str):
            type_ = _get_layer_class(type_)

        # get the arguments for this type
        args = self.args.get(type_)
        if args:
            for key, val in args.items():
                kwargs.setdefault(key, val)

        # get the arguments for all types
        layer_args = getattr(type_, '__layer_args__', None)
        if layer_args:  # only use known args
            args = self.args.get(None)
            if args:
                for key, val in args.items():
                    if key in layer_args:
                        kwargs.setdefault(key, val)

        return kwargs

    def build(self, type_: Union[str, type], *args, **kwargs):
        """
        Build the layer with default arguments.

        Args:
            type_: The layer type.
            *args: The positional arguments.
            **kwargs: The named arguments, which may override the default
                arguments.

        Returns:
            The built layer object.
        """
        if isinstance(type_, str):
            type_ = _get_layer_class(type_)
        return type_(*args, **self.get_kwargs(type_, **kwargs))


def get_default_layer_args() -> LayerArgs:
    """Get the global default `LayerArgs` instance."""
    return _layer_args_stack.top()


_layer_args_stack = ContextStack[LayerArgs](lambda: LayerArgs(None))


class SequentialBuilder(object):
    """A class that helps to build a sequence layers."""

    in_shape: List[Optional[int]]
    out_shape: List[Optional[int]]
    layer_args: LayerArgs
    layers: List[T.Module]

    def __init__(self,
                 in_spec: Union[
                     Optional[int],
                     Sequence[Optional[int]],
                     'SequentialBuilder'] = NOT_SET,
                 *,
                 in_shape: Sequence[Optional[int]] = NOT_SET,
                 in_channels: Optional[int] = NOT_SET,
                 in_size: Sequence[Optional[int]] = NOT_SET,
                 in_builder: 'SequentialBuilder' = NOT_SET,
                 layer_args: LayerArgs = NOT_SET):
        """
        Construct a new :class:`SequentialBuilder`.

        Args:
            in_spec: Positional argument, maybe the input shape, the number
                of input channels, or another instance of `SequentialBuilder`,
                whose layer arguments will be cloned and `out_shape` will be
                used as the `in_shape` of this :class:`SequentialBuilder`.
            in_shape: The input shape.
            in_channels: The number of input channels.
            in_size: The input spatial size.  Can be specified
                only if `in_channels` is specified, or `in_spec` is a int.
            in_builder: Explicitly specify the previous sequential builder.
                The `layer_args` of this `in_builder` will be copied to the
                new sequential builder, if `layer_args` is not specified.
            layer_args: If specified, the layer arguments will be copied
                to the new sequential builder.  This will also override
                the layer args of `in_builder`.
        """
        # parse the argument
        if int(in_spec is not NOT_SET) + int(in_shape is not NOT_SET) + \
                int(in_channels is not NOT_SET) + int(in_builder is not NOT_SET) != 1:
            raise ValueError(
                'One and only one of `in_spec`, `in_shape`, `in_channels` and '
                '`in_builder` should be specified.'
            )

        if layer_args is not NOT_SET:
            layer_args = LayerArgs(layer_args)

        if isinstance(in_spec, SequentialBuilder):
            in_builder = in_spec
            if layer_args is NOT_SET:
                layer_args = LayerArgs(in_builder.layer_args)
        elif hasattr(in_spec, '__iter__'):
            in_shape = in_spec
        elif in_spec is not NOT_SET:
            in_channels = in_spec

        if layer_args is NOT_SET:
            layer_args = LayerArgs()

        if in_size is not NOT_SET and in_channels is NOT_SET:
            raise ValueError(
                '`in_size` can be specified only when `in_channels` '
                'is specified, or `in_spec` is None or an integer.'
            )

        if in_shape is not NOT_SET:
            in_shape = list(in_shape)
        elif in_channels is not NOT_SET:
            if in_size is NOT_SET:
                in_size = []
            in_shape = _unsplit_channel_spatial(in_channels, in_size)
        else:
            in_shape = list(in_builder.out_shape)

        # create the object
        self.in_shape = in_shape
        self.out_shape = in_shape
        self.layer_args = layer_args
        self.layers = []

    def _assert_out_shape(self,
                          shape: Optional[Sequence[bool]] = None,
                          channel: Optional[bool] = None,
                          spatial: Optional[Sequence[bool]] = None,
                          at_least: bool = False) -> List[Optional[int]]:
        if shape is None:
            if channel is None:  # pragma: no cover
                raise ValueError('`channel` must be specified when `shape` is not.')
            shape = _unsplit_channel_spatial(channel, spatial or [])

        ndims = len(shape)
        if at_least:
            if len(self.out_shape) < ndims:
                raise ValueError(
                    f'The previous output shape is expected to be '
                    f'at least {ndims}d: got output shape {self.out_shape}.'
                )
        else:
            if len(self.out_shape) != ndims:
                raise ValueError(
                    f'The previous output shape is expected to be '
                    f'exactly {ndims}d: got output shape {self.out_shape}.'
                )

        for i, (d, s) in enumerate(
                zip(shape[::-1], self.out_shape[::-1]), 1):
            if d and s is None:
                raise ValueError(
                    f'Axis {-i} of the previous output shape is expected '
                    f'to be deterministic: got output shape {self.out_shape}.'
                )

        return self.out_shape

    def _split_out_shape(self,
                         channel: Optional[bool] = None,
                         spatial: Optional[Sequence[bool]] = None
                         ) -> Tuple[Optional[int], List[Optional[int]]]:
        out_shape = self._assert_out_shape(channel=channel, spatial=spatial)
        return _split_channel_spatial(out_shape)

    def set_args(self,
                 type_or_types_: Union[str, type, Sequence[Union[str, type]]],
                 **kwargs) -> 'SequentialBuilder':
        """
        Set layer default arguments.

        Args:
            type_or_types_: The layer type or types.
            **kwargs: The default arguments.

        Returns:
            This sequential builder object.
        """
        if isinstance(type_or_types_, (str, type)):
            type_or_types_ = [type_or_types_]

        for type_ in type_or_types_:
            if type_ == '*':
                self.layer_args.set_args(type_, **kwargs)
            else:
                if isinstance(type_, str):
                    type_ = _get_layer_class(type_)

                layer_args_ = getattr(type_, '__layer_args__', None)
                if layer_args_ and 'output_padding' in layer_args_:
                    # suggest it's a deconv layer, add 'output_size' to the valid args list
                    layer_args_ = list(layer_args_) + ['output_size']

                self.layer_args.set_args(
                    type_or_types_=type_,
                    layer_args_=layer_args_,
                    **kwargs
                )

        return self

    @contextmanager
    def arg_scope(self,
                  type_or_types_: Union[str, type, Sequence[Union[str, type]]],
                  **kwargs) -> Generator['SequentialBuilder', None, None]:
        """
        Set layer default arguments within a scope, which will be restore to
        the previous values after exiting the scope.

        Args:
            type_or_types_: The layer type or types.
            **kwargs: The default arguments.

        Yields:
            This builder object itself.
        """
        old_layer_args = self.layer_args.copy()
        self.set_args(type_or_types_, **kwargs)
        try:
            yield self
        finally:
            self.layer_args = old_layer_args

    def add(self,
            layer: T.Module,
            out_shape: List[Optional[int]] = NOT_SET,
            *,
            out_channels: Optional[int] = NOT_SET,
            out_size: List[Optional[int]] = NOT_SET
            ) -> 'SequentialBuilder':
        """
        Manually add a layer to this builder.

        Args:
            layer: The layer to be added.
            out_shape: The new output shape.
            out_channels: The new output channels.  Should be specified and
                only be specified when `out_shape` is not.
            out_size: The new spatial shape.  Should only be specified
                when `out_channels` is specified.

        Returns:
            This sequential builder object.
        """
        if out_size is not NOT_SET and out_channels is NOT_SET:
            raise ValueError('`out_size` can only be specified when '
                             '`out_channels` is specified.')
        if (out_shape is NOT_SET) == (out_channels is NOT_SET):
            raise ValueError('Either `out_shape` or `out_channels` should be '
                             'specified, but not both.')

        if out_channels is not NOT_SET:
            if out_size is NOT_SET:
                out_size = []
            out_shape = _unsplit_channel_spatial(out_channels, out_size)

        self.layers.append(layer)
        self.out_shape = out_shape
        return self

    def build(self, flatten_to_ndims: bool = False) -> T.Module:
        """
        Build the sequential layer.

        Args:
            flatten_to_ndims: Whether or not to wrap the sequential layer
                with a :class:`FlattenToNDims` layer?

        Returns:
            The built sequential layer.
        """
        if not self.layers:
            return Identity()
        elif len(self.layers) == 1:
            layer = self.layers[0]
        else:
            layer = Sequential(self.layers)

        if flatten_to_ndims:
            layer = FlattenToNDims(layer, ndims=len(self.in_shape) + 1)
        return layer

    def as_input(self, layer_args: LayerArgs = NOT_SET) -> 'SequentialBuilder':
        """
        Construct a new :class:`SequentialBuilder` whose `in_shape` is the
        `out_shape` of this builder.

        Args:
            layer_args: If specified, override the `layer_args` of this builder.

        Returns:
            The new :class:`SequentialBuilder`.
        """
        return SequentialBuilder(self, layer_args=layer_args)

    # ---- identity layer (add no layer) ----
    def identity(self):
        return self

    # ---- activation ----
    def _make_activation(self, type_, **kwargs):
        self._assert_out_shape((False,), at_least=True)
        layer = self.layer_args.build(type_, **kwargs)
        return self.add(layer, self.out_shape)

    def relu(self):
        return self._make_activation(ReLU)

    def leaky_relu(self):
        return self._make_activation(LeakyReLU)

    def sigmoid(self):
        return self._make_activation(Sigmoid)

    def tanh(self):
        return self._make_activation(Tanh)

    def hard_tanh(self, min_val: float = -1., max_val: float = 1.):
        return self._make_activation(HardTanh, min_val=min_val, max_val=max_val)

    def log_softmax(self):
        return self._make_activation(LogSoftmax)

    # ---- fully-connected layers ----
    def _fully_connected(self, layer_cls, out_features, **kwargs):
        in_features, _ = self._split_out_shape(True)
        layer = self.layer_args.build(layer_cls, in_features, out_features, **kwargs)
        return self.add(layer, [out_features])

    def linear(self, out_features: int, **kwargs):
        return self._fully_connected(Linear, out_features, **kwargs)

    def dense(self, out_features: int, **kwargs):
        return self._fully_connected(Dense, out_features, **kwargs)

    # ---- convolution layers ----
    def _conv_nd(self, spatial_ndims, conv_cls, out_channels, **kwargs):
        kwargs = self.layer_args.get_kwargs(conv_cls, **kwargs)
        if 'kernel_size' not in kwargs:
            raise ValueError('The `kernel_size` argument is required.')

        in_channels, in_size = self._split_out_shape(True, [False] * spatial_ndims)

        # validate the arguments
        kernel_size = validate_conv_size('kernel_size', kwargs['kernel_size'], spatial_ndims)
        stride = validate_conv_size('stride', kwargs.get('stride', 1), spatial_ndims)
        dilation = validate_conv_size('dilation', kwargs.get('dilation', 1), spatial_ndims)
        padding = validate_padding(
            kwargs.get('padding', PaddingMode.DEFAULT), kernel_size, dilation, spatial_ndims)

        # calculate the output shape
        out_size = _calculate_conv_output_size(in_size, kernel_size, stride, padding, dilation)
        out_shape = _unsplit_channel_spatial(out_channels, out_size)

        # build the layer
        layer = conv_cls(in_channels, out_channels, **kwargs)
        return self.add(layer, out_shape)

    def linear_conv1d(self,
                      out_channels: int,
                      **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(1, LinearConv1d, out_channels, **kwargs)

    def linear_conv2d(self,
                      out_channels: int,
                      **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(2, LinearConv2d, out_channels, **kwargs)

    def linear_conv3d(self,
                      out_channels: int,
                      **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(3, LinearConv3d, out_channels, **kwargs)

    def conv1d(self,
               out_channels: int,
               **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(1, Conv1d, out_channels, **kwargs)

    def conv2d(self,
               out_channels: int,
               **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(2, Conv2d, out_channels, **kwargs)

    def conv3d(self,
               out_channels: int,
               **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(3, Conv3d, out_channels, **kwargs)

    def res_block1d(self,
                    out_channels: int,
                    **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(1, ResBlock1d, out_channels, **kwargs)

    def res_block2d(self,
                    out_channels: int,
                    **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(2, ResBlock2d, out_channels, **kwargs)

    def res_block3d(self,
                    out_channels: int,
                    **kwargs) -> 'SequentialBuilder':
        return self._conv_nd(3, ResBlock3d, out_channels, **kwargs)

    # ---- deconvolution layers ----
    def _deconv_nd(self, spatial_ndims, deconv_cls, out_channels, output_size, **kwargs):
        kwargs = self.layer_args.get_kwargs(deconv_cls, **kwargs)
        if 'kernel_size' not in kwargs:
            raise ValueError('The `kernel_size` argument is required.')

        if output_size is not NOT_SET:
            kwargs.pop('output_size', None)
        else:
            output_size = kwargs.pop('output_size', NOT_SET)
        in_channels, in_size = self._split_out_shape(True, [False] * spatial_ndims)

        # validate the arguments
        kernel_size = validate_conv_size('kernel_size', kwargs['kernel_size'], spatial_ndims)
        stride = validate_conv_size('stride', kwargs.get('stride', 1), spatial_ndims)
        dilation = validate_conv_size('dilation', kwargs.get('dilation', 1), spatial_ndims)
        padding = validate_padding(
            kwargs.get('padding', PaddingMode.DEFAULT), kernel_size, dilation, spatial_ndims)

        if 'output_padding' in kwargs and output_size is not NOT_SET:
            raise ValueError('`output_padding` and `output_size` cannot be both specified.')
        elif output_size is not NOT_SET:
            if len(output_size) != spatial_ndims:
                raise ValueError(
                    f'`output_size` is expected to be {spatial_ndims}d: '
                    f'got {output_size}.'
                )
            if any(i is None for i in in_size):
                raise ValueError(
                    f'Specifying `output_size` instead of `output_padding` '
                    f'is supported only when the previous output shape '
                    f'is all deterministic.'
                )
            output_padding = T.utils.calculate_deconv_output_padding(
                in_size, output_size, kernel_size, stride, padding, dilation)
            out_size = output_size
        elif 'output_padding' in kwargs:
            output_padding = validate_output_padding(
                kwargs.get('output_padding', 0), stride, dilation, spatial_ndims)
            out_size = None
        else:
            output_padding = [0] * spatial_ndims
            out_size = None

        # calculate the output shape if not specified
        if out_size is None:
            out_size = _calculate_deconv_output_size(
                in_size, kernel_size, stride, padding, output_padding, dilation)
        out_shape = _unsplit_channel_spatial(out_channels, out_size)

        # build the layer
        kwargs['output_padding'] = output_padding
        layer = deconv_cls(in_channels, out_channels, **kwargs)
        return self.add(layer, out_shape)

    def linear_conv_transpose1d(self,
                                out_channels: int,
                                output_size: List[int] = NOT_SET,
                                **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            1, LinearConvTranspose1d, out_channels, output_size=output_size, **kwargs)

    def linear_conv_transpose2d(self,
                                out_channels: int,
                                output_size: List[int] = NOT_SET,
                                **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            2, LinearConvTranspose2d, out_channels, output_size=output_size, **kwargs)

    def linear_conv_transpose3d(self,
                                out_channels: int,
                                output_size: List[int] = NOT_SET,
                                **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            3, LinearConvTranspose3d, out_channels, output_size=output_size, **kwargs)

    def conv_transpose1d(self,
                         out_channels: int,
                         output_size: List[int] = NOT_SET,
                         **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            1, ConvTranspose1d, out_channels, output_size=output_size, **kwargs)

    def conv_transpose2d(self,
                         out_channels: int,
                         output_size: List[int] = NOT_SET,
                         **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            2, ConvTranspose2d, out_channels, output_size=output_size, **kwargs)

    def conv_transpose3d(self,
                         out_channels: int,
                         output_size: List[int] = NOT_SET,
                         **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            3, ConvTranspose3d, out_channels, output_size=output_size, **kwargs)

    def res_block_transpose1d(self,
                              out_channels: int,
                              output_size: List[int] = NOT_SET,
                              **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            1, ResBlockTranspose1d, out_channels, output_size=output_size, **kwargs)

    def res_block_transpose2d(self,
                              out_channels: int,
                              output_size: List[int] = NOT_SET,
                              **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            2, ResBlockTranspose2d, out_channels, output_size=output_size, **kwargs)

    def res_block_transpose3d(self,
                              out_channels: int,
                              output_size: List[int] = NOT_SET,
                              **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            3, ResBlockTranspose3d, out_channels, output_size=output_size, **kwargs)

    # aliases for the deconvolution layers
    linear_deconv1d = linear_conv_transpose1d
    linear_deconv2d = linear_conv_transpose2d
    linear_deconv3d = linear_conv_transpose3d
    deconv1d = conv_transpose1d
    deconv2d = conv_transpose2d
    deconv3d = conv_transpose3d

    # ---- pool layers ----
    def _pool_nd(self, spatial_ndims, pool_cls, **kwargs):
        kwargs = self.layer_args.get_kwargs(pool_cls, **kwargs)
        if 'kernel_size' not in kwargs:
            raise ValueError('The `kernel_size` argument is required.')

        in_channels, in_size = self._split_out_shape(False, [False] * spatial_ndims)

        # validate the arguments
        kernel_size = validate_conv_size('kernel_size', kwargs['kernel_size'], spatial_ndims)
        stride = validate_conv_size('stride', kwargs.get('stride', kernel_size), spatial_ndims)
        dilation = [1] * spatial_ndims
        padding = validate_padding(kwargs.get('padding', PaddingMode.DEFAULT), kernel_size, dilation, spatial_ndims)

        # calculate the output shape
        out_size = _calculate_conv_output_size(in_size, kernel_size, stride, padding, dilation)
        out_shape = _unsplit_channel_spatial(in_channels, out_size)

        # build the layer
        layer = pool_cls(**kwargs)
        return self.add(layer, out_shape)

    def avg_pool1d(self, **kwargs) -> 'SequentialBuilder':
        return self._pool_nd(1, AvgPool1d, **kwargs)

    def avg_pool2d(self, **kwargs) -> 'SequentialBuilder':
        return self._pool_nd(2, AvgPool2d, **kwargs)

    def avg_pool3d(self, **kwargs) -> 'SequentialBuilder':
        return self._pool_nd(3, AvgPool3d, **kwargs)

    def max_pool1d(self, **kwargs) -> 'SequentialBuilder':
        return self._pool_nd(1, MaxPool1d, **kwargs)

    def max_pool2d(self, **kwargs) -> 'SequentialBuilder':
        return self._pool_nd(2, MaxPool2d, **kwargs)

    def max_pool3d(self, **kwargs) -> 'SequentialBuilder':
        return self._pool_nd(3, MaxPool3d, **kwargs)

    def _global_avg_pool_nd(self, spatial_ndims, pool_cls, **kwargs):
        kwargs = self.layer_args.get_kwargs(pool_cls, **kwargs)
        keepdims = kwargs.get('keepdims', False)

        in_channels, in_size = self._split_out_shape(False, [False] * spatial_ndims)
        if keepdims:
            out_shape = _unsplit_channel_spatial(in_channels, [1] * spatial_ndims)
        else:
            out_shape = [in_channels]
        layer = pool_cls(**kwargs)
        return self.add(layer, out_shape)

    def global_avg_pool1d(self, **kwargs) -> 'SequentialBuilder':
        return self._global_avg_pool_nd(1, GlobalAvgPool1d, **kwargs)

    def global_avg_pool2d(self, **kwargs) -> 'SequentialBuilder':
        return self._global_avg_pool_nd(2, GlobalAvgPool2d, **kwargs)

    def global_avg_pool3d(self, **kwargs) -> 'SequentialBuilder':
        return self._global_avg_pool_nd(3, GlobalAvgPool3d, **kwargs)

    # ---- reshape layers ----
    def reshape(self, shape: Sequence[int]) -> 'SequentialBuilder':
        # check the input and output shape
        in_shape = self.out_shape
        in_count = 1
        for s in in_shape:
            if s is None:
                in_count = None
                break
            else:
                in_count *= s

        shape = list(shape)
        out_neg_one_count = 0
        out_count = 1

        for s in shape:
            if s == -1:
                if out_neg_one_count > 0:
                    raise ValueError(f'Too many "-1" in `shape`: '
                                     f'got {shape!r}')
                else:
                    out_neg_one_count += 1
            elif s <= 0:
                raise ValueError(f'`shape` is not a valid shape: '
                                 f'{shape!r}')
            else:
                out_count *= s

        if in_count is not None:
            if (not out_neg_one_count and out_count != in_count) or \
                    (out_neg_one_count and in_count % out_count != 0):
                raise ValueError(f'The previous output shape cannot be reshape '
                                 f'into the new `shape`: {self.out_shape!r} vs '
                                 f'{shape!r}.')

            out_shape = [
                s if s != -1 else (in_count // out_count)
                for s in shape
            ]
        else:
            out_shape = [s if s != -1 else None for s in shape]

        return self.add(ReshapeTail(len(in_shape), shape), out_shape)

    def flatten(self) -> 'SequentialBuilder':
        return self.reshape([-1])

    def _channel_first_to_last_nd(self, spatial_ndims, layer_cls):
        in_shape = self._assert_out_shape([False] * (spatial_ndims + 1))
        out_shape = in_shape[1:] + in_shape[:1]
        return self.add(layer_cls(), out_shape)

    def channel_first_to_last1d(self):
        return self._channel_first_to_last_nd(1, ChannelFirstToLast1d)

    def channel_first_to_last2d(self):
        return self._channel_first_to_last_nd(2, ChannelFirstToLast2d)

    def channel_first_to_last3d(self):
        return self._channel_first_to_last_nd(3, ChannelFirstToLast3d)

    def _channel_last_to_first_nd(self, spatial_ndims, layer_cls):
        in_shape = self._assert_out_shape([False] * (spatial_ndims + 1))
        out_shape = in_shape[-1:] + in_shape[:-1]
        return self.add(layer_cls(), out_shape)

    def channel_last_to_first1d(self):
        return self._channel_last_to_first_nd(1, ChannelLastToFirst1d)

    def channel_last_to_first2d(self):
        return self._channel_last_to_first_nd(2, ChannelLastToFirst2d)

    def channel_last_to_first3d(self):
        return self._channel_last_to_first_nd(3, ChannelLastToFirst3d)

    if T.IS_CHANNEL_LAST:
        channel_last_to_default1d = \
            channel_last_to_default2d = \
            channel_last_to_default3d = \
            channel_default_to_last1d = \
            channel_default_to_last2d = \
            channel_default_to_last3d = \
            identity
    else:
        channel_last_to_default1d = channel_last_to_first1d
        channel_last_to_default2d = channel_last_to_first2d
        channel_last_to_default3d = channel_last_to_first3d
        channel_default_to_last1d = channel_first_to_last1d
        channel_default_to_last2d = channel_first_to_last2d
        channel_default_to_last3d = channel_first_to_last3d
