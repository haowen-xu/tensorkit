import re
from contextlib import contextmanager
from typing import *

from mltk.utils import NOT_SET

from .activation import *
from .composed import *
from .core import *
from .pool import *
from .resnet import *
from .shape_ import *
from .. import tensor as T
from ..arg_check import *
from ..typing_ import *

__all__ = ['SequentialBuilder']


def _get_layer_class(name: str) -> type:
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
            l = T.utils.calculate_deconv_output_size(d[i], [k], [s], [p], [op], [d])[0]
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

    args: Dict[type, Dict[str, Any]]

    def __init__(self, layer_args: Optional['LayerArgs'] = None):
        """
        Construct a new :class:`LayerArgs` instance.

        Args:
            layer_args: Clone from this :class:`LayerArgs` instance.
        """
        if layer_args is not None:
            self.args = {type_: {key: val for key, val in type_args.items()}
                         for type_, type_args in layer_args.args.items()}
        else:
            self.args = {}

    def set_args(self,
                 type_or_types_: Union[
                     str, Type[T.Module], Sequence[Union[str, Type[T.Module]]]],
                 **kwargs):
        """
        Set default arguments for the specified layer types.

        Args:
            type_or_types_: The layer type or types.
            **kwargs: The default arguments to be set.
        """
        if isinstance(type_or_types_, (str, type)):
            type_or_types_ = [type_or_types_]

        for type_ in type_or_types_:
            if isinstance(type_, str):
                type_ = _get_layer_class(type_)
            if type_ not in self.args:
                self.args[type_] = {}
            self.args[type_].update(kwargs)

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
        layer_args = self.args.get(type_)
        if layer_args:
            for key, val in layer_args.items():
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
        return type_(*args, **self.get_kwargs(type_, **kwargs))


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
                 in_spatial_shape: List[int] = NOT_SET,
                 in_builder: 'SequentialBuilder' = NOT_SET):
        """
        Construct a new :class:`SequentialBuilder`.

        Args:
            in_spec: Positional argument, maybe the input shape, the number
                of input channels, or another instance of `SequentialBuilder`,
                whose layer arguments will be cloned and `out_shape` will be
                used as the `in_shape` of this :class:`SequentialBuilder`.
            in_shape: The input shape.
            in_channels: The number of input channels.
            in_spatial_shape: The input spatial shape.  Can be specified
                only if `in_channels` is specified, or `in_spec` is a int.
            in_builder: Explicitly specify the previous sequential builder.
        """

        # parse the argument
        if int(in_spec is not NOT_SET) + int(in_shape is not NOT_SET) + \
                int(in_channels is not NOT_SET) + int(in_builder is not NOT_SET) != 1:
            raise ValueError(
                'One and only one of `in_spec`, `in_shape`, `in_channels` and '
                '`in_builder` should be specified.'
            )

        if isinstance(in_spec, SequentialBuilder):
            in_builder = in_spec
            layer_args = LayerArgs(in_builder.layer_args)
        elif hasattr(in_spec, '__iter__'):
            in_shape = in_spec
            layer_args = LayerArgs()
        else:
            in_channels = in_spec
            layer_args = LayerArgs()

        if in_spatial_shape is not NOT_SET and in_channels is NOT_SET:
            raise ValueError(
                '`in_spatial_shape` can be specified only when `in_channels` '
                'is specified, or `in_spec` is None or an integer.'
            )

        if in_shape is not NOT_SET:
            in_shape = list(in_shape)
        elif in_channels is not NOT_SET:
            if in_spatial_shape is NOT_SET:
                in_spatial_shape = []
            in_shape = _unsplit_channel_spatial(in_channels, in_spatial_shape)
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
            if channel is None:
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
        self.layer_args.set_args(type_or_types_, **kwargs)
        return self

    @contextmanager
    def arg_scope(self,
                  type_or_types_: Union[str, type, Sequence[Union[str, type]]],
                  **kwargs) -> Generator[None, None, None]:
        """
        Set layer default arguments within a scope, which will be restore to
        the previous values after exiting the scope.

        Args:
            type_or_types_: The layer type or types.
            **kwargs: The default arguments.
        """
        old_layer_args = self.layer_args
        layer_args = LayerArgs(old_layer_args)
        layer_args.set_args(type_or_types_, **kwargs)
        self.layer_args = layer_args
        try:
            yield
        finally:
            self.layer_args = old_layer_args

    def add(self,
            layer: T.Module,
            out_shape: List[Optional[int]] = NOT_SET,
            *,
            out_channels: Optional[int] = NOT_SET,
            out_spatial_shape: List[Optional[int]] = NOT_SET
            ) -> 'SequentialBuilder':
        """
        Manually add a layer to this builder.

        Args:
            layer: The layer to be added.
            out_shape: The new output shape.
            out_channels: The new output channels.  Should be specified and
                only be specified when `out_shape` is not.
            out_spatial_shape: The new spatial shape.  Should only be specified
                when `out_channels` is specified.

        Returns:
            This sequential builder object.
        """
        if (out_shape is NOT_SET) == (out_channels is NOT_SET):
            raise ValueError('Either `out_shape` or `out_channels` should be '
                             'specified, but not both.')
        if out_spatial_shape is not NOT_SET and out_channels is NOT_SET:
            raise ValueError('`out_spatial_shape` can only be specified when '
                             '`out_channels` is specified.')

        if out_channels is not NOT_SET:
            if out_spatial_shape is NOT_SET:
                out_spatial_shape = []
            out_shape = _unsplit_channel_spatial(out_channels, out_spatial_shape)

        self.layers.append(layer)
        self.out_shape = out_shape
        return self

    def build(self, flatten_to_ndims: bool = True) -> T.Module:
        """
        Build the sequential layer.

        Args:
            flatten_to_ndims: Whether or not to wrap the sequential layer
                with a :class:`FlattenToNDims` layer?

        Returns:
            The built sequential layer.
        """
        if not self.layers:
            raise RuntimeError('No layer has been added.')
        elif len(self.layers) == 1:
            layer = self.layers[0]
        else:
            layer = Sequential(self.layers)

        if flatten_to_ndims:
            layer = FlattenToNDims(layer, ndims=len(self.in_shape) + 1)
        return layer

    # ---- activation ----
    def _make_activation(self, type_):
        self._assert_out_shape((False,), at_least=True)
        layer = self.layer_args.build(type_)
        return self.add(layer, self.out_shape)

    def relu(self):
        return self._make_activation(ReLU)

    def leaky_relu(self):
        return self._make_activation(LeakyReLU)

    def sigmoid(self):
        return self._make_activation(Sigmoid)

    def tanh(self):
        return self._make_activation(Tanh)

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
        in_channels, in_size = self._split_out_shape(True, [False] * spatial_ndims)

        # validate the arguments
        kwargs = self.layer_args.get_kwargs(conv_cls, **kwargs)
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
        in_channels, in_size = self._split_out_shape(True, [False] * spatial_ndims)

        # validate the arguments
        kwargs = self.layer_args.get_kwargs(deconv_cls, **kwargs)
        kernel_size = validate_conv_size('kernel_size', kwargs['kernel_size'], spatial_ndims)
        stride = validate_conv_size('stride', kwargs.get('stride', 1), spatial_ndims)
        dilation = validate_conv_size('dilation', kwargs.get('dilation', 1), spatial_ndims)
        padding = validate_padding(
            kwargs.get('padding', PaddingMode.DEFAULT), kernel_size, dilation, spatial_ndims)

        if 'output_padding' in kwargs and output_size is not NOT_SET:
            raise ValueError('`output_padding` and `out_shape` cannot be both specified.')
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
            out_size = output_size
            output_padding = [
                T.utils.calculate_deconv_output_padding(*args)
                for args in zip(
                    in_size, output_size, kernel_size, stride, padding, dilation)
            ]
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
            1, LinearConvTranspose1d, out_channels, output_size, **kwargs)

    def linear_conv_transpose2d(self,
                                out_channels: int,
                                output_size: List[int] = NOT_SET,
                                **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            2, LinearConvTranspose2d, out_channels, output_size, **kwargs)

    def linear_conv_transpose3d(self,
                                out_channels: int,
                                output_size: List[int] = NOT_SET,
                                **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            3, LinearConvTranspose3d, out_channels, output_size, **kwargs)

    def conv_transpose1d(self,
                         out_channels: int,
                         output_size: List[int] = NOT_SET,
                         **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            1, ConvTranspose1d, out_channels, output_size, **kwargs)

    def conv_transpose2d(self,
                         out_channels: int,
                         output_size: List[int] = NOT_SET,
                         **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            2, ConvTranspose2d, out_channels, output_size, **kwargs)

    def conv_transpose3d(self,
                         out_channels: int,
                         output_size: List[int] = NOT_SET,
                         **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            3, ConvTranspose3d, out_channels, output_size, **kwargs)

    def res_block_transpose1d(self,
                              out_channels: int,
                              output_size: List[int] = NOT_SET,
                              **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            1, ResBlockTranspose1d, out_channels, output_size, **kwargs)

    def res_block_transpose2d(self,
                              out_channels: int,
                              output_size: List[int] = NOT_SET,
                              **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            2, ResBlockTranspose2d, out_channels, output_size, **kwargs)

    def res_block_transpose3d(self,
                              out_channels: int,
                              output_size: List[int] = NOT_SET,
                              **kwargs) -> 'SequentialBuilder':
        return self._deconv_nd(
            3, ResBlockTranspose3d, out_channels, output_size, **kwargs)

    # aliases for the deconvolution layers
    linear_deconv1d = linear_conv_transpose1d
    linear_deconv2d = linear_conv_transpose2d
    linear_deconv3d = linear_conv_transpose3d
    deconv1d = conv_transpose1d
    deconv2d = conv_transpose2d
    deconv3d = conv_transpose3d

    # ---- pool layers ----
    def _pool_nd(self, spatial_ndims, pool_cls, **kwargs):
        in_channels, in_size = self._split_out_shape(True, [False] * spatial_ndims)

        # validate the arguments
        kwargs = self.layer_args.get_kwargs(pool_cls, **kwargs)
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
        in_channels, in_size = self._split_out_shape(True, [False] * spatial_ndims)
        keepdims = kwargs.get('keepdims', False)
        if keepdims:
            out_shape = _unsplit_channel_spatial(in_channels, [1] * spatial_ndims)
        else:
            out_shape = [in_channels]
        layer = pool_cls(**self.layer_args.get_kwargs(pool_cls, **kwargs))
        return self.add(layer, out_shape)

    def global_avg_pool1d(self, **kwargs) -> 'SequentialBuilder':
        return self._global_avg_pool_nd(1, GlobalAvgPool1d, **kwargs)

    def global_avg_pool2d(self, **kwargs) -> 'SequentialBuilder':
        return self._global_avg_pool_nd(2, GlobalAvgPool2d, **kwargs)

    def global_avg_pool3d(self, **kwargs) -> 'SequentialBuilder':
        return self._global_avg_pool_nd(3, GlobalAvgPool3d, **kwargs)
