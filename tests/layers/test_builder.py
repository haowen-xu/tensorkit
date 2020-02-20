from itertools import product

import mltk
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.layers import *
from tests.helper import *
from tests.ops import *


class _RecordInitArgsLayer(BaseLayer):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = tuple(args)
        self.kwargs = dict(kwargs)

    def __repr__(self):
        return repr((self.args, self.kwargs))

    def __eq__(self, other):
        if isinstance(other, _RecordInitArgsLayer):
            args, kwargs = other.args, other.kwargs
        else:
            args, kwargs = other
        return args == self.args and kwargs == self.kwargs


class LayerArgsTestCase(TestCase):

    def test_set_args(self):
        # empty default args
        args = tk.layers.LayerArgs()
        self.assertEqual(args.get_kwargs(_RecordInitArgsLayer), {})

        o = args.build(_RecordInitArgsLayer)
        self.assertIsInstance(o, _RecordInitArgsLayer)
        self.assertEqual(o, ((), {}))

        # set default args
        args.set_args(_RecordInitArgsLayer, d=4)
        self.assertEqual(args.get_kwargs(_RecordInitArgsLayer), {'d': 4})
        self.assertEqual(args.get_kwargs(_RecordInitArgsLayer, c=3, d=5), {'c': 3, 'd': 5})

        o = args.build(_RecordInitArgsLayer)
        self.assertIsInstance(o, _RecordInitArgsLayer)
        self.assertEqual(o, ((), {'d': 4}))

        o = args.build(_RecordInitArgsLayer, 1, 2, c=3, d=5)
        self.assertIsInstance(o, _RecordInitArgsLayer)
        self.assertEqual(o, ((1, 2), {'c': 3, 'd': 5}))

        # inherit default args from previous instance
        args2 = tk.layers.LayerArgs(args)
        args2.set_args([_RecordInitArgsLayer], c=5)
        self.assertEqual(args2.get_kwargs(_RecordInitArgsLayer), {'c': 5, 'd': 4})
        self.assertEqual(args.get_kwargs(_RecordInitArgsLayer), {'d': 4})  # should not change

    def test_layer_names_as_types(self):
        args = tk.layers.LayerArgs()
        args.set_args(['dense', 'conv2d'], activation=tk.layers.LeakyReLU)
        args.set_args(['conv2d'], kernel_size=3)

        self.assertEqual(args.get_kwargs('dense'), {'activation': tk.layers.LeakyReLU})
        self.assertEqual(args.get_kwargs('conv2d'), {
            'activation': tk.layers.LeakyReLU,
            'kernel_size': 3,
        })

        l1 = args.build('dense', 4, 4)
        self.assertIsInstance(l1[1], tk.layers.LeakyReLU)
        l2 = args.build('conv2d', 4, 4)
        self.assertIsInstance(l2[1], tk.layers.LeakyReLU)
        self.assertEqual(T.shape(l2[0].weight_store()), [4, 4, 3, 3])


def sequential_builder_standard_check(ctx,
                                      fn_name,
                                      layer_cls,
                                      input_shape,
                                      input_mask,
                                      args,
                                      builder_args,
                                      kwargs,
                                      layer_kwargs=None,
                                      builder_kwargs=None,
                                      output_mask=None,
                                      at_least=None):
    if output_mask is None:
        output_mask = input_mask
    x = T.random.randn([3] + input_shape)

    # the expected layer
    T.random.seed(1234)
    layer_kwargs = dict(layer_kwargs or {})
    layer_kwargs.update(kwargs)
    layer0 = layer_cls(*args, **layer_kwargs)
    y = layer0(x)
    output_shape = T.shape(y)[1:]

    def fn(input_shape, output_shape, kwargs,
           builder_set_arg_layer, builder_set_arg_kwargs):
        T.random.seed(1234)
        builder = SequentialBuilder(input_shape)
        ctx.assertEqual(builder.in_shape, input_shape)
        if builder_set_arg_kwargs:
            ctx.assertIs(
                builder.set_args(
                    builder_set_arg_layer, **builder_set_arg_kwargs),
                builder,
            ),
        ctx.assertIs(
            getattr(builder, fn_name)(*builder_args, **kwargs),
            builder,
        )
        ctx.assertEqual(builder.out_shape, output_shape)
        layer = builder.build(False)
        ctx.assertIsInstance(layer, layer_cls)
        assert_allclose(layer(x), y, rtol=1e-4, atol=1e-6)

    def apply_mask(shape, mask):
        return [s if m else None for s, m in zip(shape, mask)]

    # do check various ways to specify the arguments
    builder_kwargs = dict(builder_kwargs or {})
    builder_kwargs.update(kwargs)
    fn(input_shape, output_shape, builder_kwargs, None, {})
    fn(input_shape, output_shape, {}, layer_cls, builder_kwargs)
    fn(input_shape, output_shape, {}, fn_name, builder_kwargs)

    if False in input_mask:
        fn(apply_mask(input_shape, input_mask),
           apply_mask(output_shape, output_mask),
           {}, fn_name, builder_kwargs)

    # check some common error checks
    if 'kernel_size' in builder_kwargs:
        kwargs2 = dict(builder_kwargs)
        kwargs2.pop('kernel_size')
        with pytest.raises(ValueError,
                           match='The `kernel_size` argument is required'):
            fn(input_shape, output_shape, kwargs2, None, {})

    if 'output_size' not in builder_kwargs:
        for i, m in enumerate(input_mask):
            if not m:
                continue
            input_shape2 = list(input_shape)
            input_shape2[i] = None
            with pytest.raises(ValueError,
                               match=f'Axis {i - len(input_shape)} of the previous '
                                     f'output shape is expected to be deterministic'):
                fn(input_shape2, output_shape, builder_kwargs, None, {})

    if len(input_shape) >= 1:
        input_shape2 = list(input_shape[:-1])
        if len(input_shape) == at_least:
            with pytest.raises(ValueError,
                               match=f'The previous output shape is expected to '
                                     f'be at least {len(input_shape)}d'):
                fn(input_shape2, output_shape, builder_kwargs, None, {})
        elif at_least is None:
            with pytest.raises(ValueError,
                               match=f'The previous output shape is expected to '
                                     f'be exactly {len(input_shape)}d'):
                fn(input_shape2, output_shape, builder_kwargs, None, {})


class SequentialBuilderTestCase(TestCase):

    def test_construct(self):
        def assert_in_shape(b, s):
            self.assertEqual(b.in_shape, s)
            self.assertEqual(b.out_shape, s)

        # test the input shape
        assert_in_shape(SequentialBuilder(3), [3])
        assert_in_shape(SequentialBuilder(None), [None])
        assert_in_shape(SequentialBuilder(in_channels=3), [3])
        assert_in_shape(SequentialBuilder(in_channels=None), [None])

        assert_in_shape(SequentialBuilder([3]), [3])
        assert_in_shape(SequentialBuilder([3, 4]), [3, 4])
        assert_in_shape(SequentialBuilder((3, 4)), [3, 4])
        assert_in_shape(SequentialBuilder([None, None]), [None, None])
        assert_in_shape(SequentialBuilder(in_shape=[3, 4]), [3, 4])
        assert_in_shape(SequentialBuilder(in_shape=(3, 4)), [3, 4])
        assert_in_shape(SequentialBuilder(in_shape=(None, None)), [None, None])

        assert_in_shape(
            SequentialBuilder(5, in_size=[3, 4]),
            make_conv_shape([], 5, [3, 4]),
        )
        assert_in_shape(
            SequentialBuilder(in_channels=5, in_size=[3, 4]),
            make_conv_shape([], 5, [3, 4]),
        )
        assert_in_shape(
            SequentialBuilder(in_channels=5, in_size=(3, 4)),
            make_conv_shape([], 5, [3, 4]),
        )
        assert_in_shape(
            SequentialBuilder(in_channels=5, in_size=(None, None)),
            [s if s == 5 else None for s in make_conv_shape([], 5, [3, 4])],
        )

        # test copy layer_args
        layer_args = LayerArgs()
        layer_args.set_args(['dense', 'conv2d'], activation=tk.layers.LeakyReLU)
        layer_args.set_args('conv2d', kernel_size=3)
        builder = SequentialBuilder(5, layer_args=layer_args)
        self.assertEqual(
            builder.layer_args.get_kwargs(Dense),
            {'activation': tk.layers.LeakyReLU}
        )
        self.assertEqual(
            builder.layer_args.get_kwargs(Conv2d),
            {'activation': tk.layers.LeakyReLU, 'kernel_size': 3}
        )

        # test in_builder
        in_shape0 = make_conv_shape([], 5, [3, 4])
        for in_shape in (in_shape0, [None if i != 5 else i for i in in_shape0]):
            # test init from another build
            builder0 = SequentialBuilder(in_shape)
            builder0.set_args(['dense', 'conv2d'], activation=tk.layers.LeakyReLU)
            builder0.set_args('conv2d', kernel_size=3)

            builder = SequentialBuilder(builder0)
            assert_in_shape(builder, in_shape)
            self.assertEqual(
                builder.layer_args.get_kwargs(Dense),
                {'activation': tk.layers.LeakyReLU}
            )
            self.assertEqual(
                builder.layer_args.get_kwargs(Conv2d),
                {'activation': tk.layers.LeakyReLU, 'kernel_size': 3}
            )

            # test override builder args with layer_args
            layer_args = LayerArgs()
            layer_args.set_args(['dense'], activation=tk.layers.Sigmoid)
            builder = SequentialBuilder(builder0, layer_args=layer_args)
            assert_in_shape(builder, in_shape)
            self.assertEqual(
                builder.layer_args.get_kwargs(Dense),
                {'activation': tk.layers.Sigmoid}
            )
            self.assertEqual(
                builder.layer_args.get_kwargs(Conv2d),
                {}
            )

        # test arg errors
        with pytest.raises(ValueError,
                           match='One and only one of `in_spec`, `in_shape`, '
                                 '`in_channels` and `in_builder` should be '
                                 'specified'):
            _ = SequentialBuilder()

        arg_values = {
            'in_spec': [3, 4],
            'in_shape': [5, 6],
            'in_channels': 7,
            'in_builder': builder0,
        }
        for arg1, arg2 in product(
                ['in_spec', 'in_shape', 'in_channels', 'in_builder'],
                ['in_spec', 'in_shape', 'in_channels', 'in_builder']):
            if arg1 == arg2:
                continue
            with pytest.raises(ValueError,
                               match='One and only one of `in_spec`, `in_shape`, '
                                     '`in_channels` and `in_builder` should be '
                                     'specified'):
                _ = SequentialBuilder(**{arg1: arg_values[arg1],
                                         arg2: arg_values[arg2]})
        for arg in ['in_spec', 'in_shape', 'in_builder']:
            with pytest.raises(ValueError,
                               match='`in_size` can be specified only when '
                                     '`in_channels` is specified, or `in_spec` '
                                     'is None or an integer'):
                _ = SequentialBuilder(in_size=[8, 9], **{arg: arg_values[arg]})

    def test_arg_scope(self):
        builder = SequentialBuilder(5)
        self.assertEqual(builder.layer_args.get_kwargs(Dense), {})
        self.assertEqual(builder.layer_args.get_kwargs(Conv2d), {})
        with builder.arg_scope(['conv2d', Dense], activation=LeakyReLU):
            self.assertEqual(builder.layer_args.get_kwargs(Dense),
                             {'activation': LeakyReLU})
            self.assertEqual(builder.layer_args.get_kwargs(Conv2d),
                             {'activation': LeakyReLU})
            with builder.arg_scope('dense', activation=Sigmoid, normalizer=BatchNorm):
                self.assertEqual(builder.layer_args.get_kwargs(Dense),
                                 {'activation': Sigmoid, 'normalizer': BatchNorm})
                with builder.arg_scope(Conv2d, activation=Tanh, normalizer=BatchNorm2d):
                    self.assertEqual(builder.layer_args.get_kwargs(Conv2d),
                                     {'activation': Tanh, 'normalizer': BatchNorm2d})
                self.assertEqual(builder.layer_args.get_kwargs(Conv2d),
                                 {'activation': LeakyReLU})
            self.assertEqual(builder.layer_args.get_kwargs(Dense),
                             {'activation': LeakyReLU})
        self.assertEqual(builder.layer_args.get_kwargs(Dense), {})
        self.assertEqual(builder.layer_args.get_kwargs(Conv2d), {})

    def test_add(self):
        def fn(in_shape, layer, out_shape):
            # test using `out_shape`
            builder = SequentialBuilder(in_shape)
            self.assertIs(builder.add(layer, out_shape), builder)
            self.assertEqual(builder.out_shape, out_shape)
            self.assertIs(builder.build(False), layer)

            with pytest.raises(ValueError,
                               match='`out_size` can only be specified when '
                                     '`out_channels` is specified'):
                _ = builder.add(layer, out_shape, out_size=[])

            # test using `out_channels` and `out_size`
            def g(out_channels, **out_size_args):
                builder = SequentialBuilder(in_shape)
                self.assertIs(
                    builder.add(layer, out_channels=out_channels, **out_size_args),
                    builder
                )
                self.assertEqual(builder.out_shape, out_shape)
                self.assertIs(builder.build(False), layer)

                # test error
                with pytest.raises(ValueError,
                                   match='Either `out_shape` or `out_channels` '
                                         'should be specified, but not both'):
                    _ = builder.add(layer, out_shape, out_channels=out_channels,
                                    **out_size_args)
                with pytest.raises(ValueError,
                                   match='Either `out_shape` or `out_channels` '
                                         'should be specified, but not both'):
                    _ = builder.add(layer)

            if len(out_shape) > 1:
                if T.IS_CHANNEL_LAST:
                    out_channels, out_size = out_shape[-1], out_shape[:-1]
                else:
                    out_channels, out_size = out_shape[0], out_shape[1:]
                g(out_channels, out_size=out_size)
            else:
                g(out_shape[0], out_size=[])
                g(out_shape[0])

        fn([5], Linear(5, 3), [3])
        fn([None], Linear(5, 3), [None])
        fn(make_conv_shape([], 5, [6, 7]),
           Conv2d(5, 3, kernel_size=1),
           make_conv_shape([], 3, [6, 7]))
        fn(make_conv_shape([], None, [None, None]),
           Conv2d(5, 3, kernel_size=1),
           make_conv_shape([], None, [None, None]))

    def test_build(self):
        builder = SequentialBuilder(5)
        self.assertIsInstance(builder.build(), Identity)
        self.assertIsInstance(builder.build(False), Identity)

        # build with one layer
        builder.dense(4)
        l1 = builder.build(False)
        self.assertIsInstance(l1, Dense)
        l = builder.build(True)
        self.assertIsInstance(l, FlattenToNDims)
        x = T.random.randn([3, 5])
        assert_allclose(l(x), l1(x), rtol=1e-4, atol=1e-6)

        # build with two layers
        builder.linear(3)
        l = builder.build(False)
        self.assertIsInstance(l, Sequential)
        self.assertIs(l[0], l1)
        l2 = l[-1]
        self.assertIsInstance(l2, Linear)
        l = builder.build(True)
        self.assertIsInstance(l, FlattenToNDims)
        x = T.random.randn([3, 5])
        assert_allclose(l(x), l2(l1(x)), rtol=1e-4, atol=1e-6)

    def test_identity(self):
        for in_shape in ([], [5], [3, 4, 5]):
            sequential_builder_standard_check(
                ctx=self, fn_name='identity', layer_cls=Identity,
                input_shape=in_shape, input_mask=[False] * len(in_shape),
                args=(), builder_args=(), kwargs={}, at_least=0,
            )

    def test_activation(self):
        for name in ['relu', 'leaky_relu', 'sigmoid', 'tanh', 'log_softmax']:
            layer_cls = tk.layers.get_activation_class(name)
            for in_shape in ([5], [3, 4, 5]):
                sequential_builder_standard_check(
                    ctx=self, fn_name=name, layer_cls=layer_cls,
                    input_shape=in_shape, input_mask=[False] * len(in_shape),
                    args=(), builder_args=(), kwargs={}, at_least=1,
                )

    def test_linear(self):
        sequential_builder_standard_check(
            ctx=self, fn_name='linear', layer_cls=Linear, input_shape=[5],
            input_mask=[True], args=(5, 4), builder_args=(4,),
            kwargs={'weight_norm': True},
        )
        sequential_builder_standard_check(
            ctx=self, fn_name='dense', layer_cls=Dense, input_shape=[5],
            input_mask=[True], args=(5, 4), builder_args=(4,),
            kwargs={'weight_norm': True, 'activation': LeakyReLU},
        )

    def test_conv_and_deconv(self):
        for spatial_ndims in (1, 2, 3):
            input_shape = make_conv_shape([], 5, [15, 16, 17][:spatial_ndims])
            input_mask = [i == 5 for i in input_shape]
            for fn_name, layer_cls in zip(
                    [
                        f'linear_conv{spatial_ndims}d',
                        f'conv{spatial_ndims}d',
                        f'res_block{spatial_ndims}d'
                    ],
                    [
                        getattr(tk.layers, f'LinearConv{spatial_ndims}d'),
                        getattr(tk.layers, f'Conv{spatial_ndims}d'),
                        getattr(tk.layers, f'ResBlock{spatial_ndims}d'),
                    ]):
                kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 'half',
                          'weight_norm': True}
                if not fn_name.startswith('linear_'):
                    kwargs['activation'] = LeakyReLU
                sequential_builder_standard_check(
                    ctx=self, fn_name=fn_name, layer_cls=layer_cls,
                    input_shape=input_shape, input_mask=input_mask,
                    args=(5, 4), builder_args=(4,),
                    kwargs=kwargs
                )

    def test_deconv(self):
        for spatial_ndims in (1, 2, 3):
            output_size = [16, 17, 18][:spatial_ndims]
            output_shape = make_conv_shape([], 4, output_size)
            layer0 = getattr(tk.layers, f'LinearConv{spatial_ndims}d')(
                4, 5, kernel_size=3, stride=2, padding='half',
                weight_init=tk.init.ones
            )
            y = layer0(T.zeros([1] + output_shape))
            input_shape = T.shape(y)[1:]
            input_channel, input_size = T.utils.split_channel_spatial_shape(input_shape)

            for fn_name, layer_cls in zip(
                    [
                        f'linear_conv_transpose{spatial_ndims}d',
                        f'linear_deconv{spatial_ndims}d',
                        f'conv_transpose{spatial_ndims}d',
                        f'deconv{spatial_ndims}d',
                        f'res_block_transpose{spatial_ndims}d'
                    ],
                    [
                        getattr(tk.layers, f'LinearConvTranspose{spatial_ndims}d'),
                        getattr(tk.layers, f'LinearConvTranspose{spatial_ndims}d'),
                        getattr(tk.layers, f'ConvTranspose{spatial_ndims}d'),
                        getattr(tk.layers, f'ConvTranspose{spatial_ndims}d'),
                        getattr(tk.layers, f'ResBlockTranspose{spatial_ndims}d'),
                    ]):
                # without output_shape
                kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 'half',
                          'weight_norm': True}
                input_mask = [i == 5 for i in input_shape]
                if not fn_name.startswith('linear_'):
                    kwargs['activation'] = LeakyReLU
                sequential_builder_standard_check(
                    ctx=self, fn_name=fn_name, layer_cls=layer_cls,
                    input_shape=input_shape, input_mask=input_mask,
                    args=(5, 4), builder_args=(4,),
                    kwargs=kwargs
                )
                kwargs['output_padding'] = 0
                sequential_builder_standard_check(
                    ctx=self, fn_name=fn_name, layer_cls=layer_cls,
                    input_shape=input_shape, input_mask=input_mask,
                    args=(5, 4), builder_args=(4,),
                    kwargs=kwargs
                )

                # with output_shape
                kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 'half',
                          'weight_norm': True}
                layer_kwargs = {
                    'output_padding': T.utils.calculate_deconv_output_padding(
                        input_size=input_size,
                        output_size=output_size,
                        kernel_size=[3] * spatial_ndims,
                        stride=[2] * spatial_ndims,
                        padding=[(1, 1)] * spatial_ndims,
                        dilation=[1] * spatial_ndims,
                    )
                }
                builder_kwargs = {'output_size': output_size}
                input_mask = [True] * spatial_ndims
                if not fn_name.startswith('linear_'):
                    kwargs['activation'] = LeakyReLU
                sequential_builder_standard_check(
                    ctx=self, fn_name=fn_name, layer_cls=layer_cls,
                    input_shape=input_shape, input_mask=input_mask,
                    args=(5, 4), builder_args=(4,),
                    kwargs=kwargs, layer_kwargs=layer_kwargs,
                    builder_kwargs=builder_kwargs,
                )

                # test errors
                builder = SequentialBuilder(input_shape)
                fn = getattr(builder, fn_name)
                with pytest.raises(ValueError,
                                   match='`output_padding` and `output_size` '
                                         'cannot be both specified'):
                    fn(5, kernel_size=1,
                       output_padding=1,
                       output_size=[2, 3, 4][:spatial_ndims])

                with pytest.raises(ValueError,
                                   match=f'`output_size` is expected to be '
                                         f'{spatial_ndims}d'):
                    fn(5, kernel_size=1,
                       output_size=[2, 3, 4, 5][:spatial_ndims + 1])

                builder = SequentialBuilder(
                    [i if i == 5 else None for i in input_shape])
                fn = getattr(builder, fn_name)
                with pytest.raises(ValueError,
                                   match='Specifying `output_size` instead of '
                                         '`output_padding` is supported only '
                                         'when the previous output shape '
                                         'is all deterministic.'):
                    fn(5, kernel_size=1, output_size=[2, 3, 4][:spatial_ndims])

    def test_pool(self):
        for spatial_ndims in (1, 2, 3):
            input_shape = make_conv_shape([], 5, [15, 16, 17][:spatial_ndims])
            input_mask = [False] * (spatial_ndims + 1)
            for fn_name, layer_cls in zip(
                    [
                        f'avg_pool{spatial_ndims}d',
                        f'max_pool{spatial_ndims}d',
                    ],
                    [
                        getattr(tk.layers, f'AvgPool{spatial_ndims}d'),
                        getattr(tk.layers, f'MaxPool{spatial_ndims}d'),
                    ]):
                kwargs = {'kernel_size': 3, 'stride': 2, 'padding': 'half'}
                sequential_builder_standard_check(
                    ctx=self, fn_name=fn_name, layer_cls=layer_cls,
                    input_shape=input_shape, input_mask=input_mask,
                    args=(), builder_args=(), kwargs=kwargs
                )

    def test_global_avg_pool(self):
        for spatial_ndims in (1, 2, 3):
            input_shape = make_conv_shape([], 5, [15, 16, 17][:spatial_ndims])
            input_mask = [False] * (spatial_ndims + 1)

            for keepdims in [True, False, None]:
                if keepdims:
                    output_mask = [i != 5 for i in input_shape]
                else:
                    output_mask = [False]

                kwargs = {'keepdims': keepdims} if keepdims is not None else {}
                sequential_builder_standard_check(
                    ctx=self,
                    fn_name=f'global_avg_pool{spatial_ndims}d',
                    layer_cls=getattr(tk.layers, f'GlobalAvgPool{spatial_ndims}d'),
                    input_shape=input_shape, input_mask=input_mask,
                    args=(), builder_args=(), kwargs=kwargs,
                    output_mask=output_mask
                )

    def test_channel_transpose_layers(self):
        for spatial_ndims in (1, 2, 3):
            input_shape = [15, 16, 17, 18][:spatial_ndims + 1]
            input_mask = [False] * (spatial_ndims + 1)
            for fn_name, layer_cls in zip(
                    [
                        f'channel_first_to_last{spatial_ndims}d',
                        f'channel_last_to_first{spatial_ndims}d',
                        f'channel_default_to_last{spatial_ndims}d',
                        f'channel_last_to_default{spatial_ndims}d',
                    ],
                    [
                        getattr(tk.layers, f'ChannelFirstToLast{spatial_ndims}d'),
                        getattr(tk.layers, f'ChannelLastToFirst{spatial_ndims}d'),
                        getattr(tk.layers, f'ChannelDefaultToLast{spatial_ndims}d'),
                        getattr(tk.layers, f'ChannelLastToDefault{spatial_ndims}d'),
                    ]):
                sequential_builder_standard_check(
                    ctx=self, fn_name=fn_name, layer_cls=layer_cls,
                    input_shape=input_shape, input_mask=input_mask,
                    args=(), builder_args=(), kwargs={}
                )
