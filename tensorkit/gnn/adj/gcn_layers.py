from enum import Enum
from typing import *

from mltk import NOT_SET

from ...arg_check import *
from ...layers import (DEFAULT_WEIGHT_INIT, DEFAULT_BIAS_INIT,
                       BaseLayer, ModuleList, ParamStore, Identity,
                       NullParamStore, Sequential, Linear, SimpleParamStore)
from ...tensor import (IS_CHANNEL_LAST, Module, Tensor, shape,
                       reshape, concat, add_n)
from ...tensor.sparse import matmul, sparse_jit_method
from ...typing_ import *

__all__ = [
    'GCNMergeMode',
    'GCNIdentity', 'GCNSequential', 'PartitionedGCNSequential',

    # multiple partitions GCN layers
    'PartitionedGCNLayer', 'PartitionedGCNLayer1d', 'PartitionedGCNLayer2d',
    'PartitionedGCNLayer3d',

    # single partition GCN layers
    'GCNLayer', 'GCNLayer1d', 'GCNLayer2d', 'GCNLayer3d',

    # self-loop GCN layers
    'GCNSelfLoop', 'GCNSelfLoop1d', 'GCNSelfLoop2d', 'GCNSelfLoop3d',

    # standard GCN layers for easy creation
    'GCNDense', 'PartitionedGCNDense',
]


class GCNMergeMode(str, Enum):
    ADD = 'add'
    CONCAT = 'concat'


class GCNIdentity(BaseLayer):
    """A GCN layer that returns the `input` tensor without modification."""

    def forward(self,
                input: Tensor,
                adj: Optional[Tensor] = None) -> Tensor:
        return input


class GCNSequential(BaseLayer):
    """A GCN layer that sequentially calls its children GCN layers."""

    __constants__ = ('gcn_modules',)

    gcn_modules: ModuleList

    def __init__(self, modules: List[Module]):
        super().__init__()
        self.gcn_modules = ModuleList(list(modules))

    def forward(self, input: Tensor, adj: Tensor) -> Tensor:
        for m in self.gcn_modules:
            input = m(input, adj)
        return input


class PartitionedGCNSequential(BaseLayer):
    """
    A partitioned GCN layer that sequentially calls its children partitioned
    GCN layers.
    """

    __constants__ = ('gcn_modules',)

    gcn_modules: ModuleList

    def __init__(self, modules: List[Module]):
        super().__init__()
        self.gcn_modules = ModuleList(list(modules))

    def forward(self, input: Tensor, adj: List[Tensor]) -> Tensor:
        for m in self.gcn_modules:
            input = m(input, adj)
        return input


# ---- multi-relational GCN layer ----
class PartitionedGCNLayer(BaseLayer):
    """
    Graph convolution layer, whose edges can be partitioned into multiple
    groups, such that its adjacency matrix can be decomposed as:

    .. math::

        \\mathbf{A} = \\mathbf{A}_1 + \\dots + \\mathbf{A}_k + \\lambda \\mathbf{I}

    Such a partitioned GCN layer can then be formulated as:

    .. math::

        \\begin{align}
            \\hat{\\mathbf{H}}^{(l)} &= \\sum_{i=1}^k f_i(\\mathbf{A}_k \\mathbf{H}^{(l)}) +
                \\lambda f_0(\\mathbf{H}^{(l)}) \\
            \\mathbf{H}^{(l + 1)} &= g(\\hat{\\mathbf{H}}^{(l)})
        \\end{align}

    where `g(X)` is further decomposed as `g(X) = activation(normalizer(X))`.
    """

    __constants__ = (
        'partition_modules', 'self_module', 'bias_store', 'post_linear',
        'use_bias', 'use_self_module', 'self_weight', 'use_post_linear',
        'n_partitions', 'feature_matrix_ndims',
    )

    partition_modules: ModuleList
    self_module: Module
    bias_store: Module
    post_linear: Module  # activation(normalizer(X))
    use_bias: bool
    use_self_module: bool
    self_weight: float
    use_post_linear: bool

    merge_mode: int  # 0 = 'add', 1 = 'concat'
    n_partitions: int
    feature_matrix_ndims: int
    feature_axis: int

    def __init__(self,
                 modules: Sequence[Module],
                 self_module: Optional[Module] = None,
                 self_weight: float = 1.,
                 bias_store: Optional[ParamStore] = None,
                 normalizer: Optional[Module] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 feature_matrix_ndims: int = NOT_SET,
                 feature_axis: int = NOT_SET,
                 merge_mode: Union[str, GCNMergeMode] = 'add'
                 ):
        # validate the parameters
        if feature_matrix_ndims is NOT_SET:
            feature_matrix_ndims = self._get_feature_matrix_ndims()
        else:
            feature_matrix_ndims = int(feature_matrix_ndims)
            if feature_matrix_ndims < 2:
                raise ValueError(f'`feature_matrix_ndims` must be at least 2: '
                                 f'got {feature_matrix_ndims}.')

        if feature_axis is NOT_SET:
            feature_axis = -1 if IS_CHANNEL_LAST else -(feature_matrix_ndims - 1)
        else:
            feature_axis = int(feature_axis)
            if not (-feature_matrix_ndims < feature_axis < 0):
                raise ValueError('`feature_axis` out of range.')

        merge_mode = {
            GCNMergeMode.ADD: 0,
            GCNMergeMode.CONCAT: 1,
        }[GCNMergeMode(merge_mode)]

        modules = list(modules)
        if not modules:
            raise ValueError(
                '`modules` is required not to be empty.  '
                'If you just need a self-loop, use `GCNSelfLoop` instead.')

        use_self_module = self_module is not None
        if self_module is None:
            self_module = Identity()
        self_weight = float(self_weight)

        use_bias = bias_store is not None
        if bias_store is None:
            bias_store = NullParamStore()

        # compose 'use_post_linear'
        post_linear = []
        if normalizer is not None:
            post_linear.append(validate_layer('normalizer', normalizer))
        if activation is not None:
            post_linear.append(get_layer_from_layer_or_factory('activation',
                                                               activation))

        use_post_linear = not not post_linear
        if not post_linear:
            post_linear = Identity()
        elif len(post_linear) == 1:
            post_linear = post_linear[0]
        else:
            post_linear = Sequential(post_linear)

        # build the GCN layer
        super().__init__()
        self.partition_modules = ModuleList(modules)
        self.self_module = self_module
        self.bias_store = bias_store
        self.post_linear = post_linear
        self.use_bias = use_bias
        self.use_self_module = use_self_module
        self.self_weight = self_weight
        self.use_post_linear = use_post_linear
        self.n_partitions = len(modules)
        self.feature_matrix_ndims = feature_matrix_ndims
        self.feature_axis = feature_axis
        self.merge_mode = merge_mode

    def _get_feature_matrix_ndims(self) -> int:
        return 2

    @sparse_jit_method
    def _forward(self, input: Tensor, adj: List[Tensor]) -> Tensor:
        if len(adj) != self.n_partitions:
            raise ValueError(
                '`adj` is expected to have {} element(s), but got {}.'.
                format(self.n_partitions, len(adj))
            )

        # ---- message passing by adjacency matrix ----
        # input shape: (N, B1, B2, ..., K1, K2, ...)
        input_shape = shape(input)
        i_rank = len(input_shape)
        two_dimensional_case = (self.feature_matrix_ndims == i_rank == 2)

        # reshape to `(N, B1 * B2 * ... * K1 * K2 * ...)`
        if not two_dimensional_case:
            if i_rank < self.feature_matrix_ndims:
                raise ValueError(
                    '`input` is expected to be at least {}d, got shape {}.'.
                    format(self.feature_matrix_ndims, input_shape)
                )
            input = reshape(input, [input_shape[0], -1])

        # compute the outputs of modules
        merge_mode = self.merge_mode
        outputs: List[Tensor] = []
        output_shape = (
            [-1] +
            input_shape[i_rank - self.feature_matrix_ndims + 1:]
        )

        i = 0
        for m in self.partition_modules:
            # apply the adjacency matrix A_i
            m_output = matmul(adj[i], input)

            # reshape to `(N * B1 * B2 * ..., K1, K2, ...)`
            if not two_dimensional_case:
                m_output = reshape(m_output, output_shape)

            # apply the `f_i()` transformation
            m_output = m(m_output)
            outputs.append(m_output)

            # move to next module
            i += 1

        # compute the self-loop output
        if self.use_self_module:
            if not two_dimensional_case:
                input = reshape(input, output_shape)
            m_output = self.self_module(input)
            outputs.append(m_output)

        # merge if "concat", or sum if "add"
        if merge_mode == 0:
            input = add_n(outputs)
        else:
            input = concat(outputs, axis=self.feature_axis)

        # de-reference intermediate results to free the memory immediately
        outputs = []
        m_output = input

        # add bias
        if self.use_bias:
            m_output = input = input + self.bias_store()

        # apply post-linear
        if self.use_post_linear:
            m_output = input = self.post_linear(input)

        # reshape to the final output shape: `(N, B1, B2, ..., K1, K2, ...)`
        if not two_dimensional_case:
            output_shape = (
                input_shape[:i_rank - self.feature_matrix_ndims + 1] +
                shape(input)[1:]
            )
            m_output = input = reshape(input, output_shape)

        return input

    def forward(self, input: Tensor, adj: List[Tensor]) -> Tensor:
        return self._forward(input, adj)


class PartitionedGCNLayer1d(PartitionedGCNLayer):

    def _get_feature_matrix_ndims(self) -> int:
        return 3


class PartitionedGCNLayer2d(PartitionedGCNLayer):

    def _get_feature_matrix_ndims(self) -> int:
        return 4


class PartitionedGCNLayer3d(PartitionedGCNLayer):

    def _get_feature_matrix_ndims(self) -> int:
        return 5


# ---- GCN layer with `A` != `I` ----
class GCNLayer(PartitionedGCNLayer):
    """
    Graph convolution layer, whose edges can be partitioned into two groups:
    the self-loop edges and other edges, such that its adjacency matrix can
    be decomposed as:

    .. math::

        \\mathbf{A} = \\mathbf{A}_1 + \\lambda \\mathbf{I}

    See Also:
        :class:`PartitionedGCNLayer`
    """

    def __init__(self,
                 module: Module,
                 self_module: Optional[Module] = None,
                 self_weight: float = 1.,
                 bias_store: Optional[ParamStore] = None,
                 normalizer: Optional[Module] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 feature_matrix_ndims: int = NOT_SET,
                 feature_axis: int = NOT_SET,
                 merge_mode: Union[str, GCNMergeMode] = 'add'
                 ):
        super().__init__(
            modules=[module],
            self_module=self_module,
            self_weight=self_weight,
            bias_store=bias_store,
            normalizer=normalizer,
            activation=activation,
            feature_matrix_ndims=feature_matrix_ndims,
            feature_axis=feature_axis,
            merge_mode=merge_mode,
        )

    def forward(self, input: Tensor, adj: Tensor) -> Tensor:
        return self._forward(input, [adj])


class GCNLayer1d(GCNLayer):

    def _get_feature_matrix_ndims(self) -> int:
        return 3


class GCNLayer2d(GCNLayer):

    def _get_feature_matrix_ndims(self) -> int:
        return 4


class GCNLayer3d(GCNLayer):

    def _get_feature_matrix_ndims(self) -> int:
        return 5


# ---- GCN layer with `A` === `I` ----
class GCNSelfLoop(BaseLayer):
    """
    Graph convolution layer that uses an ordinary layer as the `f()`
    transformation of the GCN layer, with the adjacency matrix `A`
    assumed to be identity matrix `I`.
    """

    __constants__ = ('module', 'feature_matrix_ndims')

    module: Module
    """The `f()` transformation."""

    feature_matrix_ndims: int
    """
    The number of dimensions of the feature matrix 
    :math:`f(\\hat{\\mathbf{H}}^{(l)})`.  Should be at least 2.
    """

    def __init__(self, module: Module, feature_matrix_ndims: int = 2):
        feature_matrix_ndims = int(feature_matrix_ndims)
        if feature_matrix_ndims < 2:
            raise ValueError(f'`feature_matrix_ndims` must be at least 2: '
                             f'got {feature_matrix_ndims}.')

        super().__init__()
        self.module = module
        self.feature_matrix_ndims = feature_matrix_ndims

    def forward(self, input: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        """
        Apply adjacency matrix on the input feature matrix.

        .. math::

            \\hat{\\mathbf{H}}^{(l)} = \\mathbf{A} \\mathbf{H}^{(l)}

        Args:
            input: The input feature matrix :math:`\\mathbf{H}^{(l)}`.
                The input shape should be `(N, B1, B2, ..., K1, K2, ...)`,
                where `B1, B2, ...` are the batch dimensions (if exists), and
                `K1, K2, ...` are the feature dimensions (at least one).
            adj: Not used.  Will be ignored if specified.

        Returns:
            The output feature matrix :math:`\\hat{\\mathbf{H}}^{(l)}`.
        """
        # input shape: (N, B1, B2, ..., K1, K2, ...)
        input_shape = shape(input)
        i_rank = len(input_shape)
        two_dimensional_case = self.feature_matrix_ndims == i_rank == 2

        # reshape to `(N * B1 * B2 * ..., K1, K2, ...)`
        if not two_dimensional_case:
            output_shape = (
                [-1] +
                input_shape[i_rank - self.feature_matrix_ndims + 1:]
            )
            input = reshape(input, output_shape)

        # ---- apply the `f()` transformation ----
        input = self.module(input)

        # reshape to the final output shape: `(N, B1, B2, ..., K1, K2, ...)`
        if not two_dimensional_case:
            output_shape = (
                input_shape[:i_rank - self.feature_matrix_ndims + 1] +
                shape(input)[1:]
            )
            input = reshape(input, output_shape)

        return input


class GCNSelfLoop1d(GCNSelfLoop):

    def __init__(self, module: Module):
        super().__init__(module, 3)


class GCNSelfLoop2d(GCNSelfLoop):

    def __init__(self, module: Module):
        super().__init__(module, 4)


class GCNSelfLoop3d(GCNSelfLoop):

    def __init__(self, module: Module):
        super().__init__(module, 5)


# ---- standard GCN classes ----
class GCNDense(GCNLayer):
    """A standard dense GCN layer."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_self_loop: bool = False,
                 self_weight: float = 1.,
                 merge_mode: Union[str, GCNMergeMode] = 'add',
                 use_bias: Optional[bool] = None,
                 normalizer: Optional[NormalizerOrNormalizerFactory] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        if use_bias is None:
            use_bias = normalizer is None
        linear_kwargs = dict(
            use_bias=False,
            weight_norm=weight_norm,
            weight_init=weight_init,
            data_init=data_init,
            device=device,
        )
        module = Linear(in_features, out_features, **linear_kwargs)
        self_module = (Linear(in_features, out_features, **linear_kwargs)
                       if use_self_loop else None)

        if use_bias:
            out_dup = 1 + int(use_self_loop and merge_mode == 'concat')
            bias_shape = [out_features * out_dup]
            bias_store = SimpleParamStore(
                bias_shape, initializer=bias_init, device=device)
        else:
            bias_store = None

        if normalizer is not None:
            normalizer = get_layer_from_layer_or_factory(
                'normalizer', normalizer, args=(out_features,))

        if activation is not None:
            activation = get_layer_from_layer_or_factory('activation', activation)

        super().__init__(
            module=module, self_module=self_module, self_weight=self_weight,
            bias_store=bias_store, normalizer=normalizer, activation=activation,
            merge_mode=merge_mode,
        )
class GCNDense(GCNLayer):
    """A standard dense GCN layer."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 use_self_loop: bool = False,
                 self_weight: float = 1.,
                 merge_mode: Union[str, GCNMergeMode] = 'add',
                 use_bias: Optional[bool] = None,
                 normalizer: Optional[NormalizerOrNormalizerFactory] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        if use_bias is None:
            use_bias = normalizer is None
        linear_kwargs = dict(
            use_bias=False,
            weight_norm=weight_norm,
            weight_init=weight_init,
            data_init=data_init,
            device=device,
        )
        module = Linear(in_features, out_features, **linear_kwargs)
        self_module = (Linear(in_features, out_features, **linear_kwargs)
                       if use_self_loop else None)

        if use_bias:
            out_dup = (1 + int(use_self_loop)
                       if merge_mode == 'concat' else 1)
            bias_shape = [out_features * out_dup]
            bias_store = SimpleParamStore(
                bias_shape, initializer=bias_init, device=device)
        else:
            bias_store = None

        if normalizer is not None:
            normalizer = get_layer_from_layer_or_factory(
                'normalizer', normalizer, args=(out_features,))

        if activation is not None:
            activation = get_layer_from_layer_or_factory('activation', activation)

        super().__init__(
            module=module, self_module=self_module, self_weight=self_weight,
            bias_store=bias_store, normalizer=normalizer, activation=activation,
            merge_mode=merge_mode,
        )


class PartitionedGCNDense(PartitionedGCNLayer):
    """Partitioned dense GCN layer."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 n_partitions: int,
                 use_self_loop: bool = False,
                 self_weight: float = 1.,
                 merge_mode: Union[str, GCNMergeMode] = 'add',
                 use_bias: Optional[bool] = None,
                 normalizer: Optional[NormalizerOrNormalizerFactory] = None,
                 activation: Optional[LayerOrLayerFactory] = None,
                 weight_norm: WeightNormArgType = False,
                 weight_init: TensorInitArgType = DEFAULT_WEIGHT_INIT,
                 bias_init: TensorInitArgType = DEFAULT_BIAS_INIT,
                 data_init: Optional[DataInitArgType] = None,
                 device: Optional[str] = None,
                 ):
        if use_bias is None:
            use_bias = normalizer is None
        linear_kwargs = dict(
            use_bias=False,
            weight_norm=weight_norm,
            weight_init=weight_init,
            data_init=data_init,
            device=device,
        )
        modules = [Linear(in_features, out_features, **linear_kwargs)
                   for _ in range(n_partitions)]
        self_module = (Linear(in_features, out_features, **linear_kwargs)
                       if use_self_loop else None)

        if use_bias:
            out_dup = (len(modules) + int(use_self_loop)
                       if merge_mode == 'concat' else 1)
            bias_shape = [out_features * out_dup]
            bias_store = SimpleParamStore(
                bias_shape, initializer=bias_init, device=device)
        else:
            bias_store = None

        if normalizer is not None:
            normalizer = get_layer_from_layer_or_factory(
                'normalizer', normalizer, args=(out_features,))

        if activation is not None:
            activation = get_layer_from_layer_or_factory('activation', activation)

        super().__init__(
            modules=modules, self_module=self_module, self_weight=self_weight,
            bias_store=bias_store, normalizer=normalizer, activation=activation,
            merge_mode=merge_mode,
        )

