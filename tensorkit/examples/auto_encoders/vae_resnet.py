from functools import partial
from typing import *

import mltk
import numpy as np
import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.examples import utils
from tensorkit.typing_ import TensorOrData


class Config(mltk.Config):
    # model parameters
    z_dim: int = 40
    kernel_size: int = 3

    # initialization parameters
    init_batch_count: int = 10

    # train parameters
    max_epoch: int = 1000
    batch_size: int = 128
    initial_lr: float = 0.001
    lr_anneal_ratio: float = 0.1
    lr_anneal_epochs: int = 300

    # evaluation parameters
    test_n_z: int = 500
    test_batch_size: int = 64


class VAE(tk.layers.BaseLayer):

    x_shape: List[int]
    config: Config

    def __init__(self, x_shape: Sequence[int], config: Config):
        super().__init__()
        self.x_shape = list(map(int, x_shape))
        self.config = config

        # common nn parameters
        layer_args = tk.layers.LayerArgs(). \
            set_args(['res_block2d', 'res_block_transpose2d'],
                     activation=tk.layers.LeakyReLU,
                     kernel_size=config.kernel_size,
                     padding='half')

        # nn for q(z|x)
        q_builder = tk.layers.SequentialBuilder(x_shape, layer_args=layer_args)
        q_builder.res_block2d(16). \
            res_block2d(32, stride=2). \
            res_block2d(32). \
            res_block2d(64, stride=2). \
            res_block2d(64)
        conv_out_shape = q_builder.out_shape  # note: we need this shape in the deconv
        q_builder.flatten()

        self.hx_for_qz = q_builder.build()
        self.qz_mean = q_builder.as_input().linear(config.z_dim).build()
        self.qz_logstd = q_builder.as_input().linear(config.z_dim).build()

        # nn for p(x|z)
        n_channels, _ = T.utils.split_channel_spatial_shape(self.x_shape)
        p_builder = tk.layers.SequentialBuilder(config.z_dim, layer_args=layer_args)
        self.px_logits = p_builder. \
            linear(int(np.prod(conv_out_shape))). \
            reshape(conv_out_shape). \
            res_block_transpose2d(64). \
            res_block_transpose2d(32, stride=2, output_size=[14, 14]). \
            res_block_transpose2d(32). \
            res_block_transpose2d(16, stride=2, output_size=[28, 28]). \
            conv2d(n_channels, kernel_size=1). \
            build(True)

    def initialize(self, x):
        x = T.as_tensor(x)
        _ = self.get_chain(x).vi.training.sgvb()  # trigger initialization
        tk.layers.jit_compile_children(self)
        _ = self.get_chain(x).vi.training.sgvb()  # trigger JIT

    def q(self,
          x: T.Tensor,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          n_z: Optional[int] = None) -> tk.BayesianNet:
        net = tk.BayesianNet(observed=observed)
        hx = self.hx_for_qz(T.cast(x, dtype=T.float32))
        z_mean = self.qz_mean(hx)
        z_logstd = self.qz_logstd(hx)
        z = net.add('z', tk.Normal(mean=z_mean, logstd=z_logstd, event_ndims=1),
                    n_samples=n_z)
        return net

    def p(self,
          observed: Optional[Mapping[str, TensorOrData]] = None,
          n_z: Optional[int] = None) -> tk.BayesianNet:
        net = tk.BayesianNet(observed=observed)

        # sample z ~ p(z)
        z = net.add('z', tk.UnitNormal([1, self.config.z_dim], event_ndims=1),
                    n_samples=n_z)
        x_logits = self.px_logits(z.tensor)
        x = net.add('x', tk.Bernoulli(logits=x_logits, event_ndims=3))
        return net

    def get_chain(self, x, n_z: Optional[int] = None):
        latent_axis = 0 if n_z is not None else None
        return self.q(x, n_z=n_z).chain(
            self.p, observed={'x': x}, n_z=n_z, latent_axis=latent_axis)


def main(exp: mltk.Experiment[Config]):
    # prepare the data
    train_stream, _, test_stream = utils.get_mnist_streams(
        batch_size=exp.config.batch_size,
        test_batch_size=exp.config.test_batch_size,
        x_range=(0., 1.),
        use_y=False,
        mapper=utils.BernoulliSampler().as_mapper(),
    )

    utils.print_experiment_summary(
        exp, train_stream=train_stream, test_stream=test_stream)

    # build the network
    vae: VAE = VAE(train_stream.data_shapes[0], exp.config)
    params, param_names = utils.get_params_and_names(vae)
    utils.print_parameters_summary(params, param_names)
    print('')

    # initialize the network with first few batches of train data
    [init_x] = train_stream.get_arrays(max_batch=exp.config.init_batch_count)
    vae.initialize(init_x)
    mltk.print_with_time('Network initialized')

    # define the train and evaluate functions
    def train_step(x):
        chain = vae.get_chain(x)
        loss = chain.vi.training.sgvb(reduction='mean')
        return {'loss': loss}

    def eval_step(x, n_z=exp.config.test_n_z):
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            chain = vae.get_chain(x, n_z=n_z)
            loss = chain.vi.training.sgvb(reduction='mean')
            nll = -chain.vi.evaluation.is_loglikelihood(reduction='mean')
        return {'elbo': loss, 'nll': nll}

    def plot_samples(epoch=None):
        epoch = epoch or loop.epoch
        with tk.layers.scoped_eval_mode(vae), T.no_grad():
            logits = vae.p(n_z=100)['x'].distribution.logits
            images = T.reshape(
                T.cast(T.clip(T.nn.sigmoid(logits) * 255., 0., 255.), dtype=T.uint8),
                [-1, 28, 28],
            )
        utils.save_images_collection(
            images=T.to_numpy(images),
            filename=exp.abspath(f'plotting/{epoch}.png'),
            grid_size=(10, 10),
        )

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=exp.config.max_epoch)
    loop.add_callback(mltk.callbacks.StopOnNaN())
    optimizer = tk.optim.Adam(tk.layers.iter_parameters(vae))
    lr_scheduler = tk.optim.lr_scheduler.AnnealingLR(
        loop=loop,
        optimizer=optimizer,
        initial_lr=exp.config.initial_lr,
        ratio=exp.config.lr_anneal_ratio,
        epochs=exp.config.lr_anneal_epochs
    )
    loop.run_after_every(
        lambda: loop.test().run(partial(eval_step, n_z=10), test_stream),
        epochs=10
    )
    loop.run_after_every(plot_samples, epochs=10)

    # train the model
    tk.layers.set_train_mode(vae, True)
    utils.fit_model(loop=loop, optimizer=optimizer, fn=train_step,
                    stream=train_stream)

    # do the final test
    results = mltk.TestLoop().run(eval_step, test_stream)
    plot_samples('final')


if __name__ == '__main__':
    with mltk.Experiment(Config) as exp:
        with T.use_device(T.first_gpu_device()):
            main(exp)
