import mltk
import tensorkit as tk
from tensorkit import tensor as T
from tensorkit.examples import utils


class Config(mltk.Config):
    # initialization parameters
    init_batch_count: int = 10

    # train parameters
    max_epoch: int = 300
    batch_size: int = 64
    initial_lr: float = 0.001
    lr_anneal_ratio: float = 0.1
    lr_anneal_epochs: int = 100

    # test parameters
    test_batch_size: int = 256


def main(exp: mltk.Experiment[Config]):
    # prepare the data
    train_stream, _, test_stream = utils.get_mnist_streams(
        batch_size=exp.config.batch_size,
        test_batch_size=exp.config.test_batch_size,
        val_batch_size=exp.config.test_batch_size,
        x_range=(-1., 1.),
    )

    # build the network
    net: T.Module = tk.layers.SequentialBuilder(train_stream.data_shapes[0]). \
        set_args('res_block2d',
                 kernel_size=3,
                 activation=tk.layers.LeakyReLU,
                 normalizer=tk.layers.BatchNorm2d,
                 dropout=0.5,
                 data_init=tk.init.StdDataInit()). \
        res_block2d(16). \
        res_block2d(32, stride=2). \
        res_block2d(32). \
        res_block2d(64, stride=2). \
        res_block2d(64). \
        global_avg_pool2d(). \
        linear(10). \
        log_softmax(). \
        build()

    # initialize the network with first few batches of train data
    init_x, _ = train_stream.get_arrays(max_batch=exp.config.init_batch_count)
    _ = net(T.as_tensor(init_x))
    mltk.print_with_time('Network initialized')

    # we have initialized the network, now we can compile the net with JIT engine
    net = tk.layers.jit_compile(net)
    mltk.print_with_time('Network compiled with JIT')

    # the train, test and validate functions
    def train_step(x, y):
        logits = net(x)
        loss = T.nn.cross_entropy_with_logits(logits, y, reduction='mean')
        return {'loss': loss}

    def eval_step(x, y):
        with tk.layers.scoped_eval_mode(net), T.no_grad():
            logits = net(x)
            acc = utils.calculate_acc(logits, y)
        return {'acc': acc}

    # build the optimizer and the train loop
    loop = mltk.TrainLoop(max_epoch=exp.config.max_epoch)
    optimizer = tk.optim.Adam(tk.layers.get_parameters(net))
    lr_scheduler = tk.optim.lr_scheduler.AnnealingLR(
        loop=loop,
        optimizer=optimizer,
        initial_lr=exp.config.initial_lr,
        ratio=exp.config.lr_anneal_ratio,
        epochs=exp.config.lr_anneal_epochs
    )

    # run test after every 10 epochs
    loop.run_after_every(
        lambda: loop.test().run(eval_step, test_stream),
        epochs=10,
    )

    # train the model
    tk.layers.set_train_mode(net, True)
    utils.fit_model(loop=loop, optimizer=optimizer, fn=train_step,
                    stream=train_stream)

    # do the final test with the best network parameters (according to validation)
    results = mltk.TestLoop().run(eval_step, test_stream)


if __name__ == '__main__':
    with mltk.Experiment(Config) as exp:
        with T.use_device(T.first_gpu_device()):
            main(exp)
