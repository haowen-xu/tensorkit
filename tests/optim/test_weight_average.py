import copy

import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


def stepwise_average_check(ctx, factory, update_fn, get_fn):
    def clone_state(val):
        if isinstance(val, dict):
            return {k: clone_state(v) for k, v in val.items()}
        elif isinstance(val, list):
            return [clone_state(v) for v in val]
        elif isinstance(val, (T.Tensor, T.Variable)):
            return T.copy(val)
        elif isinstance(val, np.ndarray):
            return np.copy(val)
        else:
            return copy.copy(val)

    T.random.seed(1234)
    weights = [
        T.variable(shape=[4], initializer=tk.init.zeros, requires_grad=False),
        T.variable(shape=[3], initializer=tk.init.zeros, requires_grad=False),
    ]
    answers = [clone_state(w) for w in weights]
    inputs_1 = T.random.randn([7, 4])
    inputs_2 = T.random.randn([7, 3])

    # do a scan
    avg = factory(weights)
    the_states = []
    the_outputs = []
    num_updates = 0

    for batch_vals in zip(inputs_1, inputs_2):
        for weight, val in zip(weights, batch_vals):
            T.assign(weight, val)

        the_states.append(clone_state(avg.get_state_dict()))
        avg.update()

        with avg.temporarily_commit():
            the_outputs.extend(clone_state(w) for w in weights)
            for i, val in enumerate(batch_vals):
                answers[i] = update_fn(answers[i], val, num_updates)
            num_updates += 1
            for weight, ans in zip(weights, answers):
                assert_allclose(weight, get_fn(ans, num_updates), rtol=1e-4, atol=1e-6)

        for weight, val in zip(weights, batch_vals):
            assert_allclose(weight, val, rtol=1e-4, atol=1e-6)

    # test enabled = False
    avg = factory(weights, enabled=False)
    for x1, x2, state, output in zip(inputs_1, inputs_2, the_states, the_outputs):
        batch_vals = [x1, x2]
        for weight, val in zip(weights, batch_vals):
            T.assign(weight, val)
        avg.update()

    avg.commit()  # should still affect weights even if enabled is False
    for avg_val in avg.get_state_dict()['averages']:
        assert_allclose(avg_val, T.zeros_like(avg_val), rtol=1e-4, atol=1e-6)
    for weight in weights:
        assert_allclose(weight, T.zeros_like(weight), rtol=1e-4, atol=1e-6)

    # do another scan using backup states
    avg = factory(weights, enabled=False)
    avg.set_enabled(True)
    for x1, x2, state, output in zip(inputs_1, inputs_2, the_states, the_outputs):
        batch_vals = [x1, x2]
        for weight, val in zip(weights, batch_vals):
            T.assign(weight, val)

        avg.set_state_dict(state)
        avg.update()

        with avg.temporarily_commit():
            the_outputs.extend(clone_state(w) for w in weights)
        for weight, val in zip(weights, batch_vals):
            assert_allclose(weight, val, rtol=1e-4, atol=1e-6)

    # try set bad state
    avg = factory(weights)
    state = dict(avg.get_state_dict())
    state['averages'] = []
    with pytest.raises(ValueError, match='Bad state'):
        avg.set_state_dict(state)


def full_scan_average_check(ctx, factory, input_x, expected):
    weight = T.variable(T.shape(input_x)[1:], initializer=tk.init.zeros,
                        requires_grad=False)
    avg = factory([weight])
    for x in input_x:
        T.assign(weight, x)
        avg.update()
    avg.commit()
    assert_allclose(weight, expected, atol=1e-4, rtol=1e-6)


class WeightAveragingTestCase(TestCase):

    def test_MeanAveraging(self):
        # step-wise check
        factory = tk.optim.WeightMeanAveraging

        def update_fn(old_val, new_val, num_updates):
            return (old_val * num_updates + new_val) / (num_updates + 1.)

        def get_fn(val, num_updates):
            return val

        stepwise_average_check(self, factory, update_fn, get_fn)

        # overall check
        input_x = T.random.randn([7, 4])
        full_scan_average_check(
            self, factory, input_x, T.reduce_mean(input_x, axis=[0]))

    def test_MovingAveraging(self):
        # step-wise check
        for decay in (0.9, 0.99):
            for zero_debias in (True, False):
                factory = lambda weights, **kwargs: tk.optim.WeightMovingAveraging(
                    weights, decay=decay, zero_debias=zero_debias, **kwargs)

                def update_fn(old_val, new_val, num_updates):
                    return decay * old_val + (1. - decay) * new_val

                if zero_debias:
                    def get_fn(val, num_updates):
                        if num_updates > 0:
                            return val / (1. - decay ** num_updates)
                        else:
                            return val
                else:
                    def get_fn(val, num_updates):
                        return val

                stepwise_average_check(self, factory, update_fn, get_fn)

        # overall check
        input_x = T.expand(T.random.randn([4]), [7, 4])
        factory = lambda weights, **kwargs: tk.optim.WeightMovingAveraging(
            weights, decay=0.9, zero_debias=True, **kwargs)
        full_scan_average_check(self, factory, input_x, input_x[0])
