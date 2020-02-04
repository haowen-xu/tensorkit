import unittest

import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class UtilsTestCase(unittest.TestCase):

    def test_flatten_nested_layers(self):
        layers = [tk.layers.Linear(5, 5) for _ in range(5)]
        layers2 = tk.layers.flatten_nested_layers([
            layers[0], layers[1:2], [layers[2], [layers[3], layers[4]]]
        ])
        self.assertListEqual(layers2, layers)

        with pytest.raises(TypeError,
                           match='`nested_layers` is not a nested list '
                                 'of layers.'):
            _ = tk.layers.flatten_nested_layers([1])

        with pytest.raises(TypeError,
                           match='`nested_layers` is not a nested list '
                                 'of layers.'):
            _ = tk.layers.flatten_nested_layers({'a': layers[0]})

        with pytest.raises(TypeError,
                           match='`nested_layers` is not a nested list '
                                 'of layers.'):
            _ = tk.layers.flatten_nested_layers('')

    def test_get_activation_class(self):
        x = T.random.randn([2, 3, 4])

        for origin_name, factory, args, kwargs, expected in [
                    ('Linear', None, None, None, None),
                    ('ReLU', tk.layers.ReLU, (), {}, T.nn.relu(x)),
                    ('Leaky_ReLU', tk.layers.LeakyReLU, (), {}, T.nn.leaky_relu(x)),
                    ('Leaky_ReLU', tk.layers.LeakyReLU, (0.2,), {}, T.nn.leaky_relu(x, 0.2)),
                    ('Leaky_ReLU', tk.layers.LeakyReLU, (), {'negative_slope': 0.2}, T.nn.leaky_relu(x, 0.2)),
                    ('Sigmoid', tk.layers.Sigmoid, (), {}, T.nn.sigmoid(x)),
                    ('Tanh', tk.layers.Tanh, (), {}, T.tanh(x)),
                ]:
            name_candidates = (None,) if origin_name is None else (
                origin_name,
                origin_name.lower(),
                origin_name.replace('_', ''),
                origin_name.replace('_', '').lower()
            )
            for name in name_candidates:
                err_msg = f'{name}, {factory}, {args}, {kwargs}, {expected}'
                self.assertIs(tk.layers.get_activation_class(name), factory)
                if factory is not None:
                    assert_allclose(factory(*args, **kwargs)(x), expected, err_msg=err_msg)

        # unsupported activation
        with pytest.raises(ValueError, match='Unsupported activation: invalid'):
            _ = tk.layers.get_activation_class('invalid')
