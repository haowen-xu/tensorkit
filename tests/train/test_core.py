import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from mltk import SimpleStatefulObject

import tensorkit as tk
from tests.helper import *


class TorchCheckpointTestCase(TestCase):

    def test_invalid_type(self):
        with pytest.raises(TypeError,
                           match=r'Object must be a :class:`StatefulObject`, '
                                 r'or has `state_dict\(\)` and '
                                 r'`load_state_dict\(\)` methods: got 123'):
            _ = tk.train.Checkpoint(obj=123)

    def test_save_restore(self):
        x = torch.from_numpy(np.random.normal(size=[2, 5]).astype(np.float32))

        with TemporaryDirectory() as temp_dir:
            root_dir = os.path.join(temp_dir, 'ckpt')

            # test save
            layer = torch.nn.Linear(5, 3)
            optimizer = tk.optim.Adam(tk.layers.get_parameters(layer))

            obj = SimpleStatefulObject()
            obj.value = 123456
            ckpt = tk.train.Checkpoint(obj=obj, optimizer=optimizer, layer=layer)
            ckpt.save(root_dir)

            # test restore
            layer2 = torch.nn.Linear(5, 3)
            optimizer2 = tk.optim.Adam(tk.layers.get_parameters(layer2))
            obj2 = SimpleStatefulObject()
            ckpt2 = tk.train.Checkpoint(obj=obj2, optimizer=optimizer2, layer=layer2)
            ckpt2.restore(root_dir)

            # todo: check the state of the optimizer

            # compare two objects
            out = layer(x)
            out2 = layer2(x)
            self.assertTrue(torch.allclose(out2, out))
            self.assertEqual(obj2.value, 123456)

            # test partial restore
            layer3 = torch.nn.Linear(5, 3)
            ckpt3 = tk.train.Checkpoint(layer=layer3)
            ckpt3.restore(root_dir)
            self.assertTrue(torch.allclose(layer3(x), out))

            # test restore error
            ckpt4 = tk.train.Checkpoint(layer=layer3, xyz=SimpleStatefulObject())
            with pytest.raises(ValueError,
                               match=f'Key \'xyz\' does not exist in '
                                     f'the state dict recovered from: '
                                     f'{root_dir}'):
                ckpt4.restore(root_dir)


