from itertools import product

import mltk
import numpy as np
import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *


class TensorStreamTestCase(TestCase):

    def test_TensorStream(self):
        x = np.random.randn(17, 3, 4)
        y = np.random.randn(17, 5)
        source = mltk.DataStream.arrays(
            [x, y], batch_size=3, random_state=np.random.RandomState())

        # test tensor stream
        for device in [None, T.CPU_DEVICE]:
            stream = tk.utils.as_tensor_stream(source, device=device)
            self.assertIsInstance(stream, tk.utils.TensorStream)
            self.assertEqual(stream.device, device or T.current_device())

            for attr in ('batch_size', 'array_count', 'data_shapes',
                         'data_length', 'random_state'):
                self.assertEqual(getattr(stream, attr), getattr(source, attr))

            out_x, out_y = stream.get_arrays()
            assert_allclose(out_x, x, rtol=1e-4, atol=1e-6)
            assert_allclose(out_y, y, rtol=1e-4, atol=1e-6)

            for batch_x, batch_y in stream:
                self.assertIsInstance(batch_x, T.Tensor)
                self.assertEqual(T.get_device(batch_x), device or T.current_device())
                self.assertIsInstance(batch_y, T.Tensor)
                self.assertEqual(T.get_device(batch_y), device or T.current_device())

            # test copy
            for device2 in [None, T.CPU_DEVICE]:
                kwargs = {'device': device2} if device2 is not None else {}
                stream2 = stream.copy(**kwargs)
                self.assertIs(stream2.source, stream.source)
                self.assertEqual(stream2.device, device2 or stream.device)

        # test prefetch
        stream = tk.utils.as_tensor_stream(source, prefetch=3)
        self.assertIsInstance(stream.source, tk.utils.TensorStream)

        out_x, out_y = stream.get_arrays()
        assert_allclose(out_x, x, rtol=1e-4, atol=1e-6)
        assert_allclose(out_y, y, rtol=1e-4, atol=1e-6)
