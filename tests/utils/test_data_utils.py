import numpy as np
import pytest

import tensorkit as tk
from tensorkit import tensor as T
from tests.helper import *
from tests.ops import *


class DataUtilsTestCase(TestCase):

    def test_channel_from_last_to_first_nd(self):
        for spatial_ndims in (1, 2, 3):
            bad_input = np.random.randn(*([7, 8, 9, 10][:spatial_ndims + 1]))
            last_to_first = getattr(
                tk.utils,
                f'numpy_channel_from_last_to_first{spatial_ndims}d'
            )
            first_to_last = getattr(
                tk.utils,
                f'numpy_channel_from_first_to_last{spatial_ndims}d'
            )
            last_to_default = getattr(
                tk.utils,
                f'numpy_channel_from_last_to_default{spatial_ndims}d'
            )
            default_to_last = getattr(
                tk.utils,
                f'numpy_channel_from_default_to_last{spatial_ndims}d'
            )

            for op in (first_to_last, last_to_first):
                with pytest.raises(ValueError,
                                   match=f'`input` is expected to be at least '
                                         f'{spatial_ndims + 2}d'):
                    _ = op(bad_input)

            for batch_shape in ([5], [2, 3]):
                x = np.random.randn(*(
                    batch_shape + [7, 8, 9, 10][:spatial_ndims + 1]))  # assume x is channel last
                y = last_to_first(x)
                assert_allclose(y, channel_to_first_nd(x, spatial_ndims))
                assert_allclose(first_to_last(y), channel_to_last_nd(y, spatial_ndims))

                if T.IS_CHANNEL_LAST:
                    assert_allclose(last_to_default(x), x)
                    assert_allclose(default_to_last(x), x)
                else:
                    assert_allclose(last_to_default(x), y)
                    assert_allclose(default_to_last(y), x)
