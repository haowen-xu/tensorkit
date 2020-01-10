from tensorkit import backend as Z

__all__ = [
    'int_dtypes', 'float_dtypes', 'number_dtypes',
    'n_samples',
]


# Not all integer or float dtypes are listed as follows.  Just some commonly
# used dtypes, enough for test.
int_dtypes = (Z.int32, Z.int64)
float_dtypes = (Z.float32, Z.float64)
number_dtypes = int_dtypes + float_dtypes

# The number of samples to take for tests which requires random samples.
n_samples = 10000
