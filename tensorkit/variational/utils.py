
def _require_multi_samples(algorithm, axis):
    if axis is None:
        raise ValueError(
            f'{algorithm} requires to take multiple samples of the latent '
            f'variables, thus the `axis` argument must be specified'
        )
