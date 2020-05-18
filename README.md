## TensorKit

[![](https://github.com/haowen-xu/tensorkit/workflows/unittest/badge.svg?branch=master)](https://github.com/haowen-xu/tensorkit/actions)
[![](https://coveralls.io/repos/github/haowen-xu/tensorkit/badge.svg?branch=master)](https://coveralls.io/github/haowen-xu/tensorkit?branch=master)

### Dependencies

This package has been tested with the following dependencies:

* Python: 3.6/3.8
* PyTorch: 1.5.0

### Installation

```bash
pip install git+https://github.com/haowen-xu/ml-essentials.git
pip install git+https://github.com/haowen-xu/tensorkit.git
```

You may also need to install [graph-tool](https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions) 
to enable certain utilities of graph neural networks.

### Examples

* Classification:
   * MLP: [classification/mlp.py](tensorkit/examples/classification/mlp.py)
   * ResNet: [classification/resnet.py](tensorkit/examples/classification/resnet.py)
* Auto Encoders:
   * VAE: [auto_encoders/vae.py](tensorkit/examples/auto_encoders/vae.py)
   * ResNet VAE: [auto_encoders/vae_resnet.py](tensorkit/examples/auto_encoders/vae_resnet.py)
   * VAE with RealNVP posterior: [auto_encoders/vae_realnvp_posterior.py](tensorkit/examples/auto_encoders/vae_realnvp_posterior.py)
 
