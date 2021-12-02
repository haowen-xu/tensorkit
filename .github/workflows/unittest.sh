#!/bin/bash

set -e

# set the work-dir ro /prj
cd /prj

# Install the dependencies
pip install -r requirements-dev.txt
pip install git+https://github.com/haowen-xu/ml-essentials.git

# Report the package versions
python --version
pytest --version
coveralls --version

# Test with PyTorch 1.10.0
pip install "torch==1.10.0" "torchvision==0.11.1"
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # to test deterministic behaviors
export CUDA_LAUNCH_BLOCKING=1  # make cuda behaviors more easily debugged

export TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=none TENSORKIT_VALIDATE_TENSORS=false
pytest --cov=tensorkit --cov-append

export TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=none TENSORKIT_VALIDATE_TENSORS=true
pytest --cov=tensorkit --cov-append

export TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=all TENSORKIT_VALIDATE_TENSORS=false
pytest --cov=tensorkit --cov-append

# submit the result to coveralls.io
coveralls
