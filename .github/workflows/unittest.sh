#!/bin/bash

set -e

# set the work-dir ro /prj
cd /prj

# Install the dependencies
pip install -r requirements-dev.txt

# Report the package versions
python --version
pytest --version
coveralls --version

# Test with PyTorch 1.5.0
pip install "torch==1.5.0"

export TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=none TENSORKIT_VALIDATE_TENSORS=false
pytest --cov=tensorkit --cov-append

export TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=none TENSORKIT_VALIDATE_TENSORS=true
pytest --cov=tensorkit --cov-append

export TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=all TENSORKIT_VALIDATE_TENSORS=false
pytest --cov=tensorkit --cov-append

# submit the result to coveralls.io
coveralls
