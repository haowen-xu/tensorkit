FROM haowenxu/ml-runtime2:cpu-py3.6

# Invalidate any cache
ARG CACHEBURST=0

# set the work-dir ro /prj
WORKDIR /prj

# Install the dependencies
RUN pip install -r requirements-dev.txt

# Report the package versions
RUN python --version && \
    pytest --version && \
    coveralls --version

# Test with PyTorch 1.3.1
RUN pip install "torch==1.3.1"

RUN TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=none TENSORKIT_VALIDATE_TENSORS=false \
    pytest --cov=tensorkit --cov-append
RUN TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=none TENSORKIT_VALIDATE_TENSORS=true \
    pytest --cov=tensorkit --cov-append
RUN TENSORKIT_BACKEND=PyTorch TENSORKIT_JIT_MODE=all TENSORKIT_VALIDATE_TENSORS=false \
    pytest --cov=tensorkit --cov-append


# submit the result to coveralls.io
RUN coveralls
