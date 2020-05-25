#!/bin/bash

FAST_TEST="${FAST_TEST:-}"

if [ "${JIT_MODE}" == "" ]; then
  JIT_MODE="${TENSORKIT_JIT_MODE:-}"
fi

if [ "${VALIDATE_TENSORS}" == "" ]; then
  VALIDATE_TENSORS="${TENSORKIT_VALIDATE_TENSORS:-}"
fi

docker build -t tensorkit . && \
docker run -ti --rm \
  -v "$(pwd)":/prj:ro \
  -w /prj \
  -e FAST_TEST="${FAST_TEST}" \
  -e TENSORKIT_JIT_MODE="${JIT_MODE}" \
  -e TENSORKIT_VALIDATE_TENSORS="${VALIDATE_TENSORS}" \
  tensorkit "$@"
