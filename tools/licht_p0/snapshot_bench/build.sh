#!/usr/bin/env bash
# Direct nvcc build for the training-snapshot prototype. Not wired into CMake.
set -euo pipefail
cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"
ARCH_FLAG="${ARCH_FLAG:--arch=native}"

exec "$NVCC" -O2 -std=c++20 $ARCH_FLAG snapshot_bench.cu -o snapshot_bench "$@"
