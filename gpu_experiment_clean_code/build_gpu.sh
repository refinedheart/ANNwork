#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
NVCC_BIN="${NVCC:-nvcc}"
OUT="${ANN_GPU_OUT:-$ROOT_DIR/gpu_main}"

ARCH_FLAGS=()
if [[ -n "${ANN_GPU_ARCH:-}" ]]; then
    ARCH_FLAGS=(-arch="$ANN_GPU_ARCH")
fi

"$NVCC_BIN" \
    -O3 \
    -std=c++17 \
    --use_fast_math \
    -lineinfo \
    "${ARCH_FLAGS[@]}" \
    "$ROOT_DIR/gpu_ann.cu" \
    "$ROOT_DIR/gpu_main.cu" \
    -o "$OUT"

echo "built $OUT"
