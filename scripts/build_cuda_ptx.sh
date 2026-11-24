#!/usr/bin/env bash
set -euo pipefail

# Compile src/integration/cuda/kernels.cu to forward-compatible PTX
# (compute_70: Volta, Turing, Ampere, Ada, Hopper, Blackwell) so Dioptrin
# can load CUDA kernels directly via the NVIDIA display driver without NVRTC.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src/integration/cuda/kernels.cu"
OUT="$ROOT/src/integration/cuda/kernels.ptx"
ARCH="${DIOPTRIN_CUDA_PTX_ARCH:-compute_70}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found on PATH" >&2
  exit 1
fi

SHA="$(sha256sum "$SRC" | awk '{print $1}')"
TMP="$(mktemp "${TMPDIR:-/tmp}/dioptrin_kernels.XXXXXX.ptx")"
trap 'rm -f "$TMP"' EXIT

nvcc --ptx \
  --std=c++17 \
  --fmad=false \
  --gpu-architecture="$ARCH" \
  -o "$TMP" \
  "$SRC"

{
  echo "// dioptrin-kernels-sha256: $SHA"
  echo "// dioptrin-ptx-arch: $ARCH"
  cat "$TMP"
} > "$OUT"

echo "Wrote $OUT ($(wc -c < "$OUT") bytes, arch=$ARCH, sha256=$SHA)"
