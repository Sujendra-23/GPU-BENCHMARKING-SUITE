#!/usr/bin/env bash
# Compiles matmul_ref.c - the exact CPU reference used to correctness-check
# the CUDA and OpenCL kernels - to WebAssembly, so dashboard/ can run the
# same check live in a browser. Requires the Emscripten SDK (emcc):
# https://emscripten.org/docs/getting_started/downloads.html
#
#   brew install emscripten   (or install via emsdk)
#
# Run from anywhere; output lands in dashboard/wasm/.
set -euo pipefail
cd "$(dirname "$0")"

OUT_DIR="../../dashboard/wasm"
mkdir -p "$OUT_DIR"

emcc -O3 matmul_ref.c \
  -o "$OUT_DIR/matmul_ref.js" \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=createMatmulRefModule \
  -s EXPORTED_FUNCTIONS=_matmul_cpu_reference,_malloc,_free \
  -s EXPORTED_RUNTIME_METHODS=HEAPF32,ccall,cwrap \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s ENVIRONMENT=web \
  -s SINGLE_FILE=0

echo "wrote $OUT_DIR/matmul_ref.js (+ matmul_ref.wasm)"
