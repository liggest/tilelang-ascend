#!/usr/bin/env bash

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Build the original fixed flash-attention standalone libtorch demo."
    echo "Set CMAKE_PREFIX_TORCH_NPU to a real libtorch_npu C++ installation,"
    echo "TORCH_NPU_SOURCE_DIR to its source root, and a dedicated TILELANG_CACHE_DIR."
    echo "CMAKE_PREFIX_TORCH may optionally select the matching libtorch installation."
    exit 0
fi

set -x
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${CMAKE_PREFIX_TORCH_NPU:-}" || -z "${TORCH_NPU_SOURCE_DIR:-}" ]]; then
    echo "Set CMAKE_PREFIX_TORCH_NPU and TORCH_NPU_SOURCE_DIR to real matching libtorch_npu development dependencies" >&2
    exit 1
fi
if [[ ! -d "$CMAKE_PREFIX_TORCH_NPU" || ! -d "$TORCH_NPU_SOURCE_DIR" ]]; then
    echo "The requested libtorch_npu development prefix/source directory does not exist" >&2
    exit 1
fi
if [[ -z "${TILELANG_CACHE_DIR:-}" ]]; then
    echo "Set a dedicated TILELANG_CACHE_DIR before compiling libop.so" >&2
    exit 1
fi
CMAKE_PREFIX_TORCH="${CMAKE_PREFIX_TORCH:-$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')}"

# Resolve the real C++ development dependencies before spending an SDK compile.
# A Python TorchNPU wheel is not a substitute for the original libtorch_npu CMake package.
cmake -S "$HERE" -B "$HERE/build" \
      -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_TORCH};${CMAKE_PREFIX_TORCH_NPU}" \
      -DTORCH_NPU_SOURCE_DIR="${TORCH_NPU_SOURCE_DIR}" || exit $?
python "$HERE/prepare.py" || { echo "Failed to prepare libop.so"; exit 1; }
cmake --build "$HERE/build" --config Release
