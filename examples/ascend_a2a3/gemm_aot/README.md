# TileLang-Ascend AOT Operator Example

This directory provides an example of implementing an **Ahead-of-Time (AOT)** compiled operator, using **GEMM (General Matrix Multiplication)** as a demonstration.
It shows how an operator can be generated, compiled, and tested through a simple automated workflow.

## Overview

The example demonstrates the full workflow of building and testing an AOT-compiled operator:
1. **Operator generation** (from Python to device source)
2. **Compilation** of the generated operator code into `kernel_lib.so`
3. **Validation** of results through test scripts

Generation and compilation are both driven by the framework: `example_gemm.py`
lowers the kernel and compiles it through the same `LibraryGenerator` used by the
JIT path, so the build stays consistent with the framework's compiler flags
across CANN versions (no standalone build script to keep in sync).

---

## Directory Structure

```
gemm_aot/
├── example_gemm.py           # Lowers the GEMM kernel and compiles it into kernel_lib.so
├── test_example_gemm.py      # Script to test the compiled AOT operator
├── run_example_gemm_aot.sh   # One-click script to run the full example
└── README.md
```

`example_gemm.py` accepts `--target {ascendc,pto}` (default `ascendc`),
`--platform` (default auto-detected) and `-o/--output` (default
`./kernel_lib.so`). The generated `kernel_lib.so` is not checked into the repo.

---

## Running the Example

We assume that you have already
[installed TileLang with Ascend support](../../../README.md#installation).

To run the complete AOT GEMM example:

```bash
cd examples/ascend_a2a3/gemm_aot
TILELANG_CACHE_DIR=/tmp/my-dedicated-a2a3-aot-cache bash run_example_gemm_aot.sh
```

## Modern A2/A3 integration

The complete original GEMM/factory, default 8192x1024x8192 dimensions, seed 42
and 1e-2 numerical tolerances are retained. Import and --help do not clear cache,
compile, allocate NPU tensors or launch. The real clear requires a dedicated
explicit TILELANG_CACHE_DIR matching the actual resolved cache descriptor.
The current CompileConfig is resolved once before lower/driver use; target.model
stores A2/A3 and keys choose AscendC/PTO. This raw AOT entry preserves its original
bare O2 compiler fallback, separately from public JIT's derived defaults.

Explicit A5 selection remains a clear outside-A2/A3 diagnostic; this namespace
does not port the modern A5 execution path. The numerical reader declares the
actual void call ABI (three static buffer pointers plus the stream). Source/SDK
validation is separate from the original NPU comparison; compilation or copying
the library alone is not numerical/performance evidence.

On an SDK-only host, pass `--platform A2` or `--platform A3` (or set
`TL_PLATFORM`) to build without device detection. The original reader still
requires a real NPU. Run the unchanged shell/explicit reader CLI for its
numerical check; importing its `test_` filename is deliberately side-effect free.
