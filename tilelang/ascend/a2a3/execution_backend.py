"""Both A2/A3 source variants use the shared Cython execution backend."""

from tilelang.backend.execution_backend import ExecutionBackendSpec

EXECUTION_BACKENDS = (ExecutionBackendSpec("cython"),)
