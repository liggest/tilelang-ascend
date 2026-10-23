"""A2/A3 source generation through the shared device-codegen interface."""

from __future__ import annotations

from tilelang import tvm
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.instrumentation import run_codegen_with_instrumentation

from .platform import resolve_a2a3_platform


_ASCENDC_NATIVE = "target.build.tilelang_ascend_a2a3"


def _build_ascendc(mod, target):
    platform = resolve_a2a3_platform(target)
    return run_codegen_with_instrumentation(
        _ASCENDC_NATIVE,
        mod,
        target,
        lambda: tvm.ffi.get_global_func(_ASCENDC_NATIVE)(mod, target, platform),
        role="device",
    )


def _pto_not_implemented(_mod, _target):
    raise NotImplementedError("A2/A3 PTO code generation is not implemented yet")


# The TA native builder emits source. The shared Cython execution backend compiles
# that source separately, so both device-codegen modes use the same builder.
ASCENDC_CODEGEN = DeviceCodegen("ascend_a2a3", build=_build_ascendc, build_without_compile=_build_ascendc)
PTO_CODEGEN = DeviceCodegen("pto_a2a3", build=_pto_not_implemented, build_without_compile=_pto_not_implemented)
