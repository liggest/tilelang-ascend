"""A2/A3 codegen entry points, pending native codegen migration."""

from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen


def _not_implemented(_mod, _target):
    raise NotImplementedError("A2/A3 Ascend code generation is not implemented yet")


ASCENDC_CODEGEN = DeviceCodegen("ascend_a2a3", build=_not_implemented, build_without_compile=_not_implemented)
PTO_CODEGEN = DeviceCodegen("pto_a2a3", build=_not_implemented, build_without_compile=_not_implemented)
