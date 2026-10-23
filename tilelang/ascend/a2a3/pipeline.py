"""A2/A3 lowering entry points.

The variant is registered before its native passes are ported so target
selection can be verified independently. Never lower it through the A5 passes.
"""

from __future__ import annotations

from tilelang import tvm
from tvm.ir import CallingConv

from tilelang.backend.pass_pipeline import PassPipeline


def get_device_kernel(mod):
    """Return the single exported A2/A3 device kernel without choosing by order."""
    functions = list(mod.functions.values())
    if len(functions) != 1 or not isinstance(functions[0], tvm.tirx.PrimFunc):
        raise ValueError("A2/A3 lowering requires exactly one exported device PrimFunc; multi-function modules are not supported yet.")
    func = functions[0]
    if not func.attrs or not func.attrs.get("global_symbol"):
        raise ValueError("A2/A3 device kernel requires a global_symbol export.")
    if int(func.attrs.get("calling_conv", CallingConv.DEFAULT)) != int(CallingConv.DEVICE_KERNEL_LAUNCH):
        raise ValueError("A2/A3 lowered kernel must use DEVICE_KERNEL_LAUNCH; explicit host calling conventions are unsupported.")
    return func


def _extract_parameters(mod):
    # Delay the engine import until compilation to keep backend registration
    # independent of the engine's import of BackendContext.
    from tilelang.engine.lower import extrac_params

    return extrac_params(get_device_kernel(mod))


def _not_implemented(_mod, _target):
    raise NotImplementedError("A2/A3 Ascend lowering is not implemented yet")


ASCENDC_PIPELINE = PassPipeline("ascend", _not_implemented, parameter_extractor=_extract_parameters)
PTO_PIPELINE = PassPipeline("ascend", _not_implemented)
