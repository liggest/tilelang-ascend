"""Automatic tensor allocation for the TA AscendC Cython calling convention.

Source: TileLang-Ascend 0deb8b93da11b7f0becbcded732d9633b7ca4e34,
``jit/adapter/base.py`` and ``jit/adapter/cython/cython_wrapper.pyx``.
"""

from __future__ import annotations

from collections.abc import Sequence

from tilelang import tvm
from tvm import tirx


def normalize_memory_indices(indices: list[int] | int | None, num_params: int, name: str) -> list[int]:
    """Apply TA's None/int/list and negative-index rules without mutating callers."""
    if indices is None:
        return []
    if isinstance(indices, int):
        indices = [indices]
    elif isinstance(indices, list):
        indices = indices.copy()
    else:
        raise ValueError(f"{name} should be a list of integers")
    for position, index in enumerate(indices):
        if not isinstance(index, int):
            raise ValueError(f"{name} should be a list of integers")
        if index >= num_params or index < -num_params:
            raise ValueError(f"{name} should be an integer between {-num_params} and {num_params - 1}")
        if index < 0:
            indices[position] = num_params + index
    return indices


def resolve_allocation_shape(
    shape: Sequence,
    symbolic_map: dict,
    symbolic_sources: dict,
    tensors: list,
    compiler_workspace: bool = False,
) -> list[int]:
    """Resolve source output/workspace expressions using main's full tensor slots.

    TA's declared workspace/output path supports arbitrary constant-resolvable
    PrimExpr shapes. Its compiler-derived GM path explicitly does not support
    dynamic shapes. Source symbols are matched by name because cached parameter
    reflection and the original frontend IR may own different Var handles.
    """
    analyzer = tvm.arith.Analyzer()
    values = {}
    for variable, source in symbolic_map.items():
        candidates = symbolic_sources.get(str(variable), [source])
        for reference, buffer_index, dimension, stride_scale in candidates:
            tensor = tensors[buffer_index]
            if tensor is not None:
                value = tensor.shape[dimension] if reference == 0 else tensor.stride(dimension) * stride_scale
                values[str(variable)] = int(value)
                break
    result = []
    for dimension in shape:
        if isinstance(dimension, (int, tirx.IntImm)):
            result.append(int(dimension))
        elif isinstance(dimension, tirx.PrimExpr):
            if compiler_workspace:
                raise NotImplementedError(
                    "A2A3 compiler-derived GM workspaces require static shapes; dynamic auto_gm_idx is unsupported in TA"
                )
            substitutions = {}
            for variable in tirx.analysis.undefined_vars(dimension):
                if str(variable) not in values:
                    raise KeyError(f"Unfounded symbolic var: {variable}")
                substitutions[variable] = tirx.IntImm(variable.dtype, values[str(variable)])
            value = analyzer.simplify(tirx.stmt_functor.substitute(dimension, substitutions))
            if not isinstance(value, tirx.IntImm):
                raise ValueError(f"Shape not constant: {dimension}")
            result.append(int(value))
        else:
            raise TypeError(f"Unsupported shape dim type: {type(dimension)} ({dimension})")
    return result


def configure_void_call(library) -> None:
    """Use the self-contained TA host launcher's void C return convention."""
    library.call.restype = None


def dump_runtime_signature(func: tirx.PrimFunc) -> str:
    """Persist the final parameter/buffer contract without compiler-only IR nodes."""
    symbol = func.attrs.get("global_symbol") if func.attrs else None
    if symbol is None:
        raise ValueError("A2A3 runtime signature requires an exported global_symbol")
    signature = tirx.PrimFunc(func.params, tirx.Evaluate(0), func.ret_type, func.buffer_map).with_attr("global_symbol", symbol)
    for key in ("calling_conv", "auto_gm_indices"):
        if key in func.attrs:
            signature = signature.with_attr(key, func.attrs[key])
    return tvm.ir.save_json(signature)


def load_runtime_signature(text: str) -> tirx.PrimFunc:
    """Restore a signature using TVM reflection, never Python pickle or eval."""
    signature = tvm.ir.load_json(text)
    if not isinstance(signature, tirx.PrimFunc) or not signature.attrs or not signature.attrs.get("global_symbol"):
        raise ValueError("Invalid A2A3 cached runtime signature")
    return signature
