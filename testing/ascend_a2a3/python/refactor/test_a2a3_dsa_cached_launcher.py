"""CPU pointer/control contracts for the local DSA launcher; no kernel output."""

import ast
import ctypes
import gc
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[4] / "examples_experiment/dsa_sparse_finetune/example_indexer_bwd.py"


@pytest.fixture
def launcher_scope():
    # Read the actual example class without importing its JIT/CLI or any NPU
    # registration. Recording callables below never compute a golden output.
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {"_CachedLauncher", "_resolve_raw_stream"}
    ]
    scope = {"torch": torch, "ctypes": ctypes, "_npu_raw_stream": None, "_npu_raw_stream_attempted": False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return scope


class RecordingLibrary:
    def __init__(self):
        self.calls = []

    def call(self, *args):
        self.calls.append(tuple(value.value if hasattr(value, "value") else value for value in args))


class ValidatingKernel:
    def __init__(self):
        self.adapter = SimpleNamespace(
            lib=RecordingLibrary(),
            params=[None, None],
            result_idx=[],
            workspace_idx=[],
            auto_gm_idx=[],
            dynamic_symbolic_map={"rows": (0, 0, 0, 1)},
            dynamic_symbolic_sources={"rows": [(0, 0, 0, 1), (0, 1, 0, 1)]},
        )
        self.fallback_calls = []
        self.result = object()
        self.error = None

    def __call__(self, *args, **kwargs):
        self.fallback_calls.append((args, kwargs))
        if self.error:
            raise self.error
        return self.result


def test_contiguous_views_retain_storage_owners_and_refresh_stream(launcher_scope):
    args = (torch.empty(4, 3, 2, device="cpu"), torch.empty(4, 3, 2, device="cpu"))
    kernel = ValidatingKernel()
    launcher = launcher_scope["_CachedLauncher"](kernel, {0: "flat2d", 1: "flat2d"})
    assert launcher(*args, stream=37) is None
    refs = [weakref.ref(tensor) for tensor in launcher._args_refs]
    gc.collect()
    assert all(ref() is not None for ref in refs)
    assert len(refs) == 4
    first = kernel.adapter.lib.calls[0]
    assert first == (args[0].data_ptr(), args[1].data_ptr(), 12, 37)
    args[0].fill_(1)  # Content changes preserve the metadata/id fast path.
    assert launcher(*args, stream=41) is None
    assert kernel.adapter.lib.calls[-1] == (*first[:-1], 41)
    assert not kernel.fallback_calls


@pytest.mark.parametrize("descriptor", [(0, 0), (1, 0, 0, 1), (0, 0, 0, 2), (0, 99, 0, 1), (0, 0, 99, 1), (0, 0, -1, 1), (False, 0, 0, 1)])
def test_unsupported_descriptor_uses_complete_kernel(launcher_scope, descriptor):
    args = (torch.empty(4, 2, device="cpu"), torch.empty(4, 2, device="cpu"))
    kernel = ValidatingKernel()
    kernel.adapter.dynamic_symbolic_map["rows"] = descriptor
    launcher = launcher_scope["_CachedLauncher"](kernel)
    assert launcher(*args, stream=37) is kernel.result
    assert not kernel.adapter.lib.calls
    assert kernel.fallback_calls == [(args, {"stream": 37})]


def test_symbol_carrier_conflict_preserves_full_validator_exception(launcher_scope):
    args = (torch.empty(4, 2, device="cpu"), torch.empty(5, 2, device="cpu"))
    kernel = ValidatingKernel()
    kernel.error = ValueError("full Kernel validation")
    launcher = launcher_scope["_CachedLauncher"](kernel)
    with pytest.raises(ValueError) as error:
        launcher(*args, stream=37)
    assert error.value is kernel.error
    assert not kernel.adapter.lib.calls


def test_noncontiguous_fallback_receives_original_output_storage(launcher_scope):
    args = (torch.empty(4, 2, 3, device="cpu").transpose(1, 2), torch.empty(4, 3, 2, device="cpu"))
    kernel = ValidatingKernel()
    launcher = launcher_scope["_CachedLauncher"](kernel, {0: "flat2d"})
    assert launcher(*args, stream=37) is kernel.result
    assert not kernel.adapter.lib.calls
    assert all(a is b for a, b in zip(kernel.fallback_calls[0][0], args))


@pytest.mark.parametrize("mutation", ["shape", "stride", "storage_offset", "data_ptr"])
def test_same_id_storage_or_metadata_mutation_invalidates_raw_cache(launcher_scope, mutation):
    args = (torch.empty(4, 2, device="cpu"), torch.empty(4, 2, device="cpu"))
    kernel = ValidatingKernel()
    launcher = launcher_scope["_CachedLauncher"](kernel)
    launcher(*args, stream=37)
    if mutation == "shape":
        args[0].resize_(5, 2)
    elif mutation == "stride":
        args[0].as_strided_((4, 2), (2, 0))
    elif mutation == "storage_offset":
        backing = torch.empty(5, 2, device="cpu")
        args[0].set_(backing.untyped_storage(), 2, (4, 2), (2, 1))
    else:
        args[0].set_(torch.empty_like(args[0]))
    assert launcher(*args, stream=41) is kernel.result
    assert len(kernel.adapter.lib.calls) == 1
    assert len(kernel.fallback_calls) == 1
    assert launcher._key is None
    assert launcher._args_refs == ()


def test_default_stream_resolution_is_lazy_and_repeated_per_call(launcher_scope, monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(_C=SimpleNamespace(_npu_getCurrentRawStream=lambda device: calls.append(device) or 71))
    )
    args = (torch.empty(4, 2, device="cpu"), torch.empty(4, 2, device="cpu"))
    kernel = ValidatingKernel()
    launcher = launcher_scope["_CachedLauncher"](kernel)
    assert calls == []
    launcher(*args)
    launcher(*args)
    assert calls == [-1, -1]
    assert [call[-1] for call in kernel.adapter.lib.calls] == [71, 71]


def test_missing_raw_getter_retains_original_current_stream_fallback(launcher_scope, monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(_C=SimpleNamespace()))
    launcher_scope["torch"] = SimpleNamespace(
        Tensor=torch.Tensor,
        npu=SimpleNamespace(current_stream=lambda: calls.append("current_stream") or SimpleNamespace(npu_stream=73)),
    )
    kernel = ValidatingKernel()
    launcher = launcher_scope["_CachedLauncher"](kernel)
    launcher(torch.empty(4, 2, device="cpu"), torch.empty(4, 2, device="cpu"))
    assert calls == ["current_stream"]
    assert kernel.adapter.lib.calls[-1][-1] == 73
