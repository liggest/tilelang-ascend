"""Metadata and CPU-only contracts for the original native Torch binding."""

import builtins
import importlib.util
import os
from pathlib import Path
import sys

import pytest

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/torch_tl_ascend"


def _setup():
    spec = importlib.util.spec_from_file_location("adopted_torch_setup", ASSET / "setup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "arguments", [["--help"], ["--name"], ["--version"], ["egg_info"], ["dist_info"], ["build_ext", "--help"], ["build", "--dry-run"]]
)
def test_metadata_and_help_never_import_torch_or_run_operator_build(arguments, tmp_path, monkeypatch):
    # setuptools itself may add its vendored dependencies on first import;
    # distinguish that standard dependency bootstrap from the asset's old path
    # mutation, which appended the example directory to caller sys.path.
    import setuptools  # noqa: F401

    original = builtins.__import__

    def no_build_dependencies(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "tilelang", "compile_tl_op"}:
            pytest.fail("metadata/help imported a heavy build or NPU dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_build_dependencies)
    before = sys.path.copy()
    module = _setup()
    assert not module._build_requested(arguments)
    version = tmp_path / "VERSION"
    version.write_text("0.0.1\n")
    monkeypatch.setattr(module, "VERSION_FILE", version)
    captured = []
    monkeypatch.setattr(module, "setup", lambda **kwargs: captured.append(kwargs))
    module.main(arguments)
    assert captured[0]["script_args"] == arguments
    assert captured[0]["name"] == "torch_tl_ascend" and captured[0]["version"] == "0.0.1"
    assert captured[0]["ext_modules"][0].name == "torch_tl_ascend._inner"
    assert sys.path == before


@pytest.mark.parametrize("command", ["build", "build_py", "build_ext", "bdist", "bdist_wheel", "install", "develop", "editable_wheel"])
def test_standard_actual_build_commands_retain_operator_build_boundary(command):
    assert _setup()._build_requested([command])


def test_real_compiled_binding_rejects_Meta_and_invalid_fixed_signature_before_device(request):
    """Requires a real built package, never fake registration or NPU execution."""
    package = os.environ.get("A2A3_TORCH_EXTENSION_PACKAGE")
    if not package:
        pytest.skip("requires the independently built real torch_tl_ascend extension package")
    import torch
    import json

    spec = importlib.util.spec_from_file_location("torch_tl_ascend", Path(package) / "__init__.py", submodule_search_locations=[package])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    operator = torch.ops.tl_ascend.flash_attention.default
    shape = (4, 16, 4096, 128)
    half = torch.empty(shape, dtype=torch.float16, device="meta")
    outcomes = []
    with pytest.raises(RuntimeError, match="Meta/torch.compile is not supported") as caught:
        operator(half, half, half)
    outcomes.append({"dispatch": "Meta", "error": str(caught.value)})
    # Explicit redispatch exercises the genuine C++ PrivateUse1 entry's early
    # metadata diagnostics using storage-free Meta carriers, without NPU APIs.
    dispatch = torch._C.DispatchKeySet(torch._C.DispatchKey.PrivateUse1)
    cases = [
        ((torch.empty(4, device="meta"), half, half), "Q must be a 4-D tensor"),
        ((half, torch.empty(4, device="meta"), half), "K must be a 4-D tensor"),
        ((half, half, torch.empty(4, device="meta")), "V must be a 4-D tensor"),
        ((torch.empty((1, 16, 4096, 128), dtype=torch.float16, device="meta"), half, half), "compiled BHSD shape must be"),
        ((half.float(), half, half), "compiled dtype must be float16"),
        ((torch.empty_strided(shape, (0, 0, 0, 0), dtype=torch.float16, device="meta"), half, half), "must be contiguous"),
        ((half, half, half), "must be NPU tensors"),
    ]
    for arguments, message in cases:
        with pytest.raises(RuntimeError, match=message) as caught:
            operator.redispatch(dispatch, *arguments)
        outcomes.append({"dispatch": "PrivateUse1-with-Meta-carriers", "expected": message, "error": str(caught.value)})
    request.node.user_properties.append(("actual_binding_diagnostics", json.dumps(outcomes)))
