"""Standalone preparation and dependency diagnostics; no device execution."""

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest
import tilelang

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/torch_tl_ascend/demo_libtorch"


def _module():
    spec = importlib.util.spec_from_file_location("adopted_libtorch_prepare", ASSET / "prepare.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_import_help_and_invalid_arguments_are_free_of_heavy_dependencies_or_mkdir(monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "tilelang", "compile_tl_op"}:
            pytest.fail("prepare import/help must not import compiler or backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: pytest.fail("import/help created a library directory"))
    module = _module()
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "fixed flash-attention" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        module.main(["--shape", "1"])
    assert caught.value.code == 2


@pytest.mark.parametrize("conflict", [False, True])
def test_prepare_cache_refusal_precedes_output_directory_or_kernel_build(conflict, monkeypatch, tmp_path):
    module = _module()
    module.LIB = tmp_path / "must-not-be-created"
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        monkeypatch.setattr(module, "_compiler_helpers", lambda: pytest.fail("refused preparation imported compiler"))
        message = "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=message):
        module.main([])
    assert not module.LIB.exists()


@pytest.mark.parametrize("missing", ["unset", "nonexistent", "cache"])
def test_build_dependency_refusal_precedes_any_Python_SDK_or_CMake_command(missing, monkeypatch, tmp_path):
    script = ASSET / "build.sh"
    environment = os.environ.copy()
    environment.pop("CMAKE_PREFIX_TORCH_NPU", None)
    environment.pop("TORCH_NPU_SOURCE_DIR", None)
    if missing == "nonexistent":
        environment.update(CMAKE_PREFIX_TORCH_NPU=str(tmp_path / "absent"), TORCH_NPU_SOURCE_DIR=str(tmp_path / "absent"))
    elif missing == "cache":
        environment.update(CMAKE_PREFIX_TORCH_NPU=str(tmp_path), TORCH_NPU_SOURCE_DIR=str(tmp_path))
        environment.pop("TILELANG_CACHE_DIR", None)
    result = subprocess.run(["bash", str(script)], env=environment, capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "cmake -S" not in result.stderr and "python" not in result.stderr
    assert "Set CMAKE_PREFIX_TORCH_NPU" in result.stderr or "does not exist" in result.stderr or "Set a dedicated" in result.stderr


def test_build_help_is_dependency_and_cache_free():
    result = subprocess.run(["bash", str(ASSET / "build.sh"), "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0 and "real libtorch_npu" in result.stdout


@pytest.mark.a2a3_native
def test_actual_public_prepare_exports_the_real_fixed_library(tmp_path, monkeypatch, request):
    import json
    import torch
    from tilelang import tvm
    from tilelang.jit.kernel import JITKernel
    from tilelang.jit.adapter.cython import adapter as adapter_module
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang.ascend.a2a3.compiler import resolve_compile_config

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    try:
        resolve_compile_config(target)
    except (ValueError, FileNotFoundError) as error:
        pytest.skip(f"requires actual AscendC toolchain/resources: {error}")
    module = _module()
    helper = module._compiler_helpers()
    dependency = os.environ.get("A2A3_INTEROP_DEPENDENCY_ROOT")
    if dependency:
        monkeypatch.setattr(helper, "EXAMPLE_ROOT", Path(dependency))
    if not (helper.EXAMPLE_ROOT / "flash_attention/flash_attn_bhsd.py").is_file():
        pytest.skip("requires the adopted sibling flash-attention asset")
    owned = str(tmp_path / "owned-cache")
    monkeypatch.setenv("TILELANG_CACHE_DIR", owned)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", owned)
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.setattr(A2A3CythonKernelCache(), "_memory_cache", {})
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda self: {})
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda device: 0))
    module.LIB = tmp_path / "owned-output"
    module.main([])
    kernel = helper.get_kernel(helper.get_jit_op_func(), 4, 4096, 16, 128)
    assert isinstance(kernel, JITKernel) and kernel.adapter.lib.call is not None
    exported = module.LIB / "libop.so"
    assert exported.read_bytes()[:4] == b"\x7fELF" and exported.read_bytes() == helper.so_path_of(kernel).read_bytes()
    source = tmp_path / "actual_prepare_flash.cc"
    source.write_text(kernel.get_kernel_source())
    prototype = next(line.strip() for line in source.read_text().splitlines() if "void call(" in line)
    assert prototype.count("uint8_t*") == 7 and "aclrtStream stream" in prototype
    request.node.user_properties.append(("torch_compile_artifacts", json.dumps([str(source), str(exported)])))
