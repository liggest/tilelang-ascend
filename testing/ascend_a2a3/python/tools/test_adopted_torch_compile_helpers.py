"""Host contracts for the original Torch interop compilation helpers."""

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import tilelang
from tilelang.env import CacheState

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/torch_tl_ascend/compile_tl_op/flash_attention.py"


def _module():
    spec = importlib.util.spec_from_file_location("adopted_torch_compile", ASSET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _owned(module, tmp_path, monkeypatch):
    path = str(tmp_path / "owned-cache")
    monkeypatch.setenv("TILELANG_CACHE_DIR", path)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", path)


def _factory(module, monkeypatch):
    dependency = os.environ.get("A2A3_INTEROP_DEPENDENCY_ROOT")
    if dependency:
        monkeypatch.setattr(module, "EXAMPLE_ROOT", Path(dependency))
    if not (module.EXAMPLE_ROOT / "flash_attention/flash_attn_bhsd.py").is_file():
        pytest.skip("requires the adopted sibling flash_attention asset")
    return module.get_jit_op_func()


def test_import_does_not_patch_library_generator_or_change_path(monkeypatch):
    from tilelang.jit.adapter.libgen import LibraryGenerator

    original = LibraryGenerator.load_lib
    path = sys.path.copy()
    monkeypatch.setattr(tilelang, "disable_cache", lambda: pytest.fail("import altered cache"))
    _module()._ensure_lib_path()
    utility = ASSET.parent / "util/wrap_libgen.py"
    spec = importlib.util.spec_from_file_location("adopted_wrap_libgen", utility)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert LibraryGenerator.load_lib is original and sys.path == path


@pytest.mark.parametrize("conflicts", [False, True])
def test_exports_refuse_missing_or_conflicting_owned_cache_before_copy_or_compile(conflicts, tmp_path, monkeypatch):
    module = _module()
    if conflicts:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    monkeypatch.setattr(module, "get_jit_op_func", lambda: pytest.fail("refused export loaded factory"))
    monkeypatch.setattr(module.shutil, "copy", lambda *args: pytest.fail("refused export copied files"))
    for operation in [module.update_package_files, lambda: module.update_so(tmp_path / "libop.so")]:
        with pytest.raises(RuntimeError, match=message):
            operation()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_get_kernel_preserves_original_delegate_and_restores_raw_cache(enabled, force, fails, tmp_path, monkeypatch):
    module = _module()
    _owned(module, tmp_path, monkeypatch)
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    token = object()
    calls = []

    def original_delegate(*args, **kwargs):
        calls.append((args, kwargs, CacheState._enabled))
        if fails:
            raise ValueError("original factory failure")
        return token

    if fails:
        with pytest.raises(ValueError, match="^original factory failure$"):
            module.get_kernel(original_delegate, 4, 4096, 16, 128, force_compile=force, original=True)
    else:
        assert module.get_kernel(original_delegate, 4, 4096, 16, 128, force_compile=force, original=True) is token
    assert calls == [((4, 4096, 16, 128), {"original": True}, False if force else enabled)]
    assert CacheState._enabled is enabled


def test_library_path_protocol_preserves_original_missing_path_error():
    module = _module()
    with pytest.raises(RuntimeError, match="^Compiled library path not found!$"):
        module.so_path_of(SimpleNamespace(adapter=SimpleNamespace(lib_generator=SimpleNamespace(libpath=None))))
    expected = Path("/owned/cache/kernel_lib.so")
    assert module.so_path_of(SimpleNamespace(adapter=SimpleNamespace(lib_generator=SimpleNamespace(libpath=str(expected))))) == expected


def test_real_adopted_factory_source_path_and_module_memo_do_not_change_sys_path(monkeypatch):
    from tilelang.jit import JITImpl

    module = _module()
    path = sys.path.copy()
    factory = _factory(module, monkeypatch)
    assert isinstance(factory, JITImpl)
    assert module.get_jit_op_func() is factory
    assert module.source_path_of(factory) == (module.EXAMPLE_ROOT / "flash_attention/flash_attn_bhsd.py").resolve()
    assert sys.path == path


@pytest.mark.a2a3_native
def test_actual_public_compile_export_and_warm_force_memo(tmp_path, monkeypatch, request):
    """Real lowering/SDK/CDLL/adapter; unused launch providers are CPU-isolated."""
    import json
    import torch
    from tilelang.jit.kernel import JITKernel
    from tilelang.jit.adapter.cython import adapter as adapter_module
    from tilelang.ascend.a2a3.compiler import resolve_compile_config
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
    from tilelang import tvm

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    try:
        resolve_compile_config(target)
    except (ValueError, FileNotFoundError) as error:
        pytest.skip(f"requires actual AscendC toolchain/resources: {error}")
    module = _module()
    _owned(module, tmp_path, monkeypatch)
    monkeypatch.setattr(A2A3CythonKernelCache(), "_memory_cache", {})
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    factory = _factory(module, monkeypatch)
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda self: {})
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda device: 0))
    caller_raw = CacheState._enabled
    kernel = module.get_kernel(factory, 4, 4096, 16, 128)
    assert isinstance(kernel, JITKernel) and kernel.adapter.lib.call is not None
    assert module.get_kernel(factory, 4, 4096, 16, 128, force_compile=True) is kernel
    assert CacheState._enabled is caller_raw
    params = kernel.adapter.params
    assert len(params) == 7
    assert [int(x) for x in params[0].shape] == [4, 16, 4096, 128]
    assert [int(x) for x in params[4].shape] == [4096, 64, 64]
    assert [int(x) for x in params[6].shape] == [4096, 64, 128]
    monkeypatch.setattr(module, "PACKAGE_ROOT", tmp_path / "package")
    monkeypatch.setattr(module, "PACKAGE_SOURCE_ROOT", tmp_path / "package/op_source")
    module.update_package_files()
    exported = tmp_path / "second.so"
    module.update_so(exported)
    assert exported.read_bytes() == module.so_path_of(kernel).read_bytes() == (module.PACKAGE_ROOT / "libop.so").read_bytes()
    assert (module.PACKAGE_SOURCE_ROOT / "flash_attn_bhsd.py").read_bytes() == module.source_path_of(factory).read_bytes()
    source_path = Path(kernel.adapter.lib_generator.get_source_path())
    source = source_path.read_text()
    prototype = next(line.strip() for line in source.splitlines() if "void call(" in line)
    assert prototype.count("uint8_t*") == 7 and "aclrtStream stream" in prototype
    import shutil

    source_copy = tmp_path / "public_flash.cc"
    shutil.copyfile(source_path, source_copy)
    request.node.user_properties.append(("torch_compile_artifacts", json.dumps([str(source_copy), str(exported)])))
