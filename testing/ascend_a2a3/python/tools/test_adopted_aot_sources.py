"""Original AOT source/CLI contracts, without running the numerical reader."""

import builtins
import importlib.util
from pathlib import Path
import runpy
import re
import sys

import pytest

from tilelang import env, tvm
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

ASSETS = Path(__file__).parents[4] / "examples/ascend_a2a3/gemm_aot"


def _module(name):
    path = ASSETS / name
    spec = importlib.util.spec_from_file_location("aot_contract_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["example_gemm.py", "test_example_gemm.py"])
@pytest.mark.parametrize("help_requested", [False, True])
def test_import_and_help_are_independent_of_heavy_runtime(name, help_requested, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded(module, *args, **kwargs):
        if module.split(".")[0] in {"tilelang", "torch", "torch_npu"}:
            pytest.fail("AOT import/help must not initialize a compiler or device runtime")
        return original_import(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(sys, "argv", [name, "--help" if help_requested else "--unrecognized-import-argument"])
    if help_requested:
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(ASSETS / name), run_name="__main__")
        assert result.value.code == 0
        assert "--m" in capsys.readouterr().out
    else:
        runpy.run_path(str(ASSETS / name), run_name="aot_import_only")


@pytest.mark.parametrize("conflict", [False, True])
def test_build_refuses_unsafe_cache_clear(conflict, monkeypatch, tmp_path):
    module = _module("example_gemm.py")
    args = module._argument_parser().parse_args([])
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "forced"))
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid cache request must not clear or elaborate")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    monkeypatch.setattr(module, "matmul", forbidden)
    match = "conflicts with the active cache-directory override" if conflict else "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=match):
        module.build(args)


@pytest.mark.parametrize("platform,match", [("A5", "outside the A2/A3 namespace"), ("unknown", "must be auto, A2, or A3")])
def test_unsupported_platform_is_diagnosed_before_clear(platform, match, monkeypatch):
    module = _module("example_gemm.py")
    args = module._argument_parser().parse_args(["--platform", platform])
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda *args: pytest.fail("unsupported platform must not clear the cache"))
    with pytest.raises(ValueError, match=match):
        module.build(args)


class _SourceCaptured(Exception):
    pass


@pytest.mark.a2a3_native
@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
@pytest.mark.parametrize("platform", ["auto", "A2", "A3"])
def test_original_default_aot_source_uses_one_frozen_raw_compiler_request(dialect, platform, monkeypatch, isolated_a2a3_cache):
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(isolated_a2a3_cache))
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    for key in ("TL_CCE_OPT_LEVEL", "TL_CCE_AUTO_SYNC", "TL_PTO_DEBUG"):
        monkeypatch.delenv(key, raising=False)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    real_resolve = compiler.resolve_compile_config
    try:
        real_resolve(target)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"requires actual A2/A3 compiler resources: {error}")
    module = _module("example_gemm.py")
    args = module._argument_parser().parse_args(["--target", dialect, "--platform", platform])
    snapshots = []
    sources = []

    def resolve(*args, **kwargs):
        config = real_resolve(*args, **kwargs)
        snapshots.append(config)
        monkeypatch.setenv("TL_PLATFORM", "A2")
        monkeypatch.setenv("TL_RUN_MODE", "sim")
        monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-freeze")
        return config

    def before_sdk(generator, *args, **kwargs):
        assert generator.config.to_metadata() == snapshots[0].to_metadata()
        assert generator.config.flags == ("-O2",)
        assert generator.config.run_mode == "npu"
        assert generator.config.platform == ("A3" if platform == "auto" else platform)
        source = generator.lib_code
        prototype = next(line for line in source.splitlines() if "void call(" in line)
        stream = "void*stream" if dialect == "pto" else "aclrtStreamstream"
        assert "uint8_t*A_handle,uint8_t*B_handle,uint8_t*C_handle," + stream in re.sub(r"\s+", "", prototype)
        sources.append(source)
        raise _SourceCaptured

    monkeypatch.setattr(compiler, "resolve_compile_config", resolve)
    monkeypatch.setattr(compiler.LibraryGenerator, "compile_lib", before_sdk)
    with pytest.raises(_SourceCaptured):
        module.build(args)
    assert len(snapshots) == len(sources) == 1
