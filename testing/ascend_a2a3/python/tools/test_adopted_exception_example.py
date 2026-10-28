"""Exception example ordering/source contracts; never inject a hardware fault."""

import builtins
import importlib.util
from pathlib import Path
import runpy
import sys

import pytest

from tilelang import env, tvm
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/exception_dump_test/example_exception_dump.py"


def _module():
    spec = importlib.util.spec_from_file_location("exception_cli_contract", ASSET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("help_requested", [False, True])
def test_import_help_do_not_initialize_runtime_or_dump_environment(help_requested, monkeypatch, capsys, tmp_path):
    original = builtins.__import__
    monkeypatch.setenv("ASCEND_DUMP_PATH", str(tmp_path / "caller"))
    monkeypatch.setenv("ASCEND_DUMP_SCENE", "caller-scene")

    def forbidden(name, *args, **kwargs):
        if name.split(".")[0] in {"tilelang", "torch", "torch_npu"}:
            pytest.fail("import/help must not initialize runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden)
    monkeypatch.setattr(sys, "argv", [str(ASSET), "--help" if help_requested else "--unknown-import-only"])
    if help_requested:
        with pytest.raises(SystemExit) as result:
            runpy.run_path(str(ASSET), run_name="__main__")
        assert result.value.code == 0 and "--dump-dir" in capsys.readouterr().out
    else:
        runpy.run_path(str(ASSET), run_name="exception_import_only")
    import os

    assert os.environ["ASCEND_DUMP_PATH"] == str(tmp_path / "caller")
    assert os.environ["ASCEND_DUMP_SCENE"] == "caller-scene"
    assert not (tmp_path / "caller").exists()


class _BeforeRuntimeImport(Exception):
    pass


def test_cli_sets_original_dump_environment_before_first_runtime_import(monkeypatch, tmp_path):
    import os

    module = _module()
    dump = tmp_path / "dump"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned-cache"))
    monkeypatch.setenv("ASCEND_DUMP_PATH", "caller")
    monkeypatch.setenv("ASCEND_DUMP_SCENE", "caller-scene")
    original = builtins.__import__
    imports = []

    def stop(name, *args, **kwargs):
        if name.split(".")[0] in {"tilelang", "torch", "torch_npu"}:
            imports.append(name)
            assert os.environ["ASCEND_DUMP_PATH"] == str(dump)
            assert os.environ["ASCEND_DUMP_SCENE"] == "aic_err_brief_dump"
            assert dump.is_dir()
            raise _BeforeRuntimeImport
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", stop)
    with pytest.raises(_BeforeRuntimeImport):
        module.main(["--dump-dir", str(dump)])
    assert imports == ["tilelang"]


@pytest.mark.parametrize("conflict", [False, True])
def test_cache_guard_stops_before_compile_or_fault(conflict, monkeypatch, tmp_path):
    module = _module()
    monkeypatch.setenv("ASCEND_DUMP_PATH", "caller")
    monkeypatch.setenv("ASCEND_DUMP_SCENE", "caller-scene")
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "env-cache"))
        monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "forced-cache"))
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid cache request must not clear, compile or inject a fault")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    monkeypatch.setattr(module, "vec_add", forbidden)
    monkeypatch.setattr(module, "launch_kernel_with_hw_exception_triggered", forbidden)
    match = "conflicts with the active cache-directory override" if conflict else "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=match):
        module.main(["--dump-dir", str(tmp_path / "dump")])


@pytest.mark.a2a3_native
def test_complete_original_factory_decorator_source_without_normal_or_fault_launch(monkeypatch):
    module = _module()
    factory = module._make_vec_add()
    assert module._make_vec_add() is factory
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, 128, 128)
    import tilelang

    configs = normalize_a2a3_pass_configs(target, factory.pass_configs)
    assert configs["tl.ascend_exception_dump"] is True and factory.out_idx == [2]
    before = tvm.ir.save_json(func)
    with tvm.transform.PassContext(config=configs):
        artifact = tilelang.lower(func, target=target)
    assert len(artifact.params) == 3 and "void call(" in artifact.kernel_source
    assert "main_kernel" in artifact.kernel_source and "exception" in artifact.kernel_source.lower()
    assert tvm.ir.save_json(func) == before
