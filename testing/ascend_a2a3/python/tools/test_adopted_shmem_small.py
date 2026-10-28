"""Original shmem worker assets: safe host entry, never distributed execution."""

import builtins
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import tilelang
from tilelang.env import CacheState
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.cache.kernel_cache import KernelCache

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/shmem"
FILES = ["shmem_mm_test_" + name + ".py" for name in ["get_nbi", "put_nbi", "ub_get_nbi", "ub_put_nbi"]]


def _module(filename):
    spec = importlib.util.spec_from_file_location("adopted_" + filename.removesuffix(".py"), ASSET / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("filename", FILES)
def test_import_help_parser_never_load_shmem_register_NPU_clear_cache_or_spawn(filename, monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch_npu", "shmem"}:
            pytest.fail("import/help must not load the external shmem/NPU runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: pytest.fail("import/help cleared cache"))
    module = _module(filename)
    monkeypatch.setattr(module.mp, "Process", lambda *args, **kwargs: pytest.fail("import/help created a worker"))
    args = module.parse_args([])
    assert (args.m, args.n, args.num_processes) == (1, 352, 8)
    changed = module.parse_args(["--m", "2", "--n", "64", "--num_processes", "3"])
    assert (changed.m, changed.n, changed.num_processes) == (2, 64, 3)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--num_processes" in capsys.readouterr().out


@pytest.mark.parametrize("filename", FILES)
@pytest.mark.parametrize("conflict", [False, True])
def test_cache_refusal_precedes_runtime_import_clear_and_process_creation(filename, conflict, tmp_path, monkeypatch):
    module = _module(filename)
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda *args: pytest.fail("refused request queried runtime dependency"))
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: pytest.fail("refused request cleared cache"))
    with pytest.raises(RuntimeError, match=message):
        module.main([])


@pytest.mark.parametrize("filename", FILES)
def test_actual_missing_runtime_extension_stops_before_clear_or_backend_registration(filename, tmp_path, monkeypatch):
    module = _module(filename)
    if importlib.util.find_spec("shmem") is not None:
        pytest.skip("this negative checks the actual missing Python shmem installation")
    path = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", path)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", path)
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch_npu":
            pytest.fail("missing shmem must be diagnosed before registering the NPU backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda self: pytest.fail("missing runtime cleared cache"))
    with pytest.raises(ModuleNotFoundError, match="pinned shmem Python extension"):
        module.main([])


@pytest.mark.parametrize("filename", FILES)
@pytest.mark.parametrize("enabled", [False, True])
def test_real_owned_clear_and_exception_before_workers_restore_caller_memory_flags_and_globals(filename, enabled, tmp_path, monkeypatch):
    module = _module(filename)
    path = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", path)
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", path)
    cache = A2A3CythonKernelCache()
    caller_memory = {"caller-owned": object()}
    monkeypatch.setattr(cache, "_memory_cache", caller_memory)
    caller_snapshot = caller_memory.copy()
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    staging = KernelCache._last_cleaned_staging_root
    caller_globals = module.M, module.N, module.num_processes
    root = Path(cache._get_cache_root())
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / "owned-clear-sentinel"
    sentinel.write_text("clear only this task namespace")
    outside = tmp_path / "caller-disk-sentinel"
    outside.write_text("must survive")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: object())
    original = builtins.__import__

    def import_boundary_only(name, *args, **kwargs):
        if name in {"torch_npu", "shmem"}:
            return SimpleNamespace()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_boundary_only)

    def before_real_processes():
        assert (module.M, module.N, module.num_processes) == (2, 64, 3)
        assert cache._memory_cache is not caller_memory and cache._memory_cache == {}
        assert not sentinel.exists() and outside.read_text() == "must survive"
        tilelang.disable_cache()
        raise RuntimeError("CPU sentinel before original Barrier or worker creation")

    monkeypatch.setattr(module, "_run_original", before_real_processes)
    with pytest.raises(RuntimeError, match="CPU sentinel before original Barrier"):
        module.main(["--m", "2", "--n", "64", "--num_processes", "3"])
    assert cache._memory_cache is caller_memory and caller_memory == caller_snapshot
    assert CacheState._enabled is enabled and KernelCache._last_cleaned_staging_root == staging
    assert (module.M, module.N, module.num_processes) == caller_globals
    assert outside.read_text() == "must survive"


@pytest.mark.a2a3_native
@pytest.mark.parametrize("filename", FILES)
@pytest.mark.parametrize("rank", range(8))
def test_original_full_default_peer_matrix_source_uses_real_private_pipeline(filename, rank, tmp_path, request):
    from tilelang import tvm

    module = _module(filename)
    name = filename.removeprefix("shmem_mm_test_").removesuffix(".py")
    factory = getattr(module, "shmem_" + name)
    peer = (rank + (-1 if "get" in name else 1)) % 8
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, 1, 352, 352, peer)
    with tvm.transform.PassContext(config=module.pass_configs):
        artifact = tilelang.lower(func, target=target)
    assert len(artifact.params) == 2
    assert all(str(p.dtype) == "int8" and [int(x) for x in p.shape] == [1, 352] for p in artifact.params)
    assert f"tl::ascend::shmem_{name}<int8_t>" in artifact.kernel_source
    assert "352, " + str(peer) in artifact.kernel_source
    prototype = next(line.strip() for line in artifact.kernel_source.splitlines() if "void call(" in line)
    assert prototype.count("uint8_t*") == 2 and "aclrtStream stream" in prototype
    source = tmp_path / (name + "-rank" + str(rank) + ".cc")
    source.write_text(artifact.kernel_source)
    request.node.user_properties.append(("shmem_current_source", str(source)))
