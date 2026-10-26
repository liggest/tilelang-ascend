"""A2A3 autotuner candidate: frozen flags and final ABI persistence.

TA source 0deb8b93da11b7f0becbcded732d9633b7ca4e34 tracks flags/autoGM but
not declared workspace in the tuner. Workspace forwarding here is an explicit
migration increment. No tuning search, benchmarking, or candidate selection is
replaced by these contracts.
"""

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import types

import pytest
import tilelang
from tilelang import tvm, env
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache
from tilelang.autotuner import AutoTuner
from tilelang.autotuner.param import CompileArgs
from tilelang.backend import create_backend_context

from test_a2a3_public_execution import TARGET, _small_func


def _factory(size=16):
    return _small_func()


@pytest.fixture
def snapshot_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(compiler, "_get_ascend_home_path", lambda **kwargs: str(tmp_path / "sdk"))
    monkeypatch.setattr(compiler, "_get_tl_root", lambda **kwargs: str(Path(__file__).parents[3]))
    monkeypatch.setattr(compiler.shutil, "which", lambda *args, **kwargs: "/unused/bisheng")
    monkeypatch.setattr(compiler, "_get_simulator_lib_path", lambda *args: str(tmp_path / "sim"))
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    for key in ("TL_CCE_OPT_LEVEL", "TL_CCE_AUTO_SYNC", "TL_PTO_DEBUG", "TILELANG_LAYOUT_COST_MODEL"):
        monkeypatch.delenv(key, raising=False)


def _args(**kwargs):
    return CompileArgs(target=TARGET, execution_backend="cython", **kwargs)


def test_default_creation_and_offline_key_have_no_target_or_compiler_probe(monkeypatch):
    import tilelang.backend.module as backend_module

    def forbidden(*args, **kwargs):
        pytest.fail("offline/default CompileArgs must not resolve a device/compiler")

    monkeypatch.setattr(backend_module, "create_backend_context", forbidden)
    monkeypatch.setattr(compiler, "resolve_compile_config", forbidden)
    args = CompileArgs()
    tuner = AutoTuner(_factory, [{"size": 16}])
    assert args._frozen is None and tuner.compile_args._frozen is None
    assert isinstance(hash(args), int)
    assert tuner.generate_cache_key(inspect.signature(_factory).parameters, {})


def test_non_a2a3_frozen_request_retains_the_exact_original_hash(snapshot_paths):
    args = CompileArgs(target={"kind": "llvm"}, execution_backend="tvm_ffi")
    data = {
        "out_idx": None,
        "execution_backend": "tvm_ffi",
        "target": str(args.target),
        "target_host": None,
        "verbose": False,
        "pass_configs": None,
    }
    expected = int.from_bytes(hashlib.sha256(json.dumps(data, sort_keys=True).encode()).digest(), "big")
    assert args.__hash__() == expected
    frozen = args.freeze()
    assert frozen.__hash__() == expected
    assert args._frozen is None and frozen is args


def test_actual_run_freezes_before_first_lookup_without_mutating_shared_default(monkeypatch, snapshot_paths):
    import tilelang.backend.module as backend_module

    context = create_backend_context(TARGET, execution_backend="cython")
    resolutions = []
    monkeypatch.setattr(backend_module, "create_backend_context", lambda *args, **kwargs: resolutions.append(args) or context)
    tuner = AutoTuner(_factory, [{"size": 16}])
    default = tuner.compile_args
    sentinel = types.SimpleNamespace(func=None)
    monkeypatch.setattr(tuner, "_memory_cache", {"frozen-key": sentinel})
    monkeypatch.setattr(
        tuner,
        "generate_cache_key",
        lambda *args: "frozen-key" if tuner.compile_args._frozen is not None else pytest.fail("lookup before freeze"),
    )
    was_enabled = env.is_cache_enabled()
    env.enable_cache()
    try:
        assert tuner.run() is sentinel
    finally:
        if not was_enabled:
            env.disable_cache()
    assert len(resolutions) == 1
    assert default._frozen is None
    assert tuner.compile_args._frozen.compiler_config.platform == "A3"
    assert tuner.compile_args.target.same_as(context.target)
    assert tuner.compile_args.execution_backend == "cython"


def test_candidate_flags_derive_from_the_frozen_environment(snapshot_paths, monkeypatch):
    base = _args(compile_flags=["-DEXPLICIT=1"]).freeze()
    before = hash(base)
    monkeypatch.setenv("TL_PLATFORM", "A2")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    monkeypatch.setenv("TL_CCE_AUTO_SYNC", "on")
    candidate = replace(base, pass_configs={"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": ["-DCANDIDATE=1"]})
    _, _, _, config = candidate._prepare_a2a3_program(_small_func())
    assert config.flags == ("-O3", "--cce-auto-sync=off", "-DCANDIDATE=1", "-DEXPLICIT=1")
    assert config.platform == "A3" and config.run_mode == "npu"
    assert hash(base) == before
    assert config.working_directory == base._frozen.compiler_config.working_directory
    # A later tuning request captures the new environment, not an old memo.
    fresh = _args(compile_flags=["-DEXPLICIT=1"]).freeze()
    assert fresh._frozen.compiler_config.flags == ("-O0", "-DEXPLICIT=1")
    assert hash(fresh) != hash(base)


def test_explicit_o2_environment_is_distinct_from_absent_default_o2(snapshot_paths, monkeypatch):
    absent = _args().freeze()
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "2")
    explicit = _args().freeze()
    assert absent._frozen.compiler_config.cache_identity == explicit._frozen.compiler_config.cache_identity
    assert hash(absent) != hash(explicit)
    a = replace(absent, pass_configs={"tl.ascend_auto_sync_vs": True})._prepare_a2a3_program(_small_func())[3]
    b = replace(explicit, pass_configs={"tl.ascend_auto_sync_vs": True})._prepare_a2a3_program(_small_func())[3]
    assert a.flags[0] == "-O3" and b.flags[0] == "-O2"


def test_frontend_attrs_and_candidate_overrides_are_preserved(snapshot_paths, monkeypatch):
    program = _small_func().with_attr("tilelang_out_idx", [2]).with_attr("tilelang_compile_flags", ["-DFUNCTION=1"])
    program = program.with_attr("tilelang_pass_configs", {"tl.ascend_auto_sync_vs": True})
    before = tvm.ir.save_json(program)
    args = _args(workspace_idx=[-2], compile_flags=["-DEXPLICIT=1"], pass_configs={"tl.ascend_auto_sync_vs": False}).freeze()
    capture = {}

    def cached(self, func, out_idx, **kwargs):
        capture.update(func=func, out_idx=out_idx, **kwargs)
        return "captured"

    monkeypatch.setattr(A2A3CythonKernelCache, "cached", cached)
    assert args.compile_program(program) == "captured"
    assert capture["out_idx"] == [2] and capture["workspace_idx"] == [-2]
    assert capture["compile_config"].flags == ("-O2", "-DFUNCTION=1", "-DEXPLICIT=1")
    assert capture["pass_configs"]["tl.ascend_auto_sync_vs"] is False
    assert tvm.ir.save_json(program) == before
    tuner = AutoTuner(_factory, [])
    tuner.compile_args = args
    merged = tuner._merge_pass_configs_into_compile_args({"tl.ascend_auto_sync_vs": True})
    assert merged._frozen is args._frozen
    assert merged.workspace_idx == [-2] and merged.compile_flags == ["-DEXPLICIT=1"]
    assert merged._prepare_a2a3_program(_small_func())[3].flags[:2] == ("-O3", "--cce-auto-sync=off")


def test_original_decorator_flags_and_workspace_reach_tuner_compile_args(snapshot_paths):
    from tilelang.autotuner import autotune

    @autotune(configs=[{"size": 16}])
    @tilelang.jit(target=TARGET, execution_backend="cython", out_idx=[2], workspace_idx=[-2], compile_flags=["-DORIGINAL=1"])
    def factory(size: int):
        return _small_func()

    tuner = factory.get_tunner()
    assert tuner.compile_args.workspace_idx == [-2]
    assert tuner.compile_args.compile_flags == ["-DORIGINAL=1"]
    assert tuner.compile_args._frozen is None
    args = tuner.compile_args.freeze()
    merged = replace(args, pass_configs={"tl.ascend_auto_sync_vs": True})
    assert merged._prepare_a2a3_program(_small_func())[3].flags == ("-O3", "--cce-auto-sync=off", "-DORIGINAL=1")


@pytest.mark.parametrize("dialect", ["ascendc", "pto"])
@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_frozen_candidates_keep_dialect_platform_and_source_overrides(snapshot_paths, monkeypatch, dialect, platform):
    target = {"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": [dialect]}
    args = CompileArgs(target=target, execution_backend="cython").freeze()
    program = _small_func().with_attr("tilelang_pass_configs", {"tl.ascend_auto_sync_vs": False, "tl.layout_cost_model": "io-aware"})
    monkeypatch.setenv("TL_PLATFORM", "A2" if platform == "A3" else "A3")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "invalid-after-freeze")
    monkeypatch.setenv("TILELANG_LAYOUT_COST_MODEL", "invalid-after-freeze")
    _, pass_configs, _, config = args._prepare_a2a3_program(program)
    assert config.platform == platform and config.dialect == dialect and config.run_mode == "npu"
    assert config.flags[0] == "-O2"
    assert pass_configs["tl.ascend_auto_sync_vs"] is False
    assert pass_configs["tl.layout_cost_model"] == "io-aware"
    assert args.freeze() is args
    candidate = replace(args, pass_configs={"tl.ascend_auto_sync_vs": True, "tl.device_compile_flags": ["-mllvm", "candidate"]})
    _, _, _, config = candidate._prepare_a2a3_program(program)
    assert config.flags[0] == "-O3"
    assert config.flags[config.flags.index("-mllvm") + 1] == "candidate"


def test_unset_layout_environment_does_not_enter_later_candidate_normalization(snapshot_paths, monkeypatch):
    args = _args().freeze()
    original_key = hash(args)
    monkeypatch.setenv("TILELANG_LAYOUT_COST_MODEL", "io-aware")
    _, pass_configs, _, _ = args._prepare_a2a3_program(_small_func())
    assert pass_configs["tl.layout_cost_model"] == "register-count"
    assert hash(args) == original_key
    fresh = _args().freeze()
    assert hash(fresh) != original_key
    assert fresh._prepare_a2a3_program(_small_func())[1]["tl.layout_cost_model"] == "io-aware"


def test_internal_cache_rejects_frozen_dialect_or_platform_conflict(snapshot_paths):
    state = _args().freeze()._frozen
    for target in [
        {"kind": "ascend", "arch": "dav-2201", "model": "A2", "keys": ["ascendc"]},
        {"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["pto"]},
    ]:
        context = create_backend_context(target, execution_backend="cython")
        with pytest.raises(ValueError, match="same.*(platform|dialect)"):
            A2A3CythonKernelCache().cached(_small_func(), backend_context=context, compile_config=state.compiler_config)


def test_shared_compile_forwarding_keeps_existing_defaults_and_new_requested_flags(monkeypatch):
    captured = []
    monkeypatch.setattr(tilelang, "compile", lambda program, **kwargs: captured.append(kwargs))
    program = _small_func()
    args = CompileArgs(target="llvm", execution_backend="tvm_ffi")
    args.compile_program(program)
    assert "workspace_idx" not in captured[-1] and "compile_flags" not in captured[-1]
    replace(args, compile_flags=["-DREQUESTED=1"]).compile_program(program)
    assert captured[-1]["compile_flags"] == ["-DREQUESTED=1"]
    assert captured[-1]["target"] == "llvm" and captured[-1]["execution_backend"] == "tvm_ffi"
