"""Original experiment examples: import/runtime isolation and full source requests."""

import argparse
import builtins
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_experiment_rest_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
REQUESTS = MANIFEST["factory_requests"]
BY_SOURCE = {entry["source_path"]: entry for entry in ENTRIES}


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_experiment_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
def test_import_does_not_execute_original_runtime(entry, monkeypatch):
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    def forbidden(*args, **kwargs):
        raise AssertionError("import executed original runtime")

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    for obj, names in [
        (tilelang, ["disable_cache", "compile", "lower"]),
        (torch, ["manual_seed", "set_default_device", "randn", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    previous = sys.modules.get("torch_npu")
    module = _load(entry)
    assert callable(module.main)
    assert sys.modules.get("torch_npu") is previous


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["source_runtime"]["clear_cache"]], ids=lambda e: e["source_path"])
def test_clear_requires_explicit_owned_cache(entry, monkeypatch):
    module = _load(entry)
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="dedicated TILELANG_CACHE_DIR"):
        module.main()


@pytest.mark.parametrize("entry", [e for e in ENTRIES if e["source_runtime"]["clear_cache"]], ids=lambda e: e["source_path"])
def test_resolved_cache_override_conflict_cannot_clear_other_cache(entry, tmp_path, monkeypatch):
    from tilelang import env
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setattr(vars(type(env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "other"))

    def forbidden(*args):
        raise AssertionError("other caller cache was cleared")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    with pytest.raises(RuntimeError, match="resolved cache differs"):
        module.main()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
@pytest.mark.parametrize("enabled", [False, True])
def test_original_cache_mode_and_registration_during_runtime_restore_caller_on_failure(entry, enabled, tmp_path, monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    module = _load(entry)
    if entry["source_runtime"].get("cache_disable_env"):
        monkeypatch.delenv(entry["source_runtime"]["cache_disable_env"], raising=False)
    monkeypatch.setattr(sys, "argv", [entry["destination_path"]])
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args: None)
    if entry["source_runtime"].get("host_core_count_probe"):
        monkeypatch.setattr(module, "_get_core_num", lambda: 20)
    if (
        entry["source_runtime"]["set_default_device"]
        or entry["source_runtime"].get("host_availability_gate")
        or entry["source_runtime"].get("constructs_device_object_before_factory")
    ):
        # Exercise the public CPU context stack without selecting an NPU.
        class HostTorch:
            def device(self, name):
                assert name == "npu"
                return torch.device("cpu")

            def __getattr__(self, name):
                if name == "npu" and entry["source_runtime"].get("host_availability_gate"):
                    return SimpleNamespace(is_available=lambda: True)
                return getattr(torch, name)

        monkeypatch.setattr(module, "torch", HostTorch())
    events = []
    original_import = builtins.__import__
    original_disable = tilelang.disable_cache

    def observe_import(name, *args, **kwargs):
        if name == "torch_npu":
            events.append("registration")
        return original_import(name, *args, **kwargs)

    def disable():
        events.append("disable")
        original_disable()

    monkeypatch.setattr(builtins, "__import__", observe_import)
    monkeypatch.setattr(tilelang, "disable_cache", disable)
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda *args: events.append("clear"))

    class StopFactory(BaseException):
        pass

    def stop(*args, **kwargs):
        events.append("inputs" if entry["source_runtime"].get("inputs_before_factory") else "factory")
        assert CacheState.is_enabled() == (False if entry["source_runtime"]["disable_cache"] else enabled)
        raise StopFactory

    request = next(r for r in REQUESTS if r["source_path"] == entry.get("source_request_owner", entry["source_path"]))
    monkeypatch.setattr(module, entry.get("runtime_factory", request["factory"]), stop)
    if entry["source_runtime"].get("inputs_before_factory"):
        # Preserve original input allocation before factory construction; fail
        # at the first input boundary without allocating any device tensor.
        monkeypatch.setattr(torch, entry["source_runtime"].get("input_constructor", "randn"), stop)
    before = CacheState.is_enabled()
    modes_before = torch.overrides._get_current_function_mode_stack()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        with pytest.raises(StopFactory):
            module.main()
        assert CacheState.is_enabled() == enabled
        modes_after = torch.overrides._get_current_function_mode_stack()
        assert len(modes_before) == len(modes_after)
        assert all(a is b for a, b in zip(modes_before, modes_after))
        expected = ["registration"]
        if entry["source_runtime"].get("additional_registration_before_disable"):
            expected.append("registration")
        if entry["source_runtime"]["disable_cache"]:
            expected.append("disable")
        elif entry["source_runtime"]["clear_cache"]:
            expected.extend(["clear"] * entry["source_runtime"]["clear_count"])
        if entry["source_runtime"].get("additional_registration_after_disable"):
            expected.append("registration")
        expected.append("inputs" if entry["source_runtime"].get("inputs_before_factory") else "factory")
        assert events == expected
    finally:
        CacheState.enable() if before else CacheState.disable()


@pytest.mark.parametrize("case", REQUESTS, ids=lambda r: r["id"])
def test_full_original_factory_lowers_without_device(case, monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    for name in ("tl.transform.A2A3HostLegalize", "target.build.tilelang_ascend_a2a3", "target.build.tilelang_pto_a2a3"):
        if tvm.ffi.get_global_func(name, allow_missing=True) is None:
            pytest.skip("requires USE_ASCEND=ON with the complete A2A3 pipeline")
    monkeypatch.setenv("TL_PLATFORM", "A3")
    factory = getattr(_load(BY_SOURCE[case["source_path"]]), case["factory"])
    if case.get("jit_target_override"):
        factory.target = case["jit_target_override"]
    dialect = "pto" if factory.target == "pto" else "ascendc"
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
    with target:
        args = [tuple(value) if i in case.get("tuple_arg_indices", []) else value for i, value in enumerate(case["args"])]
        if case.get("expected_frontend_error"):
            assert case["expected_frontend_error"] == "AssertionError"
            with pytest.raises(AssertionError):
                factory.get_tir(*args, **case["kwargs"])
            return
        prim = factory.get_tir(*args, **case["kwargs"])
    config = normalize_a2a3_pass_configs(target, factory.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=config):
        if case.get("expected_codegen_error"):
            with pytest.raises(tvm.error.InternalError, match=case["expected_codegen_error"]):
                tilelang.lower(prim, target=target)
            return
        artifact = tilelang.lower(prim, target=target)
    assert 'extern "C" void call(' in artifact.kernel_source


@pytest.mark.parametrize(
    "argv,expected",
    [
        ([], (1024, 128, 42, 256)),
        (["--m", "129", "--block_size", "64", "--seed", "17"], (129, 64, 17, 128)),
    ],
)
def test_random_original_parser_reconfigures_factory_globals(argv, expected, tmp_path, monkeypatch):
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    entry = next(e for e in ENTRIES if "/random_1d/" in e["source_path"])
    module = _load(entry)
    assert (module.M, module.BLOCK_SIZE, module.SEED, module.TOTAL_BLOCK_SIZE) == (1024, 128, 42, 256)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", lambda *args: None)
    monkeypatch.setattr(sys, "argv", [entry["destination_path"], *argv])

    class StopFactory(BaseException):
        pass

    def original_factory_boundary(M, block_size, seed, a, c):
        assert (M, block_size, seed, module.TOTAL_BLOCK_SIZE) == expected
        assert (a, c) == (1103515245, 12345)
        raise StopFactory

    monkeypatch.setattr(module, "random_1d", original_factory_boundary)
    with pytest.raises(StopFactory):
        module.main()


def test_mla_decode_layered_uses_exact_private_sibling():
    entry = next(e for e in ENTRIES if e["batch"] == "E4a" and "/test_" in e["source_path"])
    module = _load(entry)
    assert Path(module._mla_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    assert module._col_indices_cpu is None


def test_mla_decode_lazy_indices_real_CPU_values(tmp_path):
    import subprocess

    entry = next(e for e in ENTRIES if e["batch"] == "E4a" and "/test_" in e["source_path"])
    script = ROOT / entry["destination_path"]
    # Explicit CPU-only metadata probe in a separate process; parent tests keep
    # all real tensor factories forbidden and do not execute kernel/golden.
    code = (
        "import os,ctypes\nif os.environ.get('SKIP_LOADING_TILELANG_SO')=='1':\n import tvm\n ctypes.CDLL(os.path.join(os.environ['TVM_LIBRARY_PATH'],'libtilelang.so'),mode=ctypes.RTLD_GLOBAL)\nimport importlib.util,torch; s=importlib.util.spec_from_file_location('mla_cpu_metadata',"
        + repr(str(script))
        + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m._col_indices_cpu is None; before=torch.random.get_rng_state().clone(); x=m._get_col_indices_cpu(); assert x.device.type=='cpu' and x.dtype==torch.float32 and x.tolist()==list(range(m.BLOCK_N)); assert m._get_col_indices_cpu() is x and torch.equal(before,torch.random.get_rng_state()); print('CPU indices exact, cached, RNG unchanged')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=dict(__import__("os").environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CPU indices exact, cached, RNG unchanged" in result.stdout


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e["batch"] == "E4b" and "/test_" in e["source_path"]], ids=lambda e: e["source_path"]
)
def test_mla_paged_layered_uses_exact_private_sibling(entry):
    module = _load(entry)
    assert Path(module._mla_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    assert module.mla_decode_tilelang is module._mla_module.mla_decode_tilelang


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e["batch"] == "E4c" and "/test_" in e["source_path"]], ids=lambda e: e["source_path"]
)
def test_mla_persistent_layered_exact_sibling_and_core_query_fallback(entry, monkeypatch):
    module = _load(entry)
    assert Path(module._mla_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()

    class MetadataOnlyTorch:
        npu = SimpleNamespace(get_device_properties=lambda name: SimpleNamespace(cube_core_num=37))

    monkeypatch.setattr(module, "torch", MetadataOnlyTorch())
    assert module._get_core_num() == 37
    MetadataOnlyTorch.npu = SimpleNamespace()
    assert module._get_core_num() == 20


@pytest.mark.parametrize("wrong_dtype,seqlen,message", [(True, 128, "q.dtype must be fp16"), (False, 65, "must be divisible by block_N")])
def test_mla_persistent_original_host_metadata_negatives_without_tensor(wrong_dtype, seqlen, message, monkeypatch):
    import torch

    entry = next(e for e in ENTRIES if e["batch"] == "E4c" and "/test_" not in e["source_path"])
    module = _load(entry)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid original metadata reached constructor")

    monkeypatch.setattr(module, "flashattn_mla_decode", forbidden)
    dtype = torch.float32 if wrong_dtype else torch.float16
    shapes = [(1, 64, 512), (1, 64, 64), (1, seqlen, 1, 512), (1, seqlen, 1, 64)]
    metadata = [SimpleNamespace(shape=shape, ndim=len(shape), dtype=dtype, device=SimpleNamespace(type="npu")) for shape in shapes]
    with pytest.raises(AssertionError, match=message):
        module.run_mla_decode(*metadata, 64, 64, 20)


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e["batch"] == "E5a" and "/test_" in e["source_path"]], ids=lambda e: e["source_path"]
)
def test_nsa_decode_layered_uses_exact_private_sibling(entry):
    module = _load(entry)
    assert Path(module._nsa_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    assert module.nsa_decode is module._nsa_module.nsa_decode
    assert module.run_kernel is module._nsa_module.run_kernel


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e["batch"] == "E5b" and "/test_" in e["source_path"]], ids=lambda e: e["source_path"]
)
def test_nsa_fwd_layered_uses_exact_private_sibling(entry):
    module = _load(entry)
    assert Path(module._nsa_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    assert module.native_sparse_attention is module._nsa_module.native_sparse_attention


def test_nsa_fwd_original_short_sequence_host_assert_before_allocation(monkeypatch):
    import torch

    entry = next(e for e in ENTRIES if e["batch"] == "E5b" and "/test_" in e["source_path"])
    module = _load(entry)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid short sequence reached allocation")

    monkeypatch.setattr(torch, "zeros", forbidden)
    q = SimpleNamespace(shape=(2, 16, 16, 32))
    kv = SimpleNamespace(shape=(2, 16, 1, 32))
    with pytest.raises(AssertionError, match=r"seq_len \(16\) must be >= block_size \(32\)"):
        module.prepare_inputs(q, kv, kv, None, None, 32, 1, bs_pad=64)


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e["batch"] == "E5c" and "/test_" in e["source_path"]], ids=lambda e: e["source_path"]
)
def test_nsa_varlen_layered_uses_exact_private_sibling(entry):
    module = _load(entry)
    assert Path(module._nsa_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    assert module.native_sparse_attention_varlen is module._nsa_module.native_sparse_attention_varlen


def test_nsa_varlen_original_odd_group_assert_before_seed_or_allocation(monkeypatch):
    import torch

    entry = next(e for e in ENTRIES if e["batch"] == "E5c" and "/test_" in e["source_path"])
    module = _load(entry)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid original group reached seed or allocation")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch, "zeros", forbidden)
    with pytest.raises(AssertionError, match=r"groups G=HQ/H \(15\) must be even"):
        module.make_test_data(2, 64, 1, 15, 64, 1, 32, torch.float16)


@pytest.mark.parametrize(
    "entry", [e for e in ENTRIES if e["batch"] == "E5d" and "/test_" in e["source_path"]], ids=lambda e: e["source_path"]
)
def test_nsa_bwd_layered_uses_exact_private_sibling(entry):
    module = _load(entry)
    assert Path(module._nsa_module.__file__).resolve() == (ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]).resolve()
    for name in ["_compute_block_mask_cpu", "_generate_test_data", "_run_bwd_pipeline", "_run_nsa_pipeline"]:
        assert getattr(module, name) is getattr(module._nsa_module, name)


@pytest.mark.parametrize("zero_sequence", [False, True])
def test_nsa_bwd_original_host_checks_precede_mask_allocation(zero_sequence, monkeypatch):
    entry = next(e for e in ENTRIES if e["batch"] == "E5d" and "/test_" not in e["source_path"])
    module = _load(entry)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid original shape reached mask allocation or factory")

    monkeypatch.setattr(module, "_compute_block_mask_cpu", forbidden)
    monkeypatch.setattr(module, "nsa_fwd", forbidden)
    length = 0 if zero_sequence else 32
    q = SimpleNamespace(shape=(1, length, 16, 32 if zero_sequence else 48))
    kv = SimpleNamespace(shape=(1, length, 1, 32))
    message = "seq_len \\(T\\) must be >= 1" if zero_sequence else "Q shape"
    with pytest.raises(AssertionError, match=message):
        module._run_nsa_pipeline(q, kv, kv, q, None, None, 1, length, 1, 16, 32, 1, 32)


def test_dsa_layered_uses_exact_private_sibling():
    entry = next(e for e in ENTRIES if e["batch"] == "E5e" and "/test_" in e["source_path"])
    module = _load(entry)
    expected = ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]
    assert Path(module._dsa_module.__file__).resolve() == expected.resolve()
    assert module.indexer_bwd_interface is module._dsa_module.indexer_bwd_interface


def test_dsa_core_query_and_original_exception_fallback_are_host_contracts(monkeypatch):
    entry = next(e for e in ENTRIES if e["batch"] == "E5e" and "/test_" not in e["source_path"])
    module = _load(entry)
    # Controlled Python metadata objects only. No real device query or context.
    fake = SimpleNamespace(
        npu=SimpleNamespace(is_available=lambda: True, get_device_properties=lambda i: SimpleNamespace(cube_core_num=37))
    )
    monkeypatch.setattr(module, "torch", fake)
    assert module._get_cube_core_num() == 37
    fake.npu = SimpleNamespace(is_available=lambda: False)
    assert module._get_cube_core_num() == 20
    fake.npu = SimpleNamespace(is_available=lambda: True)
    assert module._get_cube_core_num() == 20


def test_gqa_layered_uses_exact_private_kernel_and_autograd_alias():
    entry = next(e for e in ENTRIES if e["batch"] == "E6a" and "/test_" in e["source_path"])
    module = _load(entry)
    expected = ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]
    assert Path(module._gqa_module.__file__).resolve() == expected.resolve()
    assert module.kernel is module._gqa_module.attention
    assert module.flashattn_fwd is module._gqa_module.flashattn_fwd


@pytest.mark.parametrize("disable_env", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_gqa_original_cache_env_selects_warmup_or_precision_without_running_either(disable_env, enabled, monkeypatch):
    import torch
    from tilelang.env import CacheState

    entry = next(e for e in ENTRIES if e["batch"] == "E6a" and "/test_" in e["source_path"])
    module = _load(entry)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(sys, "argv", [entry["destination_path"]])
    monkeypatch.setattr(torch, "manual_seed", lambda *args: None)
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1" if disable_env else "0")
    events = []

    class StopBeforeOriginalWorkload(BaseException):
        pass

    def stop_at(boundary):
        def stop():
            events.append(boundary)
            assert CacheState.is_enabled() == (False if disable_env else enabled)
            raise StopBeforeOriginalWorkload

        return stop

    monkeypatch.setattr(module, "_warmup_compilation", stop_at("warmup"))
    monkeypatch.setattr(module, "test_gqa_bwd_l0", stop_at("precision"))
    before = CacheState.is_enabled()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        with pytest.raises(StopBeforeOriginalWorkload):
            module.main()
        assert events == ["precision" if disable_env else "warmup"]
        assert CacheState.is_enabled() == enabled
    finally:
        CacheState.enable() if before else CacheState.disable()


def test_gqa_rendered_msprof_loader_uses_only_real_sibling_before_any_workload(tmp_path, monkeypatch):
    import ast
    import os

    entry = next(e for e in ENTRIES if e["batch"] == "E6a" and "/test_" in e["source_path"])
    module = _load(entry)
    script = module._MSPROF_TARGET_SCRIPT.format(B=1, H=4, N=128, D_qk=128, D_v=128, groups=2, is_causal=False)
    tree = ast.parse(script)
    start = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Import) and any(a.name == "importlib.util" for a in n.names))
    # Execute only the exact sibling loader, not the generated script imports,
    # tensor generation, compile, benchmark or kernel calls.
    prefix = tree.body[start : start + 4]
    code = compile(ast.Module(body=prefix, type_ignores=[]), "<private msprof loader only>", "exec")
    expected = ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]
    previous = sys.modules.get("torch_npu")
    namespace = {"os": os, "__file__": str(expected.with_name("_msprof_target_auto.py"))}
    exec(code, namespace)
    assert Path(namespace["_gqa_module"].__file__).resolve() == expected.resolve()
    assert sys.modules.get("torch_npu") is previous
    # A global old-name decoy cannot replace a missing private neighbor.
    monkeypatch.setitem(sys.modules, "example_gqa_bwd", ModuleType("example_gqa_bwd"))
    with pytest.raises(FileNotFoundError):
        exec(code, {"os": os, "__file__": str(tmp_path / "_msprof_target_auto.py")})


def test_gqa_sink_layered_exact_private_sibling():
    entry = next(e for e in ENTRIES if e["batch"] == "E6b" and "/test_" in e["source_path"])
    module = _load(entry)
    expected = ROOT / BY_SOURCE[entry["source_request_owner"]]["destination_path"]
    assert Path(module._gqa_sink_module.__file__).resolve() == expected.resolve()
    assert module.flashattn_fwd is module._gqa_sink_module.flashattn_fwd
    assert module.run_bwd_pipeline is module._gqa_sink_module.run_bwd_pipeline


@pytest.mark.parametrize("length,message", [(129, "divisible"), (131, "divisible"), (0, "must be > 0")])
def test_gqa_sink_original_host_guards_precede_tensor_or_factory(length, message, monkeypatch):
    import torch

    entry = next(e for e in ENTRIES if e["batch"] == "E6b" and "/test_" in e["source_path"])
    module = _load(entry)

    def forbidden(*args, **kwargs):
        raise AssertionError("host-invalid input reached a tensor or factory")

    monkeypatch.setattr(torch, "tensor", forbidden)
    monkeypatch.setattr(module, "flashattn_fwd", forbidden)
    with pytest.raises(AssertionError, match=message):
        module._run_case("original_host_guard", 1, 4, 2, [length], [128], 128, None, "l2")


def test_gqa_sink_original_l2_192_is_host_aligned_before_first_tensor(monkeypatch):
    import torch

    entry = next(e for e in ENTRIES if e["batch"] == "E6b" and "/test_" in e["source_path"])
    module = _load(entry)
    seen = []

    class StopBeforeDeviceAllocation(BaseException):
        pass

    def stop(values, **kwargs):
        seen.append((values, kwargs))
        raise StopBeforeDeviceAllocation

    monkeypatch.setattr(torch, "tensor", stop)
    with pytest.raises(StopBeforeDeviceAllocation):
        module._run_case("l2_tailmid_seq192", 1, 4, 2, [192], [192], 128, None, "l2")
    assert seen == [([0, 192], {"dtype": torch.int32, "device": "npu"})]


@pytest.mark.parametrize("enabled", [False, True])
def test_gqa_sink_direct_layered_helper_restores_cache_and_device_on_failure(enabled, monkeypatch):
    import torch
    from tilelang.env import CacheState

    entry = next(e for e in ENTRIES if e["batch"] == "E6b" and "/test_" in e["source_path"])
    module = _load(entry)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))

    class HostTorch:
        def device(self, name):
            assert name == "npu"
            return torch.device("cpu")

        def __getattr__(self, name):
            return getattr(torch, name)

    monkeypatch.setattr(module, "torch", HostTorch())

    class StopBeforeOriginalWorkload(BaseException):
        pass

    def stop():
        assert not CacheState.is_enabled()
        raise StopBeforeOriginalWorkload

    monkeypatch.setattr(module, "test_gqa_sink_bwd_l0", stop)
    before = CacheState.is_enabled()
    modes_before = torch.overrides._get_current_function_mode_stack()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        with pytest.raises(StopBeforeOriginalWorkload):
            module.run_layered_tests("l0")
        assert CacheState.is_enabled() == enabled
        modes_after = torch.overrides._get_current_function_mode_stack()
        assert len(modes_after) == len(modes_before)
        assert all(a is b for a, b in zip(modes_before, modes_after))
    finally:
        CacheState.enable() if before else CacheState.disable()


def test_mha_backward_private_forward_dependency_and_no_import_allocator_mutation():
    import torch

    entry = next(e for e in ENTRIES if e["batch"] == "E6c" and "/test_" not in e["source_path"])
    original = torch.empty
    module = _load(entry)
    assert torch.empty is original
    assert module._FWD_DIR.resolve() == ROOT / "examples/ascend_a2a3/mha_sink_fwd_bhsd"
    assert module.flashattn_fwd_kernel is module._fwd_kernel.flashattn
    assert module.ref_program is module._fwd_test.ref_program
    test = _load(next(e for e in ENTRIES if e["batch"] == "E6c" and "/test_" in e["source_path"]))
    assert test.attention == test._mha_bwd_module.attention
    assert test.ref_program is test._mha_bwd_module.ref_program
    assert torch.empty is original


def test_mha_backward_missing_original_optional_perf_is_warning_return(monkeypatch, capsys):
    module = _load(next(e for e in ENTRIES if e["batch"] == "E6c" and "/test_" in e["source_path"]))
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "perf_mha_sink_bwd_bhsd":
            raise ImportError("fixed TA has no optional perf module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    assert module.run_perf_benchmark() is None
    assert "[WARN] perf_mha_sink_bwd_bhsd not found, skipping performance benchmark" in capsys.readouterr().out
