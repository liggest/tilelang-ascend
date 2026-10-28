"""Pure host/AST contracts; these checks never execute a device reference or compiler."""

import ast
from concurrent.futures import ThreadPoolExecutor
import copy
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import threading

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_linear_rnn_ports.json").read_text())
ENTRIES = MANIFEST["entries"]
CAUSAL, NORMALIZE = ENTRIES


def _tree(entry):
    return ast.parse((ROOT / entry["destination_path"]).read_bytes())


def _functions(entry, names, namespace):
    selected = [node for node in _tree(entry).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / entry["destination_path"]), "exec"), namespace)
    return namespace


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
def test_original_factory_and_helper_AST(entry):
    nodes = {node.name: node for node in _tree(entry).body if isinstance(node, ast.FunctionDef)}
    for name, expected in entry["original_definition_AST_sha256"].items():
        if name == "current_jit_target":
            continue  # Explicit modern-target behavior is exercised below.
        node = copy.deepcopy(nodes[name])
        if name == "linear_attention_ker":
            assert ast.unparse(node.body.pop(0)) == "core_num = _resolve_core_num()"
        assert _digest(node) == expected, name
    assert _digest(ast.Module(body=nodes["_run_original_cli"].body, type_ignores=[])) == entry["original_cli_operation_AST_sha256"]
    if entry["original_core_try_AST_sha256"]:
        tries = [n for n in ast.walk(nodes["_resolve_core_num"]) if isinstance(n, ast.Try)]
        assert len(tries) == 1 and _digest(tries[0]) == entry["original_core_try_AST_sha256"]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
def test_no_module_level_device_cache_seed_or_CLI_execution(entry):
    for node in _tree(entry).body:
        if isinstance(node, ast.Expr):
            assert isinstance(node.value, ast.Constant)
        if isinstance(node, ast.Assign):
            assert not any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and ast.unparse(n.func).startswith(("torch.", "tilelang."))
                for n in ast.walk(node)
            )
        assert not isinstance(node, (ast.For, ast.Try))
    imports = [node for node in _tree(entry).body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("torch_npu" in ast.unparse(node) for node in imports)


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
def test_complete_original_CLI_parameters(entry):
    node = next(
        node
        for node in _tree(entry).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "test_configs" for target in node.targets)
    )
    assert ast.literal_eval(node.value) == [tuple(row) for row in entry["original_CLI_source_config_list"]]
    assert hashlib.sha256((ROOT / entry["destination_path"]).read_bytes()).hexdigest() == entry["destination_sha256"]


@pytest.mark.parametrize("target", [None, "auto", "ascendc", "pto", object()])
def test_public_target_adapter_preserves_explicit_identity(target):
    namespace = _functions(NORMALIZE, ["current_jit_target"], {})
    result = namespace["current_jit_target"](SimpleNamespace(target=target))
    assert result == "auto" if target is None else result is target


def _core_namespace(query):
    return _functions(
        CAUSAL,
        ["_resolve_core_num"],
        {"core_num": None, "_core_num_lock": threading.Lock(), "torch": SimpleNamespace(npu=SimpleNamespace(get_device_properties=query))},
    )


@pytest.mark.parametrize("value", [20, 24, "32"])
def test_core_query_is_deferred_and_single_cached_original_int_conversion(value):
    calls = []

    def query(device):
        calls.append(device)
        return SimpleNamespace(cube_core_num=value)

    ns = _core_namespace(query)
    assert not calls and ns["core_num"] is None
    assert ns["_resolve_core_num"]() == int(value)
    assert ns["_resolve_core_num"]() == int(value) and calls == ["npu"]


def test_original_core_exception_fallback_is24_and_is_not_a_platform_guess():
    def failed(device):
        raise AttributeError("synthetic absent backend")

    ns = _core_namespace(failed)
    assert ns["_resolve_core_num"]() == 24
    ns["core_num"] = 28
    assert ns["_resolve_core_num"]() == 28  # An explicit caller override is not overwritten.


def test_core_interrupt_is_not_swallowed_and_lock_releases():
    def interrupted(device):
        raise KeyboardInterrupt

    ns = _core_namespace(interrupted)
    with pytest.raises(KeyboardInterrupt):
        ns["_resolve_core_num"]()
    assert ns["core_num"] is None
    ns["torch"].npu.get_device_properties = lambda device: SimpleNamespace(cube_core_num=24)
    assert ns["_resolve_core_num"]() == 24


def test_concurrent_imported_factories_share_one_environment_resolution():
    entered, release = threading.Event(), threading.Event()
    calls = []

    def query(device):
        calls.append(device)
        entered.set()
        assert release.wait(2)
        return SimpleNamespace(cube_core_num=24)

    ns = _core_namespace(query)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(ns["_resolve_core_num"])
        assert entered.wait(2)
        second = pool.submit(ns["_resolve_core_num"])
        release.set()
        assert first.result(2) == second.result(2) == 24
    assert calls == ["npu"]


class _ShapeOnlyTensor:
    """No data or device: only exercise original host call routing and kwargs."""

    def __init__(self, shape):
        self.shape = tuple(shape)

    def npu(self):
        return self

    def to(self, dtype):
        return self

    def cpu(self):
        return self

    def abs(self):
        return self

    def pow(self, value):
        return self

    def sum(self, **kwargs):
        return self

    def sqrt(self):
        return self

    def __add__(self, other):
        return self

    def __truediv__(self, other):
        return self


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
def test_CPU_only_original_CLI_dispatch_and_thresholds(entry, capsys):
    requests, runtime, assertions, seed, allocations = [], [], [], [], []
    torch = SimpleNamespace(float16="float16", manual_seed=lambda value: seed.append(value), set_printoptions=lambda **kwargs: None)

    def randn(shape):
        allocations.append(tuple(shape))
        return _ShapeOnlyTensor(shape)

    def assert_close(actual, expected, **kwargs):
        assert actual.shape == expected.shape
        assertions.append((actual.shape, kwargs))

    torch.randn = randn
    torch.testing = SimpleNamespace(assert_close=assert_close)

    def factory(name):
        def build(*args):
            requests.append((name, args))

            def kernel(*tensors):
                runtime.append((name, tensors))
                shape = tensors[0].shape
                return _ShapeOnlyTensor((*shape[:2], shape[-1] + 1, shape[-1]) if name == "ker1" else shape)

            return kernel

        return build

    ns = {
        "torch": torch,
        "test_configs": [tuple(row) for row in entry["original_CLI_source_config_list"]],
        "ref_linear_attention": lambda q, k, v: _ShapeOnlyTensor(q.shape),
        "functools": functools,
    }
    causal = entry is CAUSAL
    if causal:
        ns["linear_attention_ker"] = factory("causal")
        names = ["_compiled_ker", "linear_attention", "_run_original_cli"]
    else:
        ns.update(linear_attention_ker1=factory("ker1"), linear_attention_ker2=factory("ker2"))
        names = ["linear_attention", "_run_original_cli"]
    _functions(entry, names, ns)["_run_original_cli"]()
    assert seed == [0]
    assert len(allocations) == 3 * len(entry["original_runtime_requests"])
    assert len(runtime) == (12 if causal else 2)
    assert len(assertions) == len(entry["original_runtime_requests"])
    for (shape, kwargs), row in zip(assertions, entry["original_runtime_requests"]):
        p = row["parameters"]
        assert shape == (p["B"], p["H"], p["L"], p["D"])
        assert kwargs == {"rtol": row["rtol"], "atol": row["atol"]}
    if causal:
        assert requests == [("causal", (2, 128, 64)), ("causal", (20, 128, 64))]
        info = ns["_compiled_ker"].cache_info()
        assert info.hits == 10 and info.misses == 2
    else:
        assert requests == [("ker1", (2, 32, 512, 128, 64, 64)), ("ker2", (2, 32, 512, 128, 64, 64))]
        assert runtime[1][1][1].shape == (2, 32, 129, 128)
    assert "Kernel Output Match!" in capsys.readouterr().out


def _main_namespace(entry, monkeypatch, tmp_path, enabled):
    class CacheState:
        _enabled = enabled

        @classmethod
        def is_enabled(cls):
            return cls._enabled

    calls = []
    tilelang = ModuleType("tilelang")
    tilelang.env = SimpleNamespace(TILELANG_CACHE_DIR=str(tmp_path / "owned"))
    tilelang.disable_cache = lambda: setattr(CacheState, "_enabled", False)
    cache = ModuleType("tilelang.ascend.a2a3.kernel_cache")
    cache.A2A3CythonKernelCache = lambda: SimpleNamespace(clear_cache=lambda: calls.append("clear_owned"))
    env_module = ModuleType("tilelang.env")
    env_module.CacheState = CacheState
    for name, value in {
        "tilelang": tilelang,
        "tilelang.env": env_module,
        "tilelang.ascend.a2a3.kernel_cache": cache,
        "torch_npu": ModuleType("torch_npu"),
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    ns = {"os": os, "Path": Path, "tilelang": tilelang, "_run_original_cli": lambda: calls.append(("host_cli", CacheState._enabled))}
    return _functions(entry, ["main"], ns), CacheState, tilelang.env, calls


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
@pytest.mark.parametrize("enabled", [False, True])
def test_main_scopes_cache_and_dispatches_original_CLI(entry, enabled, monkeypatch, tmp_path):
    ns, state, env, calls = _main_namespace(entry, monkeypatch, tmp_path, enabled)
    ns["main"]()
    assert state._enabled is enabled
    assert calls == ["clear_owned", ("host_cli", False if entry is CAUSAL else enabled)]
    assert str(tmp_path / "owned") == env.TILELANG_CACHE_DIR


@pytest.mark.parametrize("enabled", [False, True])
def test_original_CLI_failure_restores_causal_raw_cache_bit(enabled, monkeypatch, tmp_path):
    ns, state, env, calls = _main_namespace(CAUSAL, monkeypatch, tmp_path, enabled)

    class OriginalFailure(RuntimeError):
        pass

    def fail():
        assert state._enabled is False
        raise OriginalFailure

    ns["_run_original_cli"] = fail
    with pytest.raises(OriginalFailure):
        ns["main"]()
    assert calls == ["clear_owned"] and state._enabled is enabled


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
@pytest.mark.parametrize("conflict", [False, True])
def test_cache_namespace_guard_precedes_clear_and_CLI(entry, conflict, monkeypatch, tmp_path):
    ns, state, env, calls = _main_namespace(entry, monkeypatch, tmp_path, True)
    if conflict:
        env.TILELANG_CACHE_DIR = str(tmp_path / "caller")
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR")
    with pytest.raises(RuntimeError, match="resolved cache differs" if conflict else "dedicated TILELANG_CACHE_DIR"):
        ns["main"]()
    assert calls == [] and state._enabled is True


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda row: Path(row["source_path"]).name)
def test_module_import_has_no_device_cache_or_seed_side_effects(entry, monkeypatch):
    import builtins
    import tilelang
    import torch
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    def forbidden(*args, **kwargs):
        raise AssertionError("import executed runtime/compiler/cache/device state")

    for owner, names in [
        (tilelang, ["compile", "lower", "disable_cache"]),
        (A2A3CythonKernelCache, ["clear_cache"]),
        (torch, ["manual_seed", "set_printoptions", "randn", "randint", "empty", "zeros"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            **{
                name: forbidden
                for name in ["get_device_properties", "is_available", "device_count", "get_device_name", "set_device", "synchronize"]
            }
        ),
        raising=False,
    )
    prior_npu = sys.modules.get("torch_npu")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_npu":
            forbidden()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_linear_import_safety_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
    assert sys.modules.get("torch_npu") is prior_npu
    if entry is CAUSAL:
        assert module.core_num is None


@pytest.mark.a2a3_native
@pytest.mark.usefixtures("isolated_a2a3_cache")
@pytest.mark.parametrize("request_spec", MANIFEST["source_requests"], ids=lambda row: row["id"])
def test_complete_original_factory_pipeline_with_explicit_source_core_sampling(request_spec, monkeypatch):
    from collections import Counter
    import re
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
    import torch

    mode = request_spec["core_sampling"]
    sampled_calls = []

    def sampled_properties(device):
        sampled_calls.append(device)
        if mode == "fallback24":
            raise RuntimeError("explicit source-only fallback sampling, no hardware property query")
        assert mode in ("properties20", "properties24")
        return SimpleNamespace(cube_core_num=20 if mode == "properties20" else 24)

    # Only a CPU namespace stand-in, never registration or a device query.
    monkeypatch.setattr(torch, "npu", SimpleNamespace(get_device_properties=sampled_properties), raising=False)
    entry = next(e for e in ENTRIES if e["source_path"] == request_spec["source_path"])
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("_L1_source_" + request_spec["id"], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = next(r for r in MANIFEST["source_contracts"] if r["id"] == request_spec["id"])
    factory = getattr(module, request_spec["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    with target:
        function = factory.get_tir(*request_spec["args"], **request_spec["kwargs"])
    params = [
        {
            "name": v.name,
            "dtype": str(function.buffer_map[v].dtype) if v in function.buffer_map else str(v.dtype),
            "shape": [str(dim) for dim in function.buffer_map[v].shape] if v in function.buffer_map else [],
        }
        for v in function.params
    ]
    assert params == contract["frontend_params"]
    assert factory.compile_flags == contract["original_compile_flags"]
    if mode == "none":
        assert sampled_calls == []
    else:
        assert sampled_calls == ["npu"] and module.core_num == contract["source_core_value"]
    configs = normalize_a2a3_pass_configs(target, module.pass_configs)
    assert configs == contract["main_effective_configs"]
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    code = artifact.kernel_source
    assert re.search(r'extern "C" void call\(([^\n]+)\)', code)[1] == contract["host_prototype"]
    views = re.findall(r"auto\s+(\w+)\s*=\s*(ascend_\w+)\.GetWithOffset<([^>]+)>\((\d+),\s*(\d+)\);", code)
    assert sorted([list(v) for v in views]) == contract["allocation_views"]
    calls = Counter()

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            calls[node.op.name.replace("tirx.", "tir.")] += 1

    tvm.tirx.stmt_functor.post_order_visit(next(iter(artifact.device_mod.functions.values())).body, visit)
    assert dict(calls) == contract["final_op_calls"]
