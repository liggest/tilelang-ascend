"""Complete original sparse cann-bench assets: source boundaries and pure host contracts."""

import builtins
import importlib.util
import ast
import json
import re
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest
import torch
import tilelang
from tilelang.jit import JITImpl
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

ASSETS = Path(__file__).parents[4] / "examples/ascend_a2a3/cann-bench"
NAMES = ("sparse_flash_attention",)
OWNED_CACHES = {
    "gru": ("_WS_CACHE", "_WS_CACHE_ORDER"),
    "lstm": ("_ZERO_CACHE",),
    "sparse_flash_attention": ("_kernel_cache", "_bitmap_cache", "_causal_tri_cache", "_dense_fb_logged", "_dense_ok_logged"),
}


def _module(name, test=True):
    path = ASSETS / name / (("test_" if test else "") + name + ".py")
    spec = importlib.util.spec_from_file_location("cann_host_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _BeforeNPUInitialization(ImportError):
    pass


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("test_asset", [False, True])
def test_import_and_help_do_not_compile_query_allocate_or_register_npu(name, test_asset, monkeypatch, capsys):
    original = builtins.__import__
    before_path = list(sys.path)
    attempted = []

    def guarded(module, *args, **kwargs):
        if module.split(".")[0] == "torch_npu":
            attempted.append(module)
            raise _BeforeNPUInitialization
        return original(module, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("import/help cannot compile, seed or allocate tensors")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(tilelang, "compile", forbidden)
    monkeypatch.setattr(tilelang, "lower", forbidden)
    monkeypatch.setattr(tilelang, "disable_cache", forbidden)
    for attr in ("manual_seed", "rand", "randn", "zeros", "empty"):
        monkeypatch.setattr(torch, attr, forbidden)
    module = _module(name, test_asset)
    assert sys.path == before_path
    monkeypatch.setattr(sys, "argv", [module.__file__, "--help"])
    with pytest.raises(SystemExit) as result:
        module.main() if test_asset else module.main(["--help"])
    assert result.value.code == 0 and "usage:" in capsys.readouterr().out
    assert not attempted


@pytest.mark.parametrize("name", NAMES)
def test_level_cli_stops_at_explicit_backend_import_before_cache_or_work(name, monkeypatch):
    module = _module(name)
    original = builtins.__import__
    attempted = []

    def guarded(import_name, *args, **kwargs):
        if import_name.split(".")[0] == "torch_npu":
            attempted.append(import_name)
            raise _BeforeNPUInitialization
        return original(import_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(tilelang, "disable_cache", lambda: pytest.fail("cache changed before registration"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args: pytest.fail("seed changed before registration"))
    monkeypatch.setattr(sys, "argv", [module.__file__, "--level", "l0"])
    with pytest.raises(_BeforeNPUInitialization):
        module.main()
    assert attempted == ["torch_npu"]


@pytest.mark.parametrize("name", NAMES)
def test_invalid_level_exits_before_backend_registration(name, monkeypatch, capsys):
    module = _module(name)
    monkeypatch.setattr(sys, "argv", [module.__file__, "--level", "not-a-level"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 2 and "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("name", NAMES)
def test_real_per_level_namespace_restores_all_caller_caches_and_aliases(name, tmp_path):
    module = _module(name)
    caller = module._KERNEL_MODULE
    alias = getattr(module, name)
    token = object()
    snapshots = []
    for key in OWNED_CACHES[name]:
        value = getattr(caller, key)
        if isinstance(value, dict):
            value["caller"] = token
            snapshot = value.copy()
        elif isinstance(value, list):
            value.append(token)
            snapshot = value.copy()
        else:
            value.add(token)
            snapshot = value.copy()
        snapshots.append((key, value, snapshot))
    if name == "sparse_flash_attention":
        caller._core_num_cached = token
    with pytest.MonkeyPatch.context() as scoped:
        module._isolate_kernel_module.__wrapped__(tmp_path, scoped)
        fresh = module._KERNEL_MODULE
        assert fresh is not caller and getattr(module, name).__globals__ is fresh.__dict__
        for key, original, _ in snapshots:
            assert getattr(fresh, key) is not original and len(getattr(fresh, key)) == 0
        if name == "sparse_flash_attention":
            assert fresh._core_num_cached is None
        # The fixture has switched actual complete modules, not fake kernels or
        # output tensors. No function that allocates/queries an NPU is called.
        for key, value in vars(caller).items():
            if isinstance(value, JITImpl):
                assert getattr(fresh, key) is not value
                assert getattr(fresh, key)._kernel_cache == {}
                assert getattr(fresh, key)._call_form_cache is not value._call_form_cache
    assert module._KERNEL_MODULE is caller and getattr(module, name) is alias
    assert all(getattr(caller, key) is value and value == snapshot for key, value, snapshot in snapshots)
    if name == "sparse_flash_attention":
        assert caller._core_num_cached is token


def test_sparse_cpu_reference_preserves_bottom_right_all_masked_rows():
    module = _module("sparse_flash_attention")
    query = torch.zeros(1, 3, 1, 2)
    key = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
    indices = torch.ones(1, 3, 1, 1, dtype=torch.int32)
    result = module.golden_sparse_flash_attention(query, key, key.clone(), indices, 1.0, "BSND", True)
    expected = torch.tensor([[[[0.0, 0.0]], [[0.0, 0.0]], [[3.0, 4.0]]]], dtype=torch.float64)
    assert torch.equal(result, expected)
    assert module.check_precision(result.float(), expected, "float32")[0]
    assert not module.check_precision(torch.zeros_like(result), expected, "float32")[0]


SOURCE_REQUESTS = json.loads(Path(__file__).with_name("data").joinpath("cann_sparse_factory_requests.json").read_text())


def _source_func(source_request):
    module = _module("sparse_flash_attention", test=False)
    factory = getattr(module, source_request["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, *source_request["args"], **source_request["kwargs"])
    return factory, target, func


@pytest.mark.a2a3_native
@pytest.mark.parametrize("source_request", SOURCE_REQUESTS["requests"], ids=lambda row: str(row["id"]) + "-" + row["factory"])
def test_every_original_positive_factory_source(source_request):
    factory, target, func = _source_func(source_request)
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target, factory.pass_configs)):
        artifact = tilelang.lower(func, target=target)
    actual = [{"dtype": str(p.dtype).removeprefix("torch."), "shape": [str(d) for d in p.shape]} for p in artifact.params]
    assert actual == source_request["expected_parameters_from_TA"]
    assert (
        next(line.strip() for line in artifact.kernel_source.splitlines() if "void call(" in line)
        == source_request["expected_host_prototype_from_TA"]
    )


@pytest.mark.a2a3_native
@pytest.mark.parametrize("source_request", SOURCE_REQUESTS["dense_failures"], ids=lambda row: row["original_entry"])
def test_original_dense_rejection_stage_and_diagnostic_preserved(source_request):
    factory, target, func = _source_func(source_request)
    before = []

    @tvm.instrument.pass_instrument
    class ObservePass:
        def run_before_pass(self, mod, info):
            before.append(info.name)

    with (
        tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target, factory.pass_configs), instruments=[ObservePass()]),
        pytest.raises(tvm.error.InternalError, match="^" + re.escape(source_request["diagnostic_tail"]) + "$") as caught,
    ):
        tilelang.lower(func, target=target)
    assert type(caught.value) is tvm.error.InternalError
    assert before[-1] == source_request["main_first_failure_pass"]


def test_fixture_retains_all_original_entries_factories_and_dense_boundary():
    data = SOURCE_REQUESTS
    assert len(data["original_entries"]) == 21
    assert sum(row["runtime_boundaries"] for row in data["original_entries"]) == 21
    assert len(data["requests"]) == 18 and len(data["dense_failures"]) == 5
    assert {row["factory"] for row in data["requests"]} == {"sparse_flash_attention_fwd", "sparse_flash_attention_fwd_v3"}
    assert {row["factory"] for row in data["dense_failures"]} == {"sparse_flash_attention_fwd_dense"}
    assert len(data["original_negative_cases"]) == 4


@pytest.mark.parametrize("fails", [False, True])
def test_original_core_cache_query_and_except20_protocol_is_cpu_only(fails, monkeypatch):
    module = _module("sparse_flash_attention", test=False)
    calls = []

    def current_device():
        calls.append("current_device")
        return 2

    def properties(device):
        assert device == 2
        calls.append("properties")
        if fails:
            raise RuntimeError("CPU-only original fallback protocol")
        return SimpleNamespace(cube_core_num=24)

    monkeypatch.setattr(
        module, "torch", SimpleNamespace(npu=SimpleNamespace(current_device=current_device, get_device_properties=properties))
    )
    assert module._core_num_cached is None
    expected = 20 if fails else 24
    assert module._ai_core_num() == expected
    assert module._ai_core_num() == expected
    assert calls == ["current_device", "properties"]


@pytest.mark.parametrize("case", SOURCE_REQUESTS["original_negative_cases"], ids=lambda row: row["name"])
def test_original_invalid_input_rejects_before_any_factory(case, monkeypatch):
    module = _module("sparse_flash_attention")

    def forbidden(*args, **kwargs):
        pytest.fail("original invalid input cannot reach a JIT factory")

    for name in {row["factory"] for row in SOURCE_REQUESTS["requests"] + SOURCE_REQUESTS["dense_failures"]}:
        monkeypatch.setattr(module._KERNEL_MODULE, name, forbidden)
    source = ast.parse(Path(module.__file__).read_text())
    node = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "test_sparse_flash_attention_l2")
    namespace = {**vars(module), "sparse_flash_attention": module.sparse_flash_attention}

    class RemoveH2D(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "npu":
                assert not node.args and not node.keywords
                return node.func.value
            return node

    expected = {"ValueError": ValueError, "AssertionError": AssertionError}[case["type"]]
    with torch.random.fork_rng(devices=[]):
        for statement in node.body:
            if isinstance(statement, ast.Assign):
                unit = RemoveH2D().visit(ast.Module(body=[statement], type_ignores=[]))
                exec(compile(ast.fix_missing_locations(unit), module.__file__, "exec"), namespace)
            elif isinstance(statement, ast.AugAssign):
                call = statement.value
                if ast.literal_eval(call.args[0]) != case["name"]:
                    continue
                assert ast.unparse(call.args[1]) == case["original_lambda"]
                expression = RemoveH2D().visit(ast.Expression(call.args[1]))
                fn = eval(compile(ast.fix_missing_locations(expression), module.__file__, "eval"), namespace)
                with pytest.raises(expected, match="^" + re.escape(case["message"]) + "$") as caught:
                    fn()
                assert type(caught.value) is expected
                break
        else:
            pytest.fail("original negative source entry missing")
