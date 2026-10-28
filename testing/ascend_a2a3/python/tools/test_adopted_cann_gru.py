"""Complete original GRU assets: pure host and source contracts."""

import builtins
import ast
import re
import json
import importlib.util
from pathlib import Path
import sys
import pytest
import torch
import tilelang
from tilelang.jit import JITImpl
from tilelang import tvm
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

ASSETS = Path(__file__).parents[4] / "examples/ascend_a2a3/cann-bench"
NAMES = ("gru",)
OWNED_CACHES = {"gru": ("_WS_CACHE", "_WS_CACHE_ORDER")}


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
    with pytest.MonkeyPatch.context() as scoped:
        module._isolate_kernel_module.__wrapped__(tmp_path, scoped)
        fresh = module._KERNEL_MODULE
        assert fresh is not caller and getattr(module, name).__globals__ is fresh.__dict__
        for key, original, _ in snapshots:
            assert getattr(fresh, key) is not original and len(getattr(fresh, key)) == 0
        for key, value in vars(caller).items():
            if isinstance(value, JITImpl):
                assert getattr(fresh, key) is not value
                assert getattr(fresh, key)._kernel_cache == {}
                assert getattr(fresh, key)._call_form_cache is not value._call_form_cache
    assert module._KERNEL_MODULE is caller and getattr(module, name) is alias
    assert all((getattr(caller, key) is value and value == snapshot for key, value, snapshot in snapshots))


@pytest.mark.parametrize("name", ["gru"])
def test_complete_original_cpu_reference_and_precision_reject_mismatch(name):
    module = _module(name)
    gates = 3
    with torch.random.fork_rng(devices=[]):
        x = torch.zeros(3, 2, 4)
        wi = [torch.zeros(gates * 16, 4)]
        wh = [torch.zeros(gates * 16, 16)]
        result = getattr(module, "golden_" + name)(x, wi, wh, inputSize=4, hiddenSize=16, bias=False)
    assert all((torch.equal(value, torch.zeros_like(value)) for value in result))
    actual = result[0].float()
    assert module.check_precision(actual, result[0], actual, "float32")[0]
    assert not module.check_precision(actual + 1, result[0], actual, "float32")[0]


SOURCE_REQUESTS = json.loads(Path(__file__).with_name("data").joinpath("cann_gru_factory_requests.json").read_text())


@pytest.mark.a2a3_native
@pytest.mark.parametrize("source_request", SOURCE_REQUESTS["requests"], ids=lambda row: str(row["id"]) + "-" + row["factory"])
def test_all_original_gru_factory_source_requests(source_request):
    module = _module("gru", test=False)
    factory = getattr(module, source_request["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, *source_request["args"], **source_request["kwargs"])
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target, factory.pass_configs)):
        artifact = tilelang.lower(func, target=target)
    actual = [{"dtype": str(p.dtype).removeprefix("torch."), "shape": [str(d) for d in p.shape]} for p in artifact.params]
    assert actual == source_request["expected_parameters_from_TA"]
    prototype = next((line.strip() for line in artifact.kernel_source.splitlines() if "void call(" in line))
    assert prototype == source_request["expected_host_prototype_from_TA"]


def test_source_fixture_preserves_every_original_entry_and_factory_call():
    data = SOURCE_REQUESTS
    assert len(data["original_entries"]) == 36
    assert len(data["original_factory_request_sequence"]) == 207
    assert sum((row["runtime_void_calls_observed"] for row in data["original_entries"])) == 211
    assert sum((row["repeat_same_input_wrapper_calls"] == 2 for row in data["original_entries"])) == 3
    assert {row["factory"] for row in data["requests"]} == {
        "_gru_pad_kernel",
        "_gru_h0pad_kernel",
        "_gru_wpad_kernel",
        "_gru_gpad_kernel",
        "_gru_xgemm_kernel",
        "_gru_seq_kernel",
        "_gru_scatter_kernel",
        "_gru_gather_kernel",
    }
    assert {row["request_id"] for row in data["original_factory_request_sequence"]} == set(range(127))


@pytest.mark.parametrize("case", SOURCE_REQUESTS["original_negative_cases"], ids=lambda row: row["name"])
def test_original_gru_illegal_inputs_reject_before_factory(case, monkeypatch):
    module = _module("gru")
    kernel_module = module._KERNEL_MODULE

    def forbidden(*args, **kwargs):
        pytest.fail("original illegal input must reject before any JIT factory")

    for name in {row["factory"] for row in SOURCE_REQUESTS["requests"]}:
        monkeypatch.setattr(kernel_module, name, forbidden)
    expected = {"AssertionError": AssertionError, "ValueError": ValueError}[case["type"]]
    with (
        torch.random.fork_rng(devices=[]),
        torch.device("meta"),
        pytest.raises(expected, match="^" + re.escape(case["message"]) + "$") as caught,
    ):
        if case["original_case_if_run_case"] is not None:
            original_case = case["original_case_if_run_case"]
            inputs = module.build_inputs(original_case)
            original_tree = ast.parse(Path(module.__file__).read_text())
            run = next(n for n in original_tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_case")
            call = next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == "_call")
            names = ("dev_x", "dev_ih", "dev_hh", "dev_b_ih", "dev_b_hh", "dev_h0")
            namespace = {"gru": module.gru, "attrs": original_case["attrs"], **dict(zip(names, inputs))}
            exec(compile(ast.Module(body=[call], type_ignores=[]), module.__file__, "exec"), namespace)
            namespace["_call"]()
        else:
            # This original lambda constructs its integer-dtype operands
            # before .npu(); the report/test carrier is a real meta Tensor.
            class MetaTransfer(ast.NodeTransformer):
                def visit_Call(self, node):
                    node = self.generic_visit(node)
                    if isinstance(node.func, ast.Attribute) and node.func.attr == "npu":
                        assert not node.args and not node.keywords
                        return node.func.value
                    return node

            node = MetaTransfer().visit(ast.parse(case["original_lambda"], mode="eval"))
            expression = compile(ast.fix_missing_locations(node), module.__file__, "eval")
            eval(expression, {"torch": torch, "gru": module.gru})()
    assert type(caught.value) is expected
