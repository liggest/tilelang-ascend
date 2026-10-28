"""Full original cann-bench transpose source requests and host contracts."""

import ast
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from tilelang import tvm
import tilelang
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

ASSETS = Path(__file__).parents[4] / "examples/ascend_a2a3/cann-bench/transpose"


def _module():
    spec = importlib.util.spec_from_file_location("cann_transpose_contract", ASSETS / "test_transpose.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_requests():
    # Extract only the original immutable scalar case lists; never execute the
    # test functions' tensor allocation, transfer, comparison or kernel calls.
    tree = ast.parse((ASSETS / "test_transpose.py").read_text())
    requests = []
    namespace = {"float": float}
    for function, name, level in [
        ("test_transpose_l0", "test_configs", "l0"),
        (None, "L1_CASES", "l1"),
        ("test_transpose_boundary", "boundary_cases", "boundary"),
    ]:
        body = tree.body if function is None else next(n.body for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        node = next(n for n in body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        values = eval(compile(ast.Expression(node.value), str(ASSETS / "test_transpose.py"), "eval"), {"__builtins__": {}}, namespace)
        for index, row in enumerate(values):
            label, shape, dtype, perm, value_range, *rest = row
            requests.append((f"{level}-{index}-{label}", shape, dtype, perm, value_range))
    requests.append(("standalone", (1024, 1024), "float16", [1, 0], (-1.0, 1.0)))
    return requests


class _FactoryCaptured(Exception):
    pass


def _capture_request(kernel_module, shape, dtype, perm, monkeypatch):
    calls = []
    for name in ("_kernel_2d_transpose", "_kernel_record_swap", "_kernel_stride"):
        factory = getattr(kernel_module, name)

        def capture(*args, _name=name, _factory=factory):
            calls.append((_name, _factory, args))
            raise _FactoryCaptured

        monkeypatch.setattr(kernel_module, name, capture)
    monkeypatch.setattr(kernel_module, "_kernel_cache", {})
    # A real meta Tensor carries the original full shape/dtype through the
    # unchanged host-only view/reshape/classification. No fake kernel or result
    # is returned, and no data/golden comparison is attempted.
    with pytest.raises(_FactoryCaptured):
        kernel_module.transpose(torch.empty(shape, dtype=getattr(torch, dtype), device="meta"), perm)
    assert len(calls) == 1
    return calls[0]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("label,shape,dtype,perm,value_range", _original_requests(), ids=lambda value: str(value))
def test_complete_original_transpose_sources(label, shape, dtype, perm, value_range, monkeypatch):
    module = _module()._KERNEL_MODULE
    _, factory, args = _capture_request(module, shape, dtype, perm, monkeypatch)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, *args)
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target, factory.pass_configs)):
        artifact = tilelang.lower(func, target=target)
    assert len(artifact.params) == 2 and "void call(" in artifact.kernel_source


@pytest.mark.parametrize(
    "perm,message",
    [
        ([0, 1, 2], "perm length 3 != input ndim 2"),
        ([0, 2], "is not a valid permutation"),
        ([0, 0], "is not a valid permutation"),
        ([0, -1], "is not a valid permutation"),
    ],
)
def test_original_illegal_permutations_are_rejected_before_compile(perm, message):
    module = _module()._KERNEL_MODULE
    with pytest.raises(ValueError, match=message):
        module.transpose(torch.empty(3, 4), perm)


def test_identity_is_a_real_cpu_clone_and_original_precision_gates_are_preserved():
    module = _module()
    x = torch.arange(12, dtype=torch.int64).reshape(3, 4)
    y = module.transpose(x, [0, 1])
    assert torch.equal(x, y) and x.data_ptr() != y.data_ptr()
    assert module.check_precision(y, x, "int64") == (True, 1.0, 0.0)
    y[0, 0] = -1
    assert module.check_precision(y, x, "int64")[0] is False
    a = torch.tensor([float("nan"), float("inf"), -float("inf"), 0.0])
    assert module.check_precision(a, a.clone(), "float32")[0] is True
    assert module.check_precision(a, a.flip(0), "float32")[0] is False


def test_help_preserves_caller_path_cache_and_does_not_enter_a_level(monkeypatch, capsys):
    before = list(sys.path)
    module = _module()
    assert sys.path == before
    monkeypatch.setattr(sys, "argv", [str(ASSETS / "test_transpose.py"), "--help"])
    monkeypatch.setattr(tilelang, "disable_cache", lambda: pytest.fail("help must not change cache state"))
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0 and "--level" in capsys.readouterr().out


@pytest.mark.parametrize(
    "level,l0,l1,expected,order",
    [
        ("l0", True, True, 0, ["l0"]),
        ("l1", True, False, 1, ["l1"]),
        ("l2", False, False, 0, ["l2"]),
        ("boundary", False, False, 0, ["boundary"]),
        ("all", True, True, 0, ["l0", "l1", "l2", "boundary"]),
        ("all", False, True, 1, ["l0", "l1", "l2", "boundary"]),
        ("all", True, False, 1, ["l0", "l1", "l2", "boundary"]),
    ],
)
def test_cli_levels_preserve_original_blocking_contract(level, l0, l1, expected, order, monkeypatch, capsys):
    module = _module()
    calls = []
    for name, result in [("l0", l0), ("l1", l1), ("l2", None), ("boundary", None)]:

        def record(_name=name, _result=result):
            calls.append(_name)
            return _result

        monkeypatch.setattr(module, "test_transpose_" + name, record)
    # These stand-ins test CLI scheduling/exit only, never kernel results or
    # precision. The complete original numerical level bodies remain intact.
    disabled = []
    monkeypatch.setattr(tilelang, "disable_cache", lambda: disabled.append(True))
    monkeypatch.setattr(torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(sys, "argv", [str(ASSETS / "test_transpose.py"), "--level", level])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == expected and calls == order and disabled == [True]
    assert ("Test Passed!" in capsys.readouterr().out) is (expected == 0)


def test_standalone_help_does_not_seed_compile_or_launch(monkeypatch, capsys):
    module = _module()._KERNEL_MODULE

    def forbidden(*args, **kwargs):
        pytest.fail("standalone --help must not enter the original device smoke")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(module, "transpose", forbidden)
    with pytest.raises(SystemExit) as result:
        module.main(["--help"])
    assert result.value.code == 0 and "standalone smoke" in capsys.readouterr().out
