"""Fixed-source RoPE/MoE ports: host contracts and complete source requests."""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_pos_moe_ports.json").read_text())
PYTHON = [e for e in MANIFEST["entries"] if e["destination_path"].endswith(".py")]
ROPE = ROOT / "examples/ascend_a2a3/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py"
BENCH = ROPE.with_name("bench.sh")


def _load(path=ROPE):
    saved = sys.modules.get("moe_token_utils")
    helper_path = path.with_name("moe_token_utils.py")
    try:
        if path.stem != "moe_token_utils" and helper_path.is_file():
            helper_spec = importlib.util.spec_from_file_location("moe_token_utils", helper_path)
            helper = importlib.util.module_from_spec(helper_spec)
            helper_spec.loader.exec_module(helper)
            sys.modules["moe_token_utils"] = helper
        spec = importlib.util.spec_from_file_location("_a2a3_pos_moe_" + path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            sys.modules.pop("moe_token_utils", None)
        else:
            sys.modules["moe_token_utils"] = saved


@pytest.mark.parametrize("entry", PYTHON, ids=lambda e: e["source_path"])
def test_import_has_no_original_runtime(entry, monkeypatch):
    import tilelang
    import torch

    def forbidden(*args, **kwargs):
        raise AssertionError("import performed runtime operation")

    before = sys.modules.get("torch_npu")
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", forbidden)
    for obj, names in [
        (tilelang, ["disable_cache", "compile", "lower"]),
        (torch, ["manual_seed", "device", "randn", "zeros", "empty", "tensor"]),
    ]:
        for name in names:
            monkeypatch.setattr(obj, name, forbidden)
    _load(ROOT / entry["destination_path"])
    assert sys.modules.get("torch_npu") is before


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_rope_runtime_registration_and_cache_restore(enabled, fail, monkeypatch):
    import tilelang
    import torch
    from tilelang.env import CacheState

    module = _load()
    events = []
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    monkeypatch.setattr(torch, "device", lambda name: events.append(name) or "npu")

    def original_runtime():
        assert events == ["npu"]
        assert module.device == "npu"
        tilelang.disable_cache()
        if fail:
            raise ValueError("original runtime failed")
        return 7

    monkeypatch.setattr(module, "_original_main", original_runtime)
    prior = CacheState.is_enabled()
    try:
        CacheState.enable() if enabled else CacheState.disable()
        if fail:
            with pytest.raises(ValueError, match="original runtime failed"):
                module.main()
        else:
            assert module.main() == 7
        assert CacheState.is_enabled() == enabled
    finally:
        CacheState.enable() if prior else CacheState.disable()


@pytest.mark.parametrize("level", ["l0", "l1", "l2", "boundary", "all"])
def test_original_rope_level_dispatch_and_nonblocking_exit(level, monkeypatch):
    module = _load()
    events = []
    monkeypatch.setattr(sys, "argv", [str(ROPE), "--level", level])
    monkeypatch.setattr(module.tilelang, "disable_cache", lambda: None)
    monkeypatch.setattr(module.torch, "manual_seed", lambda seed: events.append(seed))
    for name in ["l0", "l1", "l2", "boundary"]:
        monkeypatch.setattr(module, "test_rope_" + name, lambda n=name: events.append(n) or True)
    with pytest.raises(SystemExit) as exc:
        module._original_main()
    assert exc.value.code == 0
    assert events == [0, 42] + (["l0", "l1", "l2", "boundary"] if level == "all" else [level])


@pytest.mark.parametrize("blocking", ["l0", "l1"])
def test_original_rope_precision_failure_is_nonzero(blocking, monkeypatch):
    module = _load()
    monkeypatch.setattr(sys, "argv", [str(ROPE), "--level", blocking])
    monkeypatch.setattr(module.tilelang, "disable_cache", lambda: None)
    monkeypatch.setattr(module.torch, "manual_seed", lambda seed: None)
    monkeypatch.setattr(module, "test_rope_" + blocking, lambda: False)
    with pytest.raises(SystemExit) as exc:
        module._original_main()
    assert exc.value.code == 1


def test_original_bench_complete_command_matrix_without_profiling(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "calls.jsonl"
    fake = binaries / "msprof"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "with open(os.environ['ROPE_BENCH_CALLS'],'a') as f: "
        "f.write(json.dumps({'argv':sys.argv[1:],'PYTHONPATH':os.environ.get('PYTHONPATH')})+'\\n')\n"
        "print('Task Duration(us): 8.0')\n"
    )
    fake.chmod(0o755)
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"], ROPE_BENCH_CALLS=str(calls))
    output = tmp_path / "results"
    result = subprocess.run(
        ["bash", str(BENCH), "--python", sys.executable, "--output", str(output)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(records) == 26
    for index, shape in enumerate(MANIFEST["R1_bench_shapes"]):
        _, dimensions, layout, dtype = shape.split("|")
        for offset, side in [(0, "tl"), (1, "cann")]:
            record = records[index * 2 + offset]
            assert record["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)
            argv = record["argv"]
            assert argv[:4] == [
                "op",
                "--kernel-name=" + ("kernel_kernel" if side == "tl" else "RotaryPositionEmbedding"),
                "--warm-up=5",
                "--launch-count=20",
            ]
            application = " ".join(argv[5:])
            assert str(ROPE) in application
            assert f"--side {side}" in application
            assert f"--shape {dimensions}" in application
            assert f"--layout {layout} --dtype {dtype}" in application
    with (output / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 13
    assert all(row["tl_us"] == row["cann_us"] == "8.000" and row["speedup"] == "1.00x" for row in summary)
    # The fixture inspects orchestration only. Fake msprof never executes the
    # application, loads any kernel or passes an original golden comparison.


@pytest.mark.parametrize("case", MANIFEST["factory_requests"], ids=lambda r: r["id"])
def test_complete_original_source_request(case, monkeypatch):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    if tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3", allow_missing=True) is None:
        pytest.skip("A2A3 native source codegen is not built")
    entry = next(e for e in PYTHON if e["source_path"] == case["source_path"])
    module = _load(ROOT / entry["destination_path"])
    if entry["component"] == "M1":
        import importlib

        jit_module = importlib.import_module("tilelang.jit")
        captured = []

        def source_compile(self, *args, **kwargs):
            dialect = "pto" if self.target == "pto" else "ascendc"
            target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": [dialect, "ascend"]})
            captured.append((target, self.pass_configs))
            with target:
                return self.get_tir(*args, **kwargs)

        monkeypatch.setattr(jit_module.JITImpl, "compile", source_compile)
        function = getattr(module, case["factory"])(*case["args"], **case["kwargs"])
        assert len(captured) == 1
        target, original_configs = captured[0]
    else:
        factory = getattr(module, case["factory"])
        target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
        with target:
            function = factory.get_tir(*case["args"], **case["kwargs"])
        original_configs = module.pass_configs
    configs = normalize_a2a3_pass_configs(target, original_configs)
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    assert 'extern "C" void call(' in artifact.kernel_source


@pytest.mark.parametrize(
    "case",
    MANIFEST["M1_original_class_constructors"],
    ids=lambda r: r["class"] + ":" + r["dtype"] + ":" + str(r["original_constructor_line"]),
)
def test_complete_original_class_constructor_dispatch(case, monkeypatch):
    entry = next(e for e in PYTHON if e["source_path"] == case["source_path"])
    module = _load(ROOT / entry["destination_path"])
    calls = []
    for name in vars(module):
        if name.startswith("_build_"):

            def capture(*args, factory=name, **kwargs):
                calls.append((factory, list(args), kwargs))
                return object()

            monkeypatch.setattr(module, name, capture)
    instance = getattr(module, case["class"])(**case["kwargs"])
    request = next(r for r in MANIFEST["factory_requests"] if r["id"] == "moe_" + str(case["request_index"]))
    assert calls == [(request["factory"], request["args"], request["kwargs"])]
    for key, value in case["original_derived_state"].items():
        assert getattr(instance, key) == value


@pytest.mark.parametrize(
    "filename,entry,parameterized",
    [
        ("moe_token_permute", "test_permute", "test_permute_parameterized"),
        ("moe_token_permute_grad", "test_permute_grad", "test_permute_grad_parameterized"),
        ("moe_token_unpermute", "test_unpermute", "test_unpermute_parameterized"),
        ("moe_token_unpermute_grad", "test_unpermute_grad", "test_unpermute_grad_parameterized"),
    ],
)
def test_original_moe_failure_summary_preserves_print_only_exit(filename, entry, parameterized, monkeypatch, capsys):
    module = _load(ROOT / "examples/ascend_a2a3/moe_token_permute" / (filename + ".py"))
    calls = []

    def original_comparison_failure(pt_dtype, tl_dtype_str):
        calls.append(tl_dtype_str)
        return False

    monkeypatch.setattr(module, parameterized, original_comparison_failure)
    assert getattr(module, entry)() is None
    assert calls == ["float16", "bfloat16", "float32"]
    assert "Test failed! The precision is not correct!" in capsys.readouterr().out


def test_original_moe_utils_complete_CPU_main(tmp_path):
    path = ROOT / "examples/ascend_a2a3/moe_token_permute/moe_token_utils.py"
    code = (
        "import runpy,torch,sys; assert 'torch_npu' not in sys.modules; assert str(torch.get_default_device())=='cpu'; runpy.run_path("
        + repr(str(path))
        + ",run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Kernel Output Match" in result.stdout
