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

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/ecosystem_pos_moe_ports.json").read_text())
PYTHON = [e for e in MANIFEST["entries"] if e["destination_path"].endswith(".py")]
ROPE = ROOT / "examples/ascend_a2a3/pos_embedding/RotaryPositionEmbedding/rope_half_interleaved.py"
BENCH = ROPE.with_name("bench.sh")


def _load(path=ROPE):
    spec = importlib.util.spec_from_file_location("_a2a3_pos_moe_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
def test_complete_original_source_request(case):
    import tilelang
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    if tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3", allow_missing=True) is None:
        pytest.skip("A2A3 native source codegen is not built")
    entry = next(e for e in PYTHON if e["source_path"] == case["source_path"])
    module = _load(ROOT / entry["destination_path"])
    factory = getattr(module, case["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    with target:
        function = factory.get_tir(*case["args"], **case["kwargs"])
    configs = normalize_a2a3_pass_configs(target, module.pass_configs)
    with tvm.transform.PassContext(opt_level=3, config=configs):
        artifact = tilelang.lower(function, target=target)
    assert 'extern "C" void call(' in artifact.kernel_source
