"""Complete MoE asset host protocols and raw source APIs, no NPU execution."""

import builtins
import importlib.util
from pathlib import Path

import pytest
import torch
import tilelang
from tilelang.env import CacheState

ROOT = Path(__file__).parents[4] / "examples/ascend_a2a3/fusedmoe"
FACTORIES = [
    "single_gemm_kernel",
    "dual_gemm_kernel",
    "grouped_gemm_kernel",
    "grouped_dual_gemm_kernel",
    "dual_gemm_kernel_mma",
    "dual_gemm_kernel_mma_auto",
    "single_gemm_kernel_mma_auto",
    "dual_gemm_kernel_mma_dual128",
    "dual_gemm_kernel_fixed",
    "single_gemm_kernel_fixed",
    "grouped_gemm_kernel_fixed",
    "grouped_dual_gemm_kernel_fixed",
]


def _module(test=False):
    path = ROOT / ("test_fusedmoe.py" if test else "example_fusedmoe.py")
    spec = importlib.util.spec_from_file_location("moe_host_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("test", [False, True])
def test_import_help_skip_registration_and_preserve_sys_path_cache_rng(test, monkeypatch, capsys):
    import sys

    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torch_npu":
            pytest.fail("import/help registered NPU")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    before = list(sys.path)
    cpu = torch.random.get_rng_state().clone()
    raw = CacheState._enabled
    module = _module(test)
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0 and "usage:" in capsys.readouterr().out
    assert list(sys.path) == before and CacheState._enabled is raw and torch.equal(cpu, torch.random.get_rng_state())


@pytest.mark.parametrize("test", [False, True])
@pytest.mark.parametrize("conflict", [False, True])
def test_owned_cache_guard_precedes_real_program(test, conflict, tmp_path, monkeypatch):
    module = _module(test)
    if conflict:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "requested"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=message):
        module.main([])


def test_actual_block_metadata_retains_zero_groups_and_partial_blocks():
    module = _module()
    rows, count = module._build_block_metadata([0, 1, 64, 65], 64)
    assert rows == [[1, 0, 1], [2, 64, 64], [3, 128, 64], [3, 192, 1]] and count == 4
    assert module._build_block_metadata([0, 0], 64) == ([], 0)


def test_original_CPU_routing_clamps_topk_to_real_single_expert_without_device():
    module = _module()
    x = torch.arange(4 * 128, dtype=torch.float16, device="cpu").reshape(4, 128)
    router = torch.zeros((1, 128), dtype=torch.float16, device="cpu")
    result = module.host_preprocess(x, router, 4, 64, torch.device("cpu"))
    assert result["group_sizes_list"] == [4] and result["buf_rows"] == 64 and result["total_m_blocks"] == 1
    assert result["block_metadata"].tolist() == [[0, 0, 4]]
    assert torch.equal(result["stacked_tokens"][:4], x)
    assert result["stacked_weights"][:4].tolist() == [1.0] * 4 and torch.count_nonzero(result["stacked_weights"][4:]) == 0


def test_original_CPU_unit_routing_retains_padding_weights_and_empty_metadata():
    module = _module()
    with torch.random.fork_rng(devices=[]):
        result = module.host_preprocess_for_test([0, 1, 64, 65], 128, 4, 64, "cpu")
    assert result["buf_rows"] == 256 and result["total_m_blocks"] == 4
    assert result["block_metadata"].tolist() == [[1, 0, 1], [2, 64, 64], [3, 128, 64], [3, 192, 1]]
    assert int(result["stacked_weights"].sum()) == 130
    with torch.random.fork_rng(devices=[]):
        empty = module.host_preprocess_for_test([0, 0], 128, 2, 64, "cpu")
    assert empty["buf_rows"] == empty["total_m_blocks"] == 0 and empty["block_metadata"].shape == (0,)


def test_original_precision_and_nonblocking_boundary_failure_protocol(capsys):
    module = _module(True)

    def original_failure():
        raise ValueError("CPU protocol error, not a kernel outcome")

    assert module._run_precision("l1", "protocol", original_failure, 1e-2, 1e-2) is False
    assert module._run_boundary("l2", "protocol", original_failure) is None
    text = capsys.readouterr().out
    assert "[PRECISION_FAIL]" in text and "[BOUNDARY_WARN]" in text and "[PRECISION_PASS]" not in text


@pytest.mark.parametrize("enabled", [False, True])
def test_local_pytest_fixture_restores_exact_raw_bit_and_CPU_rng(enabled, monkeypatch):
    module = _module(True)
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    cpu = torch.random.get_rng_state().clone()
    generator = module._restore_example_cache_state.__wrapped__()
    next(generator)
    CacheState._enabled = not enabled
    torch.manual_seed(987)
    generator.close()
    assert CacheState._enabled is enabled and torch.equal(cpu, torch.random.get_rng_state())


def test_original_test_cli_defaults_and_profiler_choices():
    module = _module(True)
    args = module.parse_args([])
    assert (args.level, args.profiler) == ("l0", "do_bench")
    args = module.parse_args(["--level", "bench", "--profiler", "msprof"])
    assert (args.level, args.profiler) == ("bench", "msprof")


@pytest.mark.a2a3_native
@pytest.mark.parametrize("name", FACTORIES)
def test_complete_raw_original_factory_smoke_derived_source_API(name, tmp_path, request):
    from tilelang import tvm

    module = _module()
    factory = getattr(module, name)
    args = (64, 64, 128, 1, 1) if name.startswith("grouped_") else (64, 64, 128)
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, *args)
    with tvm.transform.PassContext(config=factory.pass_configs or {}):
        artifact = tilelang.lower(func, target=target)
    assert "void call(" in artifact.kernel_source and len(artifact.params) >= 3
    path = tmp_path / (name + ".cc")
    path.write_text(artifact.kernel_source)
    request.node.user_properties.append(("fusedmoe_current_source", str(path)))


def test_benchmark_device_arguments_and_original_callback_order_only(monkeypatch):
    module = _module(True)
    x = torch.zeros((2, 4), dtype=torch.float16, device="cpu")
    calls, options = [], []

    def shared(*args):
        calls.append(("shared", args))

    def routed(*args):
        calls.append(("routed", args))

    def protocol(callback, **kwargs):
        options.append(kwargs)
        callback()
        return float(len(options))

    monkeypatch.setattr(module, "do_bench", protocol)
    args = [x] * 10
    assert module._bench_kernels(shared, routed, *args) == (1.0, 2.0, 3.0)
    assert [name for name, _ in calls] == ["shared", "routed", "shared", "routed"]
    assert all(option == {"_n_warmup": 5, "_n_repeat": 5, "return_mode": "mean", "device": x.device} for option in options)


def test_msprof_original_command_and_child_import_prefix_without_subprocess(tmp_path, monkeypatch):
    import ast
    import subprocess

    module = _module(True)
    x = torch.zeros((2, 4), dtype=torch.float16, device="cpu")
    commands = []

    class BeforeRealProcess(BaseException):
        pass

    def stop(command, **kwargs):
        commands.append((command, kwargs))
        raise BeforeRealProcess

    monkeypatch.setattr(subprocess, "run", stop)
    with pytest.raises(BeforeRealProcess):
        module._bench_kernels_msprof(None, None, *([x] * 10), str(tmp_path))
    assert len(list(tmp_path.glob("*.pt"))) == 10
    script = (tmp_path / "prof_script.py").read_text()
    parsed = ast.parse(script)
    imports = [name.name for node in parsed.body if isinstance(node, ast.Import) for name in node.names]
    assert imports.index("torch_npu") < imports.index("tilelang")
    assert "sys.path.insert" not in script and "importlib.util.spec_from_file_location" in script
    assert str(ROOT / "example_fusedmoe.py") in script
    assert script.count("torch.npu.synchronize()") == 3
    command, options = commands[0]
    assert command == [
        "msprof",
        "op",
        f"--application=python {tmp_path}/prof_script.py",
        f"--output={tmp_path}/msprof_result",
        "--aic-metrics=ArithmeticUtilization,Memory,L2Cache,PipeUtilization",
        "--launch-count=10",
        "--warm-up=3",
        "--kernel-name=kernel_kernel",
    ]
    assert options == {"capture_output": True, "text": True, "timeout": 600}
