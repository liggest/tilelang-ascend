"""Preserve original GQA perf frontend/host contracts; no device timing."""

import builtins
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[4]
RELATIVE = "examples/ascend_a2a3/gqa_fwd_varlen/perf_gqa_fwd_varlen.py"
SOURCE_REQUESTS = (
    [("default", 8, 64, 16, 2048, 2048, 128, False), ("small", 1, 4, 2, 128, 128, 128, False)]
    + [(f"sq{s}", 8, 64, 16, s, s, 128, False) for s in [512, 1024, 2048, 4096]]
    + [(f"sq{s}_causal", 8, 64, 16, s, s, 128, True) for s in [512, 1024, 2048, 4096]]
)


def _module():
    path = ROOT / RELATIVE
    spec = importlib.util.spec_from_file_location("adopted_gqa_perf", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_actual_helpers(module, monkeypatch):
    # During isolated development, those two build-owned adopted assets live in
    # the review root. Only the test changes lookup location, never the product.
    if not Path(module.__file__).with_name("gqa_fwd_varlen.py").exists():
        review = ROOT.parents[2] / RELATIVE
        assert review.with_name("gqa_fwd_varlen.py").exists()
        monkeypatch.setattr(module, "__file__", str(review))
    module._ensure_helpers()


def test_import_help_and_pure_metadata_do_not_load_runtime_or_mutate_sys_path(monkeypatch, capsys):
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "torch_npu", "tilelang", "tvm"}:
            pytest.fail("import/help loaded a heavy runtime")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    before = list(sys.path)
    module = _module()
    assert module._ceildiv(129, 128) == 2
    assert module.compute_flops(1, 4, 128, 128, 128, True) == 2.0 * 1 * 4 * 128 * 128 * 128
    with pytest.raises(SystemExit) as error:
        module.main(["--help"])
    assert error.value.code == 0 and "causal-sweep" in capsys.readouterr().out
    assert sys.path == before


def test_all_original_cli_defaults_and_explicit_overrides():
    module = _module()
    a = module.parse_args([])
    assert (a.batch, a.heads, a.groups, a.q_seqlen, a.k_seqlen, a.dim) == (8, 64, 16, 2048, 2048, 128)
    assert (a.block_M, a.block_N, a.num_stages, a.cross_interval, a.padding, a.causal, a.with_golden, a.preset) == (
        128,
        128,
        8,
        1,
        "full",
        False,
        False,
        "default",
    )
    a = module.parse_args(["--preset", "causal-sweep", "--cross-interval", "2", "--padding", "third", "--with-golden"])
    assert (a.preset, a.cross_interval, a.padding, a.with_golden) == ("causal-sweep", 2, "third", True)


@pytest.mark.parametrize(
    "preset,expected",
    [
        ("default", SOURCE_REQUESTS[:1]),
        ("small", SOURCE_REQUESTS[1:2]),
        ("sweep", SOURCE_REQUESTS[2:6]),
        ("causal-sweep", SOURCE_REQUESTS[6:]),
    ],
)
def test_original_preset_request_order_and_failure_exit_without_fake_numerical_pass(preset, expected, monkeypatch, capsys):
    module = _module()
    requests = []
    monkeypatch.setattr(module, "torch", SimpleNamespace(float16="dtype-protocol"), raising=False)

    def capture(*args):
        requests.append(args)
        # This is a CLI control protocol check. Explicit False keeps the
        # original failure exit and never manufactures a success/golden.
        return False

    monkeypatch.setattr(module, "run_one", capture)
    with pytest.raises(SystemExit) as error:
        module._run_original(module.parse_args(["--preset", preset]))
    assert error.value.code == 1 and "Test Passed!" not in capsys.readouterr().out
    assert [r[:8] for r in requests] == expected
    assert all(r[8:] == ("full", 128, 128, 8, 1, False, "npu", "dtype-protocol") for r in requests)


@pytest.mark.parametrize("padding", ["full", "random", "third"])
@pytest.mark.parametrize("causal", [False, True])
def test_actual_original_small_CPU_input_and_mask_helpers(padding, causal, monkeypatch):
    import torch

    module = _module()
    before_path = list(sys.path)
    with torch.random.fork_rng(devices=[]):
        _prepare_actual_helpers(module, monkeypatch)
        q, k, v, mask, cq, ck, sq, sk = module.build_inputs(1, 4, 2, 128, 128, 128, causal, padding, "cpu", torch.float16, 128, 128)
        assert q.shape == (1, 4, 128, 128) and k.shape == v.shape == (1, 2, 128, 128)
        assert mask.shape == (1, 128, 128) and mask.device.type == q.device.type == "cpu"
        assert sq == sk == 128 and cq.shape == ck.shape == (2,)
        assert cq.dtype == ck.dtype == torch.int32
        assert 0 < int(cq[-1]) <= 128 and 0 < int(ck[-1]) <= 128
        assert module.flashattn.func.__code__.co_filename.endswith("gqa_fwd_varlen.py")
    assert sys.path == before_path


def test_existing_device_is_forwarded_without_changing_benchmark_counts_or_aggregation(monkeypatch):
    module = _module()
    _prepare_actual_helpers(module, monkeypatch)
    captured = []
    calls = []

    def benchmark(fn, **kwargs):
        captured.append(kwargs)
        fn()
        return 7.0

    monkeypatch.setattr(module, "do_bench", benchmark)
    assert module.bench_tilelang(lambda *args: calls.append(args), "q", "k", "v", "mask", "npu:3") == 7.0
    assert calls == [("q", "k", "v", "mask")]
    monkeypatch.setattr(module, "ref_gqa_varlen_fwd_padded", lambda *args: calls.append("actual-reference-boundary-protocol"))
    monkeypatch.setattr(module, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=lambda: calls.append("synchronize"))))
    assert module.bench_golden("q", "k", "v", "cq", "ck", 4, 2, 128, False, "npu:3") == 7.0
    assert calls[-2:] == ["actual-reference-boundary-protocol", "synchronize"]
    assert captured == [
        {"_n_warmup": 5, "_n_repeat": 5, "return_mode": "mean", "device": "npu:3"},
        {"_n_warmup": 5, "_n_repeat": 10, "return_mode": "median", "device": "npu:3"},
    ]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("case", SOURCE_REQUESTS)
def test_original_perf_preset_factory_requests_full_source(case, monkeypatch, tmp_path, request):
    import tilelang
    from tilelang import tvm

    module = _module()
    _prepare_actual_helpers(module, monkeypatch)
    name, batch, heads, groups, sq, sk, dim, causal = case
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    kwargs = dict(block_M=128, block_N=128, num_stages=8, cross_interval=1, apply_mask=causal)
    func = module.flashattn._get_tir_with_target(target, batch, groups, heads, dim, sq, sk, causal, **kwargs)
    with tvm.transform.PassContext(config=module.flashattn.pass_configs):
        artifact = tilelang.lower(func, target=target)
    assert [int(v) for v in artifact.params[0].shape] == [batch, heads, sq, dim]
    path = tmp_path / (name + ".cc")
    path.write_text(artifact.kernel_source)
    request.node.user_properties.append(("gqa_perf_current_source", str(path)))
