"""Complete upstream vector examples: original AST, host contracts and real source."""

import ast
import builtins
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch
import tilelang
from tilelang import tvm

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = json.loads((ROOT / "maint/ascend_a2a3/upstream_cann_bench_kkt_ports.json").read_text())
ENTRIES = MANIFEST["entries"]


@pytest.fixture(scope="module", autouse=True)
def _declare_adopted_device_marker(pytestconfig):
    pytestconfig.addinivalue_line("markers", "a2a3_device: original adopted A2/A3 device tests")


def _load(entry):
    path = ROOT / entry["destination_path"]
    spec = importlib.util.spec_from_file_location("upstream_vector_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(name, test=False):
    return next(e for e in ENTRIES if Path(e["source_path"]).name == ("test_" if test else "") + name + ".py")


def _digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def _original_main(node):
    node.decorator_list = [d for d in node.decorator_list if not (isinstance(d, ast.Name) and d.id == "_preserve_cache_state")]
    node.body = [n for n in node.body if not (isinstance(n, ast.Import) and any(a.name == "torch_npu" for a in n.names))]
    return node


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
def test_complete_original_functions_globals_and_CLI(entry):
    path = ROOT / entry["destination_path"]
    tree = ast.parse(path.read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for name, expected in entry["original_function_AST_sha256"].items():
        node = _original_main(functions[name]) if name == "main" else functions[name]
        assert _digest(node) == expected, name
    for expected in entry["original_globals"]:
        node = next(
            n for n in tree.body if isinstance(n, ast.Assign) and [t.id for t in n.targets if isinstance(t, ast.Name)] == expected["names"]
        )
        assert _digest(node) == expected["ast_sha256"]
    main = _original_main(functions["main"])
    assert _digest(ast.Module(body=main.body, type_ignores=[])) == entry["original_CLI_AST_sha256"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"]


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e["source_path"])
def test_import_never_compiles_allocates_seeds_registers_or_changes_caller_path(entry, monkeypatch):
    original_import = builtins.__import__
    before_path = list(sys.path)
    before_npu = sys.modules.get("torch_npu")

    def guard(name, *args, **kwargs):
        assert name.split(".")[0] != "torch_npu", "import must not register the NPU backend"
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("import cannot compile, change cache or allocate/seed tensors")

    monkeypatch.setattr(builtins, "__import__", guard)
    for name in ["compile", "lower", "disable_cache", "enable_cache"]:
        monkeypatch.setattr(tilelang, name, forbidden)
    for name in ["manual_seed", "empty", "zeros", "ones", "full", "rand", "randn"]:
        monkeypatch.setattr(torch, name, forbidden)
    _load(entry)
    assert sys.path == before_path and sys.modules.get("torch_npu") is before_npu


@pytest.mark.parametrize("name", ["dilation_2d", "depthwise_conv_2d", "chunk_scaled_dot_kkt"])
def test_legacy_level_cli_help_is_before_backend_cache_and_work(name, monkeypatch, capsys):
    module = _load(_entry(name, True))
    monkeypatch.setattr(sys, "argv", [module.__file__, "--help"])
    monkeypatch.setattr(tilelang, "disable_cache", lambda: pytest.fail("help changed cache"))
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 0 and "--level" in capsys.readouterr().out


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("failure", [True, False])
def test_original_CLI_cache_decision_is_scoped_on_return_and_failure(enabled, failure):
    from tilelang.env import CacheState

    module = _load(_entry("dilation_2d"))
    sentinel = RuntimeError("original test failed")

    @module._preserve_cache_state
    def original_host_control():
        tilelang.disable_cache()
        if failure:
            raise sentinel
        return "completed"

    prior = CacheState._enabled
    try:
        CacheState._enabled = enabled
        if failure:
            with pytest.raises(RuntimeError) as caught:
                original_host_control()
            assert caught.value is sentinel
        else:
            assert original_host_control() == "completed"
        assert CacheState._enabled is enabled
    finally:
        CacheState._enabled = prior


def test_original_CPU_goldens_preserve_layout_and_closed_form_values():
    dilation = _load(_entry("dilation_2d", True))
    x = torch.arange(1 * 8 * 20 * 32, dtype=torch.float32).reshape(1, 8, 20, 32)
    actual, Ho, Wo = dilation.golden_dilation(x, torch.zeros(3, 3, 32), 1, 1, 1, 1, "SAME")
    assert (Ho, Wo) == (8, 20)
    expected = torch.nn.functional.max_pool2d(x.double().permute(0, 3, 1, 2), 3, stride=1, padding=1).permute(0, 2, 3, 1).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    depthwise = _load(_entry("depthwise_conv_2d", True))
    x = torch.arange(2 * 64 * 16 * 16, dtype=torch.float32).reshape(2, 64, 16, 16) / 1024
    weight, bias = torch.full((64, 1, 1), 2.0), torch.full((64,), 0.5)
    actual = depthwise.golden_depthwise_conv_2d(x, weight, bias, [1, 1], [1, 1], [0, 0], [1, 1], 64)
    torch.testing.assert_close(actual, x * 2 + 0.5, rtol=0, atol=0)
    kkt = _load(_entry("chunk_scaled_dot_kkt", True))
    K, beta, G = torch.ones(1, 64, 1, 64, dtype=torch.bfloat16), torch.full((1, 64, 1), 2.0, dtype=torch.bfloat16), torch.zeros(1, 64, 1)
    expected = (torch.tril(torch.ones(64, 64), diagonal=-1) * 128).to(torch.bfloat16).reshape(1, 64, 1, 64)
    for use_g in [False, True]:
        actual = kkt.golden_chunk_scaled_dot_kkt(K, beta, G, use_g=use_g)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scenario", ["dilation_layout", "dilation_bf16", "depthwise_groups", "depthwise_dtype"])
def test_original_vector_wrapper_rejections_are_host_side(scenario, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an invalid input reached a kernel factory")

    if scenario.startswith("dilation"):
        module = _load(_entry("dilation_2d"))
        monkeypatch.setattr(module, "_get_kernel", forbidden)
        dtype = torch.bfloat16 if scenario == "dilation_bf16" else torch.float32
        layout = "NCHW" if scenario == "dilation_layout" else "NHWC"
        with pytest.raises(ValueError, match="NHWC" if scenario == "dilation_layout" else "unsupported dtype"):
            module.dilation_2d(
                torch.zeros(1, 8, 20, 32, dtype=dtype), torch.zeros(3, 3, 32, dtype=dtype), [1, 1, 1, 1], [1, 1, 1, 1], data_format=layout
            )
    else:
        module = _load(_entry("depthwise_conv_2d"))
        monkeypatch.setattr(module, "_get_kernels", forbidden)
        dtype = torch.int32 if scenario == "depthwise_dtype" else torch.float32
        groups = 1 if scenario == "depthwise_groups" else 3
        with pytest.raises(ValueError, match="groups == C" if scenario == "depthwise_groups" else "unsupported dtype"):
            module.depthwise_conv_2d(
                torch.zeros(1, 3, 4, 4, dtype=dtype),
                torch.zeros(3, 3, 3, dtype=dtype),
                torch.zeros(3, dtype=dtype),
                [3, 3],
                [1, 1],
                [1, 1],
                [1, 1],
                groups,
            )


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"chunk_size": 32}, "must be 64"),
        ({"S": 65}, "multiple"),
        ({"input_dtype": "float16"}, "bfloat16"),
        ({"chunks_per_block": 2}, "exactly 4"),
    ],
)
def test_original_KKT_rejections_happen_before_codegen(kwargs, reason, monkeypatch):
    module = _load(_entry("example_chunk_scaled_dot_kkt"))
    monkeypatch.setattr(importlib.import_module("tilelang.jit"), "compile", lambda *a, **k: pytest.fail("invalid factory compiled"))
    arguments = dict(B=1, S=64, H=1, DK=64)
    arguments.update(kwargs)
    with pytest.raises(AssertionError, match=reason):
        module.chunk_scaled_dot_kkt_fwd(**arguments)


FACTORY_CASES = [
    dict(id="dilation_ab_ry", family="dilation", shape=[2, 64, 64, 64], dtype="float16", strides=[1, 1]),
    dict(id="dilation_ab_s1", family="dilation", shape=[1, 64, 128, 128], dtype="float16", strides=[1, 1]),
    dict(id="dilation_ab_ph", family="dilation", shape=[1, 16, 128, 128], dtype="float32", strides=[2, 2]),
    dict(id="dilation_bridge", family="dilation", shape=[1, 64, 64, 63], dtype="float16", strides=[1, 1]),
    dict(id="dilation_flat", family="dilation", shape=[1, 11, 13, 17], dtype="float32", strides=[1, 1]),
    dict(
        id="depthwise_pad_conv_f16",
        family="depthwise",
        shape=[2, 64, 32, 32],
        dtype="float16",
        kernel=[3, 3],
        padding=[1, 1],
        dilation=[1, 1],
    ),
    dict(
        id="depthwise_compensated_f32",
        family="depthwise",
        shape=[1, 16, 64, 64],
        dtype="float32",
        kernel=[3, 3],
        padding=[2, 2],
        dilation=[2, 2],
    ),
    dict(
        id="depthwise_1x1_bf16", family="depthwise", shape=[2, 64, 16, 16], dtype="bfloat16", kernel=[1, 1], padding=[0, 0], dilation=[1, 1]
    ),
    *[
        dict(id="kkt_" + str(index), family="kkt", parameters=parameters)
        for index, parameters in enumerate(
            [
                [1, 32768, 32, 128, 64, True],
                [1, 32768, 32, 128, 64, False],
                [1, 64, 1, 64, 64, True],
                [1, 64, 1, 64, 64, False],
                [1, 128, 4, 128, 64, True],
                [1, 256, 2, 64, 64, True],
            ]
        )
    ],
]


class _CapturedSource(Exception):
    pass


@pytest.mark.parametrize("case", FACTORY_CASES, ids=lambda c: c["id"])
def test_complete_original_factories_build_real_A2A3_source_without_kernel_launch(case, monkeypatch):
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    if tvm.ffi.get_global_func("target.build.tilelang_ascend_a2a3", allow_missing=True) is None:
        pytest.skip("A2A3 native source builder is not compiled")
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    real_jit = tilelang.jit
    captures = []

    def source_jit(*args, **kwargs):
        return real_jit(*args, **dict(kwargs, target=target))

    def only_source(function, **kwargs):
        assert isinstance(function, tvm.tirx.PrimFunc)
        captures.append((function, kwargs))
        return function

    monkeypatch.setattr(tilelang, "jit", source_jit)
    monkeypatch.setattr(importlib.import_module("tilelang.jit"), "compile", only_source)
    if case["family"] == "kkt":
        module = _load(_entry("example_chunk_scaled_dot_kkt"))
        with target:
            module.chunk_scaled_dot_kkt_fwd(*case["parameters"])
    else:
        name = "dilation_2d" if case["family"] == "dilation" else "depthwise_conv_2d"
        module = _load(_entry(name))
        if name == "dilation_2d":
            original = module._get_kernel

            def capture(*args):
                original(*args)
                raise _CapturedSource

            monkeypatch.setattr(module, "_get_kernel", capture)
            shape, dtype = case["shape"], getattr(torch, case["dtype"])
            sh, sw = case["strides"]
            with target, pytest.raises(_CapturedSource):
                module.dilation_2d(torch.zeros(shape, dtype=dtype), torch.zeros(3, 3, shape[-1], dtype=dtype), [1, sh, sw, 1], [1, 1, 1, 1])
        else:
            original = module._dwconv_1x1 if case["kernel"] == [1, 1] else module._get_kernels

            def capture(*args, **kwargs):
                original(*args, **kwargs)
                raise _CapturedSource

            monkeypatch.setattr(module, "_dwconv_1x1" if case["kernel"] == [1, 1] else "_get_kernels", capture)
            shape, dtype = case["shape"], getattr(torch, case["dtype"])
            C = shape[1]
            with target, pytest.raises(_CapturedSource):
                module.depthwise_conv_2d(
                    torch.zeros(shape, dtype=dtype),
                    torch.zeros((C, *case["kernel"]), dtype=dtype),
                    torch.zeros(C, dtype=dtype),
                    case["kernel"],
                    [1, 1],
                    case["padding"],
                    case["dilation"],
                    C,
                )
    assert len(captures) == (2 if case["family"] == "depthwise" and case["kernel"] != [1, 1] else 1)
    for index, (function, kwargs) in enumerate(captures):
        configs = normalize_a2a3_pass_configs(target, kwargs.get("pass_configs"))
        with tvm.transform.PassContext(opt_level=3, config=configs):
            artifact = tilelang.lower(function, target=target)
        assert 'extern "C" void call(' in artifact.kernel_source
        assert "main_kernel" in artifact.kernel_source or "kernel_kernel" in artifact.kernel_source
        report = os.environ.get("A2A3_VECTOR_SOURCE_REPORT")
        if report:
            folder = Path(report) / case["id"] / str(index)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "kernel.cpp").write_text(artifact.kernel_source)
            from tilelang.ascend.a2a3.compiler import resolve_compile_config

            config = resolve_compile_config(target, pass_configs=kwargs.get("pass_configs"))
            (folder / "compile_config.json").write_text(json.dumps(config.to_metadata(), indent=2) + "\n")
            (folder / "request.json").write_text(json.dumps(case, indent=2) + "\n")
