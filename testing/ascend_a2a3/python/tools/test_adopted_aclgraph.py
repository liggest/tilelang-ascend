"""Original graph asset import/caller contracts; graph execution is separate."""

import builtins
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang.env import CacheState

ASSET = Path(__file__).parents[4] / "examples/ascend_a2a3/aclgraph/rms_rope_aclgraph.py"


def _module():
    spec = importlib.util.spec_from_file_location("adopted_graph", ASSET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_help_no_seed_cache_compile_or_backend_registration(monkeypatch, capsys):
    original = builtins.__import__
    attempted = []

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "torch_npu":
            attempted.append(name)
            raise AssertionError("import/help cannot register an NPU backend")
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("import/help cannot query devices, seed, compile or alter cache")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(torch, "device", forbidden)
    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(tilelang, "compile", forbidden)
    monkeypatch.setattr(tilelang, "lower", forbidden)
    monkeypatch.setattr(tilelang, "disable_cache", forbidden)
    module = _module()
    assert module.device == "npu"
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0 and "--shape" in capsys.readouterr().out
    assert attempted == []


@pytest.mark.parametrize(
    "args,shape,eps", [([], [16, 64, 512, 256], 1e-6), (["--shape", "1", "32", "128", "64", "--eps", "0.25"], [1, 32, 128, 64], 0.25)]
)
def test_original_argument_parser(args, shape, eps):
    result = _module().parse_args(args)
    assert result.shape == shape and result.eps == eps


def test_malformed_shape_stops_before_backend_or_cache(capsys):
    with pytest.raises(SystemExit) as caught:
        _module().main(["--shape", "1", "2", "3"])
    assert caught.value.code == 2 and "expected 4 arguments" in capsys.readouterr().err


@pytest.mark.parametrize("conflicts", [False, True])
def test_explicit_owned_cache_is_required_before_backend_or_execution(conflicts, tmp_path, monkeypatch):
    module = _module()
    original = builtins.__import__

    def forbidden_import(name, *args, **kwargs):
        if name.split(".")[0] == "torch_npu":
            pytest.fail("cache refusal must precede NPU registration")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbidden_import)
    monkeypatch.setattr(tilelang, "disable_cache", lambda: pytest.fail("refused CLI altered caller cache"))
    monkeypatch.setattr(torch, "manual_seed", lambda *args: pytest.fail("refused CLI seeded caller RNG"))
    if conflicts:
        monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "explicit"))
        monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "caller"))
        message = "conflicts with the active cache-directory override"
    else:
        monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
        message = "Set a dedicated TILELANG_CACHE_DIR"
    with pytest.raises(RuntimeError, match=message):
        module.main([])


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("custom_policy", [False, True])
def test_cli_exception_restores_exact_caller_memos_device_cache_and_CPU_rng(enabled, custom_policy, tmp_path, monkeypatch):
    module = _module()
    token = object()
    module.device = token
    factories = module.rms_norm_kernel, module.rope_kernel_in_place
    canonical_policy_type = type(factories[0]._call_form_cache)

    class CallerPolicy:
        def __init__(self, capacity):
            self.capacity = capacity
            self.entries = {"caller": token}

        def lookup(self, *args, **kwargs):
            pytest.fail("isolated CLI cannot use caller custom lookup policy")

        def store(self, *args, **kwargs):
            pytest.fail("isolated CLI cannot alter caller custom store policy")

    prior = []
    for factory in factories:
        factory._kernel_cache["caller"] = token
        if custom_policy:
            factory._call_form_cache = CallerPolicy(17)
        else:
            factory._call_form_cache.entries["caller"] = token
        policy = factory._call_form_cache
        prior.append(
            (
                factory,
                factory._kernel_cache,
                factory._kernel_cache.copy(),
                policy,
                policy.entries,
                policy.entries.copy(),
                factory.mode,
                factory.func.mode,
            )
        )
    original_import = builtins.__import__
    backend_imports = []

    def fake_backend_import(name, *args, **kwargs):
        if name == "torch_npu":
            backend_imports.append(name)
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    fake_device = object()
    monkeypatch.setattr(builtins, "__import__", fake_backend_import)
    monkeypatch.setattr(module, "torch", SimpleNamespace(device=lambda name: fake_device, random=torch.random))
    monkeypatch.setattr(CacheState, "_enabled", enabled)
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(tmp_path / "owned"))
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", str(tmp_path / "owned"))
    cpu_rng = torch.random.get_rng_state().clone()

    def before_real_graph_execution(args):
        assert module.device is fake_device
        for fresh, caller in zip((module.rms_norm_kernel, module.rope_kernel_in_place), factories):
            assert fresh is not caller and fresh._kernel_cache == {}
            assert type(fresh._call_form_cache) is canonical_policy_type
            assert len(fresh._call_form_cache) == 0
            assert callable(fresh._call_form_cache.lookup) and callable(fresh._call_form_cache.store)
        tilelang.disable_cache()
        torch.manual_seed(42)
        raise RuntimeError("CPU-only boundary before real graph execution")

    monkeypatch.setattr(module, "_run_original", before_real_graph_execution)
    with pytest.raises(RuntimeError, match="CPU-only boundary before real graph execution"):
        module.main([])
    assert backend_imports == ["torch_npu"]
    assert (module.rms_norm_kernel, module.rope_kernel_in_place) == factories
    assert CacheState._enabled is enabled and module.device is token
    assert torch.equal(cpu_rng, torch.random.get_rng_state())
    for factory, memory, snapshot, policy, entries, old_entries, mode, func_mode in prior:
        assert factory._kernel_cache is memory and memory == snapshot
        assert factory._call_form_cache is policy and policy.entries is entries and entries == old_entries
        assert factory.mode == mode and factory.func.mode == func_mode
        if custom_policy:
            assert policy.capacity == 17


def test_original_CPU_references_preserve_rotation_and_normalization():
    module = _module()
    q = torch.ones(2, 2, 4)
    assert torch.equal(module.rms_norm_reference(q, 1e-6), torch.ones_like(q, dtype=torch.float16))
    zeros = torch.zeros(2, 4)
    ones = torch.ones(2, 4)
    assert torch.equal(module.torch_rope_ref(q, zeros, ones), q.half())
    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0]).expand(2, 2, 4).half()
    assert torch.equal(module.torch_rope_ref(q, ones, zeros), expected)


SOURCE_REQUESTS = json.loads(Path(__file__).with_name("data").joinpath("aclgraph_default_requests.json").read_text())["requests"]


@pytest.mark.a2a3_native
@pytest.mark.parametrize("source_request", SOURCE_REQUESTS, ids=lambda row: row["factory"])
def test_original_default_factory_source_and_void_host_ABI(source_request):
    from tilelang import tvm
    from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs

    module = _module()
    factory = getattr(module, source_request["factory"])
    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    func = factory._get_tir_with_target(target, *source_request["args"], **source_request["kwargs"])
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(target, factory.pass_configs)):
        artifact = tilelang.lower(func, target=target)
    actual = [{"dtype": str(p.dtype).removeprefix("torch."), "shape": [str(d) for d in p.shape]} for p in artifact.params]
    assert actual == source_request["TA_expected_params"]
    assert (
        next(line.strip() for line in artifact.kernel_source.splitlines() if "void call(" in line)
        == source_request["TA_expected_host_prototype"]
    )


@pytest.mark.a2a3_native
def test_real_public_factories_use_constructor_cache_lookup_store_and_restore_caller(tmp_path, monkeypatch, request):
    """Real SDK/adapter objects; launch-only providers are isolated as in core tests."""
    import shutil
    from tilelang import tvm
    from tilelang.ascend.a2a3.compiler import resolve_compile_config
    from tilelang.jit.kernel import JITKernel
    from tilelang.jit.adapter.cython import adapter as adapter_module

    target = tvm.target.Target({"kind": "ascend", "arch": "dav-2201", "model": "A3", "keys": ["ascendc", "ascend"]})
    try:
        resolve_compile_config(target)
    except (ValueError, FileNotFoundError) as error:
        pytest.skip(f"requires actual AscendC toolchain/resources: {error}")
    module = _module()
    caller_factories = module.rms_norm_kernel, module.rope_kernel_in_place
    canonical_type = type(caller_factories[0]._call_form_cache)
    token = object()

    class CallerPolicy:
        def __init__(self, capacity):
            self.capacity = capacity
            self.entries = {"caller": token}

        def lookup(self, *args, **kwargs):
            pytest.fail("CLI cannot reuse caller custom policy")

        def store(self, *args, **kwargs):
            pytest.fail("CLI cannot mutate caller custom policy")

    originals = []
    for factory in caller_factories:
        factory._kernel_cache["caller"] = token
        factory._call_form_cache = CallerPolicy(17)
        originals.append((factory._kernel_cache, factory._kernel_cache.copy(), factory._call_form_cache, factory.mode, factory.func.mode))
    owned = str(tmp_path / "owned")
    monkeypatch.setenv("TILELANG_CACHE_DIR", owned)
    monkeypatch.setenv("TL_PLATFORM", "A3")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.setattr(vars(type(tilelang.env))["TILELANG_CACHE_DIR"], "_forced_value", owned)
    # No lowering/codegen/compiler/serialization/load/adapter creation is
    # replaced. Only unused launch validation/providers are CPU-isolated.
    monkeypatch.setattr(adapter_module.CythonKernelAdapter, "_process_buffer_device", lambda self: {})
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda device: 0))
    original_import = builtins.__import__

    def no_backend_registration(name, *args, **kwargs):
        if name == "torch_npu":
            return SimpleNamespace()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backend_registration)
    monkeypatch.setattr(module, "torch", SimpleNamespace(device=lambda name: token, random=torch.random))
    counters = {"lookup": [], "store": []}
    lookup, store = canonical_type.lookup, canonical_type.store

    def actual_lookup(self, *args, **kwargs):
        counters["lookup"].append(id(self))
        return lookup(self, *args, **kwargs)

    def actual_store(self, *args, **kwargs):
        counters["store"].append(id(self))
        return store(self, *args, **kwargs)

    monkeypatch.setattr(canonical_type, "lookup", actual_lookup)
    monkeypatch.setattr(canonical_type, "store", actual_store)
    runtime_entries = []
    kernels = []
    artifacts = []

    class RuntimeObserved(BaseException):
        pass

    class SourceScopeComplete(BaseException):
        pass

    def before_runtime(self, *args, **kwargs):
        assert isinstance(self, JITKernel) and self.adapter.lib.call is not None
        assert all(isinstance(t, torch.Tensor) and t.device.type == "meta" for t in args)
        runtime_entries.append({"shapes": [list(t.shape) for t in args], "dtypes": [str(t.dtype) for t in args]})
        raise RuntimeObserved

    monkeypatch.setattr(JITKernel, "__call__", before_runtime)

    def source_scope(args):
        tilelang.disable_cache()
        for source_request in SOURCE_REQUESTS:
            factory = getattr(module, source_request["factory"])
            assert all(factory is not caller for caller in caller_factories)
            assert type(factory._call_form_cache) is canonical_type
            assert len(factory._call_form_cache) == 0
            kernel = factory(*source_request["args"], **source_request["kwargs"])
            assert isinstance(kernel, JITKernel)
            assert factory._call_form_cache.last_kernel is kernel
            assert len(factory._call_form_cache) == 1
            assert factory(*source_request["args"], **source_request["kwargs"]) is kernel
            kernels.append(kernel)
            for extension, src in [("so", kernel.adapter.libpath), ("cc", kernel.adapter.lib_generator.get_source_path())]:
                dst = tmp_path / (source_request["factory"] + "." + extension)
                shutil.copyfile(src, dst)
                artifacts.append(str(dst))
                if extension == "so":
                    assert dst.read_bytes()[:4] == b"\x7fELF"
            inputs = [
                torch.empty([int(d) for d in p["shape"]], dtype=torch.float16, device="meta") for p in source_request["TA_expected_params"]
            ]
            if source_request["factory"] == "rms_norm_kernel":
                inputs = inputs[:1]
            with pytest.raises(RuntimeObserved):
                kernel(*inputs)
        raise SourceScopeComplete

    original_state = CacheState._enabled
    monkeypatch.setattr(module, "_run_original", source_scope)
    with pytest.raises(SourceScopeComplete):
        module.main([])
    assert (module.rms_norm_kernel, module.rope_kernel_in_place) == caller_factories
    assert CacheState._enabled is original_state
    assert len(kernels) == len(runtime_entries) == 2
    assert len(counters["lookup"]) == 4 and len(counters["store"]) == 2
    for factory, (memory, old, policy, mode, func_mode) in zip(caller_factories, originals):
        assert factory._kernel_cache is memory and memory == old
        assert factory._call_form_cache is policy and policy.capacity == 17 and policy.entries == {"caller": token}
        assert factory.mode == mode and factory.func.mode == func_mode
    request.node.user_properties.append(("actual_public_scope_artifacts", json.dumps(artifacts)))
