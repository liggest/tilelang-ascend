"""Actual shared-library restore for the final A2A3 device argument contract."""

import subprocess
from pathlib import Path

import pytest
import torch
from tilelang import tvm
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.runtime import dump_runtime_signature, load_runtime_signature
from tilelang.engine.lower import extrac_params
from tilelang.jit.adapter.base import CachedTextSource
from tilelang.jit.adapter.cython import adapter as adapter_module
from tilelang.jit.adapter.cython.adapter import CythonKernelAdapter
from tilelang.jit.adapter.libgen import LibraryGenerator
from tilelang.jit.adapter.wrapper import TLWrapper


tir = tvm.tirx
_TARGET = {"kind": "ascend", "arch": "dav-2201", "keys": ["ascend"], "model": "A3"}


def _snapshot(monkeypatch, tmp_path):
    # This suite tests restoration, not a fake successful Bisheng compilation.
    # The real compiler resolver builds its complete immutable configuration;
    # no compiler binary is invoked through these placeholders.
    monkeypatch.setattr(compiler, "_get_ascend_home_path", lambda **kwargs: str(tmp_path / "sdk"))
    monkeypatch.setattr(compiler, "_get_tl_root", lambda **kwargs: str(Path(__file__).parents[3]))
    monkeypatch.setattr(compiler.shutil, "which", lambda *args, **kwargs: "/unused/restored/bisheng")
    monkeypatch.setenv("TL_RUN_MODE", "npu")
    monkeypatch.delenv("TL_CCE_OPT_LEVEL", raising=False)
    return compiler.resolve_compile_config(tvm.target.Target(_TARGET))


def _library(tmp_path, source):
    path = tmp_path / "launcher.so"
    subprocess.run(["c++", "-shared", "-fPIC", "-x", "c++", "-", "-o", str(path)], input=source, text=True, capture_output=True, check=True)
    return path


def _signature():
    n = tir.Var("N", "int32")
    buffers = [
        tir.decl_buffer(shape, "float32", name=name)
        for name, shape in [("workspace", (n * 2,)), ("input", (n,)), ("output", (n,)), ("auto_gm", (16,))]
    ]
    func = tir.PrimFunc([buffer.data for buffer in buffers], tir.Evaluate(0), buffer_map={buffer.data: buffer for buffer in buffers})
    func = func.with_attr("global_symbol", "main").with_attr("calling_conv", tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH)
    return func.with_attr("auto_gm_indices", [3])


@pytest.fixture(autouse=True)
def _cpu_host_abi_allocator(monkeypatch):
    monkeypatch.setattr(adapter_module, "_device_providers", lambda: (lambda: torch.device("cpu"), lambda device: 0))


def test_restored_adapter_allocates_final_workspace_and_gm_with_frozen_driver(monkeypatch, tmp_path):
    snapshot = _snapshot(monkeypatch, tmp_path)
    path = _library(
        tmp_path,
        """
    #include <stdint.h>
    extern "C" void call(float *workspace, float *input, float *output, float *gm, int64_t n, void *stream) {
      gm[0] = 3;
      for (int64_t i=0; i<n; ++i) { workspace[2*i] = input[i] * 2; output[i] = workspace[2*i] + gm[0]; }
    }
    """,
    )
    final = load_runtime_signature(dump_runtime_signature(_signature()))
    buffers = {param: final.buffer_map[param] for param in list(final.params)[:3]}
    original = tir.PrimFunc(list(final.params)[:3], tir.Evaluate(0), buffer_map=buffers).with_attr("global_symbol", "main")
    # Deliberately change process options after resolution. Restore must use the
    # real driver's metadata path without reparsing these new values.
    monkeypatch.setenv("TL_PLATFORM", "A2")
    monkeypatch.setenv("TL_RUN_MODE", "sim")
    monkeypatch.setenv("TL_CCE_OPT_LEVEL", "0")
    monkeypatch.setattr(
        compiler, "resolve_compile_config", lambda *args, **kwargs: pytest.fail("restore must not resolve compiler options twice")
    )
    adapter = CythonKernelAdapter.from_database(
        params=extrac_params(final),
        result_idx=-2,
        workspace_idx=-4,
        target=tvm.target.Target(_TARGET),
        func_or_mod=original,
        runtime_func=final,
        host_kernel_source=CachedTextSource(text="source owned by codegen"),
        device_kernel_source=CachedTextSource(text="device source"),
        kernel_lib_path=str(path),
        compile_config=compiler.CompileConfig.from_metadata(snapshot.to_metadata()),
    )
    assert adapter.workspace_idx == [0]
    assert adapter.auto_gm_idx == [3]
    assert len(adapter.params) == len(adapter.prim_func.params) == 4
    assert adapter.lib_generator.compile_config.cache_identity == snapshot.cache_identity
    assert adapter.libpath == str(path)
    assert adapter.lib.call.restype is None
    source = torch.arange(9, dtype=torch.float32)
    output = adapter(source, stream=0, skip_tensor_validation=True)
    torch.testing.assert_close(output, source * 2 + 3)
    assert list(adapter.dynamic_symbolic_map.values()) == [(0, 1, 0, 1)]


def test_a2a3_wrapper_keeps_complete_host_source_verbatim():
    source = 'extern "C" void call(float *x, void *stream) { x[0] = 1; }'
    wrapper = TLWrapper(tvm.target.Target(_TARGET))
    wrapper.assign_optimized_module(tvm.IRModule({"main": _signature()}))
    assert wrapper.wrap(source) == source


def test_shared_libgen_load_retains_path_and_does_not_resolve_again(monkeypatch, tmp_path):
    snapshot = _snapshot(monkeypatch, tmp_path)
    path = _library(tmp_path, 'extern "C" void call() {}')
    monkeypatch.setattr(compiler, "resolve_compile_config", lambda *args, **kwargs: pytest.fail("frozen snapshot must be reused"))
    generator = LibraryGenerator(tvm.target.Target(_TARGET), compile_config=snapshot)
    assert generator.load_lib(str(path)).call is not None
    assert generator.get_lib_path() == str(path)
    generator.assign_compile_flags(list(snapshot.flags))
    with pytest.raises(ValueError, match="frozen"):
        generator.assign_compile_flags(["-O0"])


def test_final_parameter_count_is_required_before_library_loading(tmp_path):
    final = _signature()
    with pytest.raises(ValueError, match="final device kernel"):
        CythonKernelAdapter.from_database(
            params=extrac_params(final)[:3],
            result_idx=2,
            target=tvm.target.Target(_TARGET),
            func_or_mod=final,
            host_kernel_source=CachedTextSource(text=""),
            device_kernel_source=CachedTextSource(text=""),
            kernel_lib_path=str(tmp_path / "missing.so"),
        )


def test_frozen_ascendc_snapshot_cannot_be_used_with_pto_target(monkeypatch, tmp_path):
    snapshot = _snapshot(monkeypatch, tmp_path)
    final = _signature()
    target = tvm.target.Target({**_TARGET, "keys": ["pto", "ascend"]})
    with pytest.raises(ValueError, match="same target dialect keys"):
        CythonKernelAdapter.from_database(
            params=extrac_params(final),
            result_idx=2,
            target=target,
            func_or_mod=final,
            runtime_func=final,
            host_kernel_source=CachedTextSource(text=""),
            device_kernel_source=CachedTextSource(text=""),
            kernel_lib_path=str(tmp_path / "missing.so"),
            compile_config=snapshot,
        )


def test_a2a3_runtime_does_not_append_independent_stride_symbols():
    n, stride = tir.Var("N", "int32"), tir.Var("stride", "int32")
    buffer = tir.decl_buffer((n,), "float32", name="input", strides=(stride,))
    func = tir.PrimFunc([buffer.data], tir.Evaluate(0), buffer_map={buffer.data: buffer}).with_attr("global_symbol", "main")
    adapter = CythonKernelAdapter.__new__(CythonKernelAdapter)
    adapter.ir_module = tvm.IRModule({"main": func})
    adapter.params = extrac_params(func)
    adapter.result_idx = []
    adapter._a2a3_runtime = True
    adapter._automatic_indices = set()
    adapter.target = tvm.target.Target(_TARGET)
    assert list(adapter._process_dynamic_symbolic()) == [n]
    adapter._a2a3_runtime = False
    assert list(adapter._process_dynamic_symbolic()) == [n, stride]


def test_new_process_restore_preserves_host_b_s1_s2_shape_argument_order(monkeypatch, tmp_path):
    import json
    import os
    import sys
    import tilelang
    import tilelang_cython_wrapper
    from tilelang.engine.param import dump_kernel_params

    snapshot = _snapshot(monkeypatch, tmp_path)
    b, s1, s2 = [tir.Var(name, "int32") for name in ("B", "S1", "S2")]
    buffers = [
        tir.decl_buffer(shape, dtype, name=name)
        for name, shape, dtype in [
            ("Query", (b, s1, 1, 2048), "float16"),
            ("KEY", (b, s2, 1, 64), "float16"),
            ("QK_RES", (b, 1, s1, 32, s2), "float32"),
            ("WEIGHTS", (b, s1, 1, 32), "float32"),
            ("OUT", (b, 1, s1, 1024), "int32"),
        ]
    ]
    func = tir.PrimFunc([buffer.data for buffer in buffers], tir.Evaluate(0), buffer_map={buffer.data: buffer for buffer in buffers})
    func = func.with_attr("global_symbol", "main").with_attr("auto_gm_indices", [])
    # Match TA PrintHostFunc: explicit params first, then its shape_vars in
    # params/shape traversal order B, S1, S2, followed by the stream.
    source = """
    #include <stdint.h>
    using aclrtStream = void*;
    extern "C" void call(uint8_t *Query_handle, uint8_t *KEY_handle, uint8_t *QK_RES_handle,
                        uint8_t *WEIGHTS_handle, uint8_t *OUT_handle,
                        int64_t B, int64_t S1, int64_t S2, aclrtStream stream) {
      auto output = reinterpret_cast<int32_t*>(OUT_handle);
      output[0] = B; output[1] = S1; output[2] = S2;
      reinterpret_cast<float*>(QK_RES_handle)[0] = 1;
    }
    """
    _library(tmp_path, source)
    (tmp_path / "host.asc").write_text(source)
    (tmp_path / "signature.json").write_text(dump_runtime_signature(func))
    (tmp_path / "params.json").write_text(dump_kernel_params(extrac_params(func)))
    (tmp_path / "compiler.json").write_text(json.dumps(snapshot.to_metadata()))
    child = r"""
import importlib.util,json,sys
from pathlib import Path
import tvm
extension=sys.argv[2]
spec=importlib.util.spec_from_file_location('tilelang_cython_wrapper',extension)
module=importlib.util.module_from_spec(spec);sys.modules['tilelang_cython_wrapper']=module;spec.loader.exec_module(module)
import torch
from tilelang.ascend.a2a3 import compiler
from tilelang.ascend.a2a3.runtime import load_runtime_signature
from tilelang.engine.param import load_kernel_params
from tilelang.jit.adapter.base import CachedTextSource
from tilelang.jit.adapter.cython import adapter as impl
folder=Path(sys.argv[1])
config=compiler.CompileConfig.from_metadata(json.loads((folder/'compiler.json').read_text()))
compiler.resolve_compile_config=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('restore resolved options twice'))
impl._device_providers=lambda:(lambda:torch.device('cpu'),lambda device:0)
func=load_runtime_signature((folder/'signature.json').read_text())
adapter=impl.CythonKernelAdapter.from_database(params=load_kernel_params((folder/'params.json').read_text()),result_idx=-1,workspace_idx=-3,target=tvm.target.Target({'kind':'ascend','arch':'dav-2201','model':'A3','keys':['ascend']}),func_or_mod=func,runtime_func=func,host_kernel_source=CachedTextSource(path=str(folder/'host.asc')),device_kernel_source=CachedTextSource(path=str(folder/'host.asc')),kernel_lib_path=str(folder/'launcher.so'),compile_config=config)
output=adapter(torch.ones(2,128,1,2048,dtype=torch.float16),torch.ones(2,2048,1,64,dtype=torch.float16),torch.ones(2,128,1,32),stream=0,skip_tensor_validation=True)
print('A2A3_RESTORE_CAPTURE='+json.dumps({'shape_order':[str(var) for var in adapter.dynamic_symbolic_map],'source_slots':[list(value)for value in adapter.dynamic_symbolic_map.values()],'actual_native_values':output.flatten()[:3].tolist(),'workspace_idx':adapter.workspace_idx,'auto_gm_idx':adapter.auto_gm_idx,'platform':config.platform,'run_mode':config.run_mode}))
"""
    environment = os.environ.copy()
    environment.update(TL_PLATFORM="A2", TL_RUN_MODE="sim", TL_CCE_OPT_LEVEL="0")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(tilelang.__file__).parents[1]), str(Path(tvm.__file__).parents[1]), environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path), tilelang_cython_wrapper.__file__],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    capture = json.loads(
        next(line.removeprefix("A2A3_RESTORE_CAPTURE=") for line in result.stdout.splitlines() if line.startswith("A2A3_RESTORE_CAPTURE="))
    )
    assert capture == {
        "shape_order": ["B", "S1", "S2"],
        "source_slots": [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
        "actual_native_values": [2, 128, 2048],
        "workspace_idx": [2],
        "auto_gm_idx": [],
        "platform": "A3",
        "run_mode": "npu",
    }
