"""PTO's original 27 native pass chain, before public stage-2 activation."""

import pytest
from tilelang import tvm
import tilelang.ascend.a2a3.language as T
from tilelang.ascend.a2a3 import codegen, pipeline
from tilelang.ascend.a2a3.pass_config import normalize_a2a3_pass_configs
from tilelang.backend import create_backend_context

from _a2a3_source_kernels import indexer, indexer_dynamic, matmul, matmul_add

_NATIVE = tvm.ffi.get_global_func("tl.transform.A2A3AscendResourceScopeVerify", allow_missing=True) is not None
pytestmark = pytest.mark.skipif(not _NATIVE, reason="complete A2/A3 native compiler is not built")

# Factory order from pinned TA engine/phase.py, excluding the two AscendC-only
# managed-mask factories. HostProcesser is the source Python HostLegalize bridge.
_TA_PASSES = [
    "InferAllocScope",
    "InjectTmpBuffer",
    "AscendVidReduction",
    "BufferShapeCollector",
    "HostLegalize",
    "AscendLowerParallelToVector",
    "LayoutInference",
    "CollectBufferShapes",
    "LowerTileOp",
    "AscendTailMaskPropagation",
    "AscendWorkspaceReduction",
    "LegalizeVectorizedLoop",
    "LegalizeSafeMemoryAccess",
    "CrossCorePipeline",
    "CombineCV",
    "PipelinePlanning",
    "InjectSoftwarePipeline",
    "AscendLowerOpaqueBlock",
    "ConfigIndexBitwidth",
    "Flatten2DBuffer",
    "FlattenBuffer",
    "VectorizeLoop",
    "AscendStorageRewrite",
    "AscendMemoryPlanning",
    "AscendSyncInsert",
    "AscendSyncInsertVS",
    "AscendResourceScopeVerify",
]


@tvm.instrument.pass_instrument
class _Trace:
    def __init__(self):
        self.names = []

    def run_before_pass(self, _mod, info):
        self.names.append(info.name)


def _context(platform="A3"):
    return create_backend_context({"kind": "ascend", "arch": "dav-2201", "model": platform, "keys": ["pto", "ascend"]}, "c", "cython")


@pytest.mark.parametrize(
    "factory,args,configs",
    [
        (matmul, (256, 256, 128, 128, 128, 64), {}),
        (
            matmul_add,
            (256, 256, 128, 128, 128, 64),
            {
                "tl.ascend_auto_sync": True,
                "tl.ascend_auto_cross_core_sync": True,
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_memory_planning": True,
            },
        ),
        (indexer, (2, 1, 32, 512, 4096, 64, 1024, 512, 4, 128, 256, 64), {"tl.ascend_memory_planning": True}),
        (indexer_dynamic, (1, 32, 64, 1024, 512, 4, 128, 256, 64), {"tl.ascend_memory_planning": True}),
    ],
)
def test_source_factories_run_the_complete_pto_pass_sequence(factory, args, configs):
    context = _context()
    trace = _Trace()
    original = factory(*args)
    effective = normalize_a2a3_pass_configs(context.target, configs)
    with tvm.transform.PassContext(opt_level=3, config=effective, instruments=[trace]):
        lowered = pipeline._lower_pto(tvm.IRModule.from_expr(original), context.target)
    func = pipeline.get_device_kernel(lowered)
    names = [name.removeprefix("tl.A2A3") for name in trace.names if name.startswith("tl.A2A3")]
    assert names == _TA_PASSES
    assert func.attrs["npu_platform"] == "A3"
    assert "pto" in func.attrs["target"].keys
    assert "address_map" in func.attrs and "size_map" in func.attrs
    assert len(pipeline._extract_parameters(lowered)) == len(func.params)
    assert original.attrs.get("npu_platform") is None
    assert "tl.LowerIntrin" not in trace.names


@pytest.mark.parametrize("threads", [None, 1, 2])
@pytest.mark.parametrize("platform", ["A2", "A3"])
def test_real_launch_forms_and_pto_codegen_tail(threads, platform):
    @T.prim_func
    def kernel(A: T.Tensor((4,), "int32"), B: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=threads, is_npu=True), T.Scope("C"):
            B[0] = A[0]

    context = _context(platform)
    with tvm.transform.PassContext(config=normalize_a2a3_pass_configs(context.target)):
        lowered = pipeline._lower_pto(tvm.IRModule.from_expr(kernel), context.target)
    trace = _Trace()
    with tvm.transform.PassContext(instruments=[trace]):
        prepared = codegen._prepare_pto_codegen(lowered, context.target)
    assert trace.names == ["tirx.Simplify"]
    assert pipeline.get_device_kernel(prepared).attrs["npu_platform"] == platform


def test_candidate_chain_rejects_the_other_dialect():
    target = create_backend_context({"kind": "ascend", "arch": "dav-2201", "model": "A3"}, "c", "cython").target
    with pytest.raises(ValueError, match="with the PTO dialect key"):
        pipeline._lower_pto(tvm.IRModule({}), target)


def test_missing_pto_native_codegen_has_a_build_diagnostic(monkeypatch):
    original = tvm.ffi.get_global_func

    def lookup(name, allow_missing=False):
        return None if name == "target.build.tilelang_pto_a2a3" else original(name, allow_missing=allow_missing)

    monkeypatch.setattr(tvm.ffi, "get_global_func", lookup)
    with pytest.raises(RuntimeError, match="USE_ASCEND=ON.*missing target.build.tilelang_pto_a2a3"):
        codegen._build_pto(tvm.IRModule({}), _context().target)
