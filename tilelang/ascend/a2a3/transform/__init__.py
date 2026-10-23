"""Migrated TA pass factories; the complete A2/A3 pipeline is still pending."""

from tilelang import tvm


def BufferShapeCollector():
    return tvm.ffi.get_global_func("tl.transform.A2A3BufferShapeCollector")()


def Flatten2DBuffer():
    return tvm.ffi.get_global_func("tl.transform.A2A3Flatten2DBuffer")()


def CollectBufferShapes():
    return tvm.ffi.get_global_func("tl.transform.A2A3CollectBufferShapes")()


def AscendLowerOpaqueBlock():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendLowerOpaqueBlock")()


def LegalizeSafeMemoryAccess():
    return tvm.ffi.get_global_func("tl.transform.A2A3LegalizeSafeMemoryAccess")()


def ConfigIndexBitwidth():
    return tvm.ffi.get_global_func("tl.transform.A2A3ConfigIndexBitwidth")()


def LegalizeVectorizedLoop():
    return tvm.ffi.get_global_func("tl.transform.A2A3LegalizeVectorizedLoop")()


def FlattenBuffer():
    return tvm.ffi.get_global_func("tl.transform.A2A3FlattenBuffer")()


def AscendInferBufferScope():
    # The native pass calls the backend's default tagged zN layout factory.
    from .. import layout  # noqa: F401

    return tvm.ffi.get_global_func("tl.transform.A2A3InferAllocScope")()


def LayoutInference():
    return tvm.ffi.get_global_func("tl.transform.A2A3LayoutInference")()


def InjectTmpBuffer(target, plan_vid_reduction: bool = False):
    return tvm.ffi.get_global_func("tl.transform.A2A3InjectTmpBuffer")(target, plan_vid_reduction)


def AscendVidReduction():
    return tvm.ffi.get_global_func("tl.transform.A2A3AscendVidReduction")()


def LowerTileOp():
    return tvm.ffi.get_global_func("tl.transform.A2A3LowerTileOp")()
