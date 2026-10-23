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
