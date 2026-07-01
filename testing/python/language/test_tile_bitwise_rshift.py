"""T.tile.bitwise_rshift test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

bitwise_rshift_spec = BinaryOpSpec(
    name="bitwise_rshift",
    tile_op=T.tile.bitwise_rshift,
    golden=operator.rshift,
    supported_dtypes=["int16", "int32"],
    boundary_dtypes=["int16", "int32"],
    kernel_tensor=False,
    kernel_inplace=False,
)

register_binary_op_tests(bitwise_rshift_spec)
