"""T.tile.bitwise_lshift test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

bitwise_lshift_spec = BinaryOpSpec(
    name="bitwise_lshift",
    tile_op=T.tile.bitwise_lshift,
    golden=operator.lshift,
    supported_dtypes=["int16", "int32"],
    boundary_dtypes=["int16", "int32"],
    kernel_tensor=False,
    kernel_inplace=False,
)

register_binary_op_tests(bitwise_lshift_spec)
