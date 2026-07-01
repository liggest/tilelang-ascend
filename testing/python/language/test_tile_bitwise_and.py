"""T.tile.bitwise_and test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

bitwise_and_spec = BinaryOpSpec(
    name="bitwise_and",
    tile_op=T.tile.bitwise_and,
    golden=operator.and_,
    supported_dtypes=["int16", "int32"],
    boundary_dtypes=["int16", "int32"],
)

register_binary_op_tests(bitwise_and_spec)
