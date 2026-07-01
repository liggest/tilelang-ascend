"""T.tile.bitwise_or test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

bitwise_or_spec = BinaryOpSpec(
    name="bitwise_or",
    tile_op=T.tile.bitwise_or,
    golden=operator.or_,
    supported_dtypes=["int16", "int32"],
    boundary_dtypes=["int16", "int32"],
)

register_binary_op_tests(bitwise_or_spec)
