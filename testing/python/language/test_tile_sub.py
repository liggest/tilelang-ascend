"""T.tile.sub test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

sub_spec = BinaryOpSpec(
    name="sub",
    tile_op=T.tile.sub,
    golden=operator.sub,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_binary_op_tests(sub_spec)
