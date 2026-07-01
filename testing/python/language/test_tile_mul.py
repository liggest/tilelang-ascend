"""T.tile.mul test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

mul_spec = BinaryOpSpec(
    name="mul",
    tile_op=T.tile.mul,
    golden=operator.mul,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_binary_op_tests(mul_spec)
