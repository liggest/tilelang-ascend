"""T.tile.add test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

add_spec = BinaryOpSpec(
    name="add",
    tile_op=T.tile.add,
    golden=operator.add,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_binary_op_tests(add_spec)
