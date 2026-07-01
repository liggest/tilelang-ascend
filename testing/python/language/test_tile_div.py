"""T.tile.div test suite."""

import operator

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

div_spec = BinaryOpSpec(
    name="div",
    tile_op=T.tile.div,
    golden=operator.truediv,
    supported_dtypes=["float16", "float32"],
)

register_binary_op_tests(div_spec)
