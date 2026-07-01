"""T.tile.max test suite."""

import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

max_spec = BinaryOpSpec(
    name="max",
    tile_op=T.tile.max,
    golden=torch.maximum,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_binary_op_tests(max_spec)
