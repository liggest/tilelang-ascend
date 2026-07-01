"""T.tile.min test suite."""

import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

min_spec = BinaryOpSpec(
    name="min",
    tile_op=T.tile.min,
    golden=torch.minimum,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_binary_op_tests(min_spec)
