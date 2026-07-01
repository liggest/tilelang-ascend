"""T.tile.reciprocal test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

reciprocal_spec = UnaryOpSpec(
    name="reciprocal",
    tile_op=T.tile.reciprocal,
    golden=torch.reciprocal,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(reciprocal_spec)
