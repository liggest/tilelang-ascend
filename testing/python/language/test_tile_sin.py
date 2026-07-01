"""T.tile.sin test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

sin_spec = UnaryOpSpec(
    name="sin",
    tile_op=T.tile.sin,
    golden=torch.sin,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(sin_spec)
