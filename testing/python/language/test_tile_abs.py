"""T.tile.abs test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

abs_spec = UnaryOpSpec(
    name="abs",
    tile_op=T.tile.abs,
    golden=torch.abs,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_unary_op_tests(abs_spec)
