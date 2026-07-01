"""T.tile.cos test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

cos_spec = UnaryOpSpec(
    name="cos",
    tile_op=T.tile.cos,
    golden=torch.cos,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(cos_spec)
