"""T.tile.rsqrt test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

rsqrt_spec = UnaryOpSpec(
    name="rsqrt",
    tile_op=T.tile.rsqrt,
    golden=torch.rsqrt,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(rsqrt_spec)
