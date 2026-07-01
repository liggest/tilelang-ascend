"""T.tile.exp test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

exp_spec = UnaryOpSpec(
    name="exp",
    tile_op=T.tile.exp,
    golden=torch.exp,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(exp_spec)
