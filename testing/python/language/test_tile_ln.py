"""T.tile.ln test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

ln_spec = UnaryOpSpec(
    name="ln",
    tile_op=T.tile.ln,
    golden=torch.log,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(ln_spec)
