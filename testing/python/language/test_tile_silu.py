"""T.tile.silu test suite."""

import torch
import torch.nn.functional as F

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

silu_spec = UnaryOpSpec(
    name="silu",
    tile_op=T.tile.silu,
    golden=F.silu,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(silu_spec)
