"""T.tile.sqrt test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

sqrt_spec = UnaryOpSpec(
    name="sqrt",
    tile_op=T.tile.sqrt,
    golden=torch.sqrt,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(sqrt_spec)
