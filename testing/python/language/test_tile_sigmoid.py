"""T.tile.sigmoid test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

sigmoid_spec = UnaryOpSpec(
    name="sigmoid",
    tile_op=T.tile.sigmoid,
    golden=torch.sigmoid,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(sigmoid_spec)
