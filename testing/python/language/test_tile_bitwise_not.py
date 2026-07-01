"""T.tile.bitwise_not test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

bitwise_not_spec = UnaryOpSpec(
    name="bitwise_not",
    tile_op=T.tile.bitwise_not,
    golden=torch.bitwise_not,
    supported_dtypes=["int16", "int32"],
    boundary_dtypes=["int16", "int32"],
)

register_unary_op_tests(bitwise_not_spec)
