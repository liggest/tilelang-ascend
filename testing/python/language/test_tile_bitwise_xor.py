"""T.tile.bitwise_xor test suite."""

import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

bitwise_xor_spec = BinaryOpSpec(
    name="bitwise_xor",
    tile_op=T.tile.bitwise_xor,
    golden=torch.bitwise_xor,
    supported_dtypes=["int16", "int32"],
    boundary_dtypes=["int16", "int32"],
    kernel_scalar=False,
    kernel_inplace=False,
)

register_binary_op_tests(bitwise_xor_spec)
