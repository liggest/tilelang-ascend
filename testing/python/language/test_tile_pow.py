"""T.tile.pow test suite."""

import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests

pow_spec = BinaryOpSpec(
    name="pow",
    tile_op=T.tile.pow,
    golden=torch.pow,
    supported_dtypes=["float16", "float32"],
    kernel_scalar=False,
    kernel_inplace=False,
)

register_binary_op_tests(pow_spec)
