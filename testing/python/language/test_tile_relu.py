"""T.tile.relu test suite."""

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests

relu_spec = UnaryOpSpec(
    name="relu",
    tile_op=T.tile.relu,
    golden=torch.relu,
    supported_dtypes=["float16", "float32", "int16", "int32"],
)

register_unary_op_tests(relu_spec)
