"""T.tile.round test suite."""

import math

import torch

import tilelang.language as T

from base import UnaryOpSpec, register_unary_op_tests


def _make_round():
    """Closure: hide count parameter, return a (dst, src) callable."""
    def _op(dst, src):
        count = math.prod(dst.shape)
        return T.tile.round(dst, src, count)
    return _op


round_spec = UnaryOpSpec(
    name="round",
    tile_op=_make_round(),
    golden=torch.round,
    supported_dtypes=["float16", "float32"],
)

register_unary_op_tests(round_spec)
