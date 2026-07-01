"""T.tile.clamp_max test suite."""

import math

import pytest
import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests


def _make_clamp_max():
    def _op(dst, src, scalar):
        count = math.prod(dst.shape)
        return T.tile.clamp_max(dst, src, scalar, count)
    return _op


clamp_max_spec = BinaryOpSpec(
    name="clamp_max",
    tile_op=_make_clamp_max(),
    golden=lambda a, s: torch.clamp(a, max=s),
    supported_dtypes=["float16", "float32"],
    kernel_tensor=False,
    kernel_inplace=False,
)

Cases = register_binary_op_tests(clamp_max_spec)


class TestTileClamp_maxE2E(Cases.E2E):
    @pytest.mark.parametrize("scalar_val", [0.0, 0.5, -1.0, 100.0])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_tensor_op_scalar(self, scalar_val, dtype, target):
        super().test_tensor_op_scalar(scalar_val, dtype, target)
