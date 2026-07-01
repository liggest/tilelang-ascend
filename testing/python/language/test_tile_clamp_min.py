"""T.tile.clamp_min test suite."""

import math

import pytest
import torch

import tilelang.language as T

from base import BinaryOpSpec, register_binary_op_tests


def _make_clamp_min():
    def _op(dst, src, scalar):
        count = math.prod(dst.shape)
        return T.tile.clamp_min(dst, src, scalar, count)
    return _op


clamp_min_spec = BinaryOpSpec(
    name="clamp_min",
    tile_op=_make_clamp_min(),
    golden=lambda a, s: torch.clamp(a, min=s),
    supported_dtypes=["float16", "float32"],
    kernel_tensor=False,
    kernel_inplace=False,
)

Cases = register_binary_op_tests(clamp_min_spec)


class TestTileClamp_minE2E(Cases.E2E):
    @pytest.mark.parametrize("scalar_val", [0.0, -0.5, 1.0, -100.0])
    @pytest.mark.parametrize("target", ["ascendc", "pto"])
    def test_tensor_op_scalar(self, scalar_val, dtype, target):
        super().test_tensor_op_scalar(scalar_val, dtype, target)
