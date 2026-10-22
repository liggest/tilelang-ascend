"""Target predicates and explicit aliases for the A2/A3 variant."""

from __future__ import annotations

from tvm.target import Target

from tilelang.ascend.target import is_a2a3_target
from tilelang.backend.target import TargetLike, register_target_normalizer


def target_is_ascend_a2a3(target: Target) -> bool:
    return is_a2a3_target(target) and "pto" not in target.keys


def target_is_pto_a2a3(target: Target) -> bool:
    return is_a2a3_target(target) and "pto" in target.keys


def normalize_ascendc_target(target: TargetLike) -> Target | None:
    """The explicit AscendC alias always selects A2/A3, even without a device."""
    if isinstance(target, str) and target.strip() == "ascendc":
        return Target({"kind": "ascend", "arch": "dav-2201"})
    return None


register_target_normalizer("ascendc", normalize_ascendc_target)
