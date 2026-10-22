"""A2/A3 lowering entry points.

The variant is registered before its native passes are ported so target
selection can be verified independently. Never lower it through the A5 passes.
"""

from __future__ import annotations

from tilelang.backend.pass_pipeline import PassPipeline


def _not_implemented(_mod, _target):
    raise NotImplementedError("A2/A3 Ascend lowering is not implemented yet")


ASCENDC_PIPELINE = PassPipeline("ascend", _not_implemented)
PTO_PIPELINE = PassPipeline("ascend", _not_implemented)
