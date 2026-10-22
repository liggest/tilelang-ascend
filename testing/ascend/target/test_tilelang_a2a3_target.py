"""Hardware-independent coverage for the A2/A3 backend boundary."""

import pytest

from tilelang import tvm
from tilelang.ascend import target as ascend_target
from tilelang.backend import create_backend_context
from tilelang.backend.target import determine_target
from tvm.target import Target


@pytest.mark.parametrize(
    ("platform", "expected_arch"),
    [("A2", "dav-2201"), ("a3", "dav-2201"), ("A5", None)],
)
def test_platform_override(monkeypatch, platform, expected_arch):
    monkeypatch.setenv("TL_PLATFORM", platform)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: None)
    assert ascend_target.detect_a2a3_arch() == expected_arch


def test_invalid_platform_override_is_actionable(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "unknown")
    with pytest.raises(ValueError, match="Invalid TL_PLATFORM.*expected A2, A3, or A5"):
        determine_target("ascend", return_object=True)


@pytest.mark.parametrize(
    ("soc", "expected_arch"),
    [
        ("Ascend910B1", "dav-2201"),
        ("Ascend910_9392", "dav-2201"),
        ("Ascend950PR_9599", None),
        ("Ascend910_9599", None),
        ("unknown", None),
        (None, None),
    ],
)
def test_soc_detection(monkeypatch, soc, expected_arch):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: soc)
    assert ascend_target.detect_a2a3_arch() == expected_arch


@pytest.mark.parametrize(
    ("target", "expected_backend"),
    [
        ("ascend", "ascend_a2a3"),
        ("asc", "ascend_a2a3"),
        ("pto", "pto_a2a3"),
        ("ascendc", "ascend_a2a3"),
        ({"kind": "ascend", "arch": "dav-2201"}, "ascend_a2a3"),
        ({"kind": "ascend", "arch": "dav-2201", "keys": ["pto", "ascend"]}, "pto_a2a3"),
    ],
)
def test_a2a3_target_forms(monkeypatch, target, expected_backend):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    context = create_backend_context(target, "c", "auto")
    assert context.name == expected_backend
    assert context.execution_backend.name == "cython"
    assert str(context.target.attrs["arch"]) == "dav-2201"


def test_explicit_alias_selects_a2a3_without_device(monkeypatch):
    monkeypatch.delenv("TL_PLATFORM", raising=False)
    monkeypatch.setattr(ascend_target, "_get_soc_name", lambda: None)
    assert create_backend_context("ascendc", "c", "auto").name == "ascend_a2a3"
    assert create_backend_context("ascend", "c", "auto").name == "ascend"
    assert create_backend_context("pto", "c", "auto").name == "pto"


def test_a5_and_a2a3_predicates_are_disjoint(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A5")
    assert create_backend_context("ascend", "c", "auto").name == "ascend"
    assert create_backend_context("pto", "c", "auto").name == "pto"
    assert create_backend_context(Target({"kind": "ascend", "arch": "dav-2201"}), "c", "auto").name == "ascend_a2a3"


def test_auto_ascend_detector_selects_a2a3(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A2")
    detected = ascend_target._detect_ascend_target()
    assert detected is not None
    assert create_backend_context(detected, "c", "auto").name == "ascend_a2a3"


def test_pto_execution_backend_is_not_a2a3_pto(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    with pytest.raises(ValueError, match="Allowed: cython"):
        create_backend_context("ascend", "c", "pto")


def test_unported_pipeline_fails_explicitly(monkeypatch):
    monkeypatch.setenv("TL_PLATFORM", "A3")
    context = create_backend_context("ascend", "c", "auto")
    with pytest.raises(NotImplementedError, match="A2/A3 Ascend lowering"):
        context.lower(tvm.IRModule({}))
    with pytest.raises(NotImplementedError, match="A2/A3 Ascend code generation"):
        context.codegen_device(tvm.IRModule({}), compile_device=False)
