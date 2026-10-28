"""Do not clear a caller's default cache from the adopted arithmetic CLI."""

from pathlib import Path
import runpy
import sys

import pytest


def test_arithmetic_cli_requires_explicit_cache_before_clearing(monkeypatch):
    path = Path(__file__).parents[1] / "language/test_tilelang_ascend_language_tile_arith.py"
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", [str(path)])
    with pytest.raises(RuntimeError, match="Set a dedicated TILELANG_CACHE_DIR"):
        runpy.run_path(str(path), run_name="__main__")


def test_arithmetic_cli_rejects_conflicting_cache_override_before_clearing(monkeypatch, tmp_path):
    import torch
    import tilelang
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    path = Path(__file__).parents[1] / "language/test_tilelang_ascend_language_tile_arith.py"
    requested = tmp_path / "requested"
    resolved = tmp_path / "active-override"
    for directory in (requested, resolved):
        directory.mkdir()
        (directory / "caller-sentinel").write_text("keep")
    descriptor = vars(type(tilelang.env))["TILELANG_CACHE_DIR"]
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(requested))
    monkeypatch.setattr(descriptor, "_forced_value", str(resolved))
    monkeypatch.setattr(sys, "argv", [str(path)])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("conflicting cache paths must stop before clearing or allocating")

    monkeypatch.setattr(A2A3CythonKernelCache, "clear_cache", forbidden)
    monkeypatch.setattr(torch.Tensor, "npu", forbidden, raising=False)
    for name in ("randn", "rand", "randint", "empty", "zeros", "ones", "full", "tensor"):
        monkeypatch.setattr(torch, name, forbidden)
    with pytest.raises(RuntimeError, match="TILELANG_CACHE_DIR conflicts with the active cache-directory override"):
        runpy.run_path(str(path), run_name="__main__")
    assert calls == []
    assert descriptor._forced_value == str(resolved)
    for directory in (requested, resolved):
        assert (directory / "caller-sentinel").read_text() == "keep"
