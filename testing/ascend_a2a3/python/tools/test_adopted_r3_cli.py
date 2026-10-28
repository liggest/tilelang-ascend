"""Do not clear a caller's default cache from the adopted r3 CLI."""

from pathlib import Path
import runpy
import sys

import pytest


@pytest.mark.parametrize("suffix", ["select", "transpose"])
def test_r3_cli_requires_explicit_cache_before_clearing(monkeypatch, suffix):
    path = Path(__file__).parents[1] / f"language/test_tilelang_ascend_language_{suffix}.py"
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", [str(path)])
    with pytest.raises(RuntimeError, match="Set a dedicated TILELANG_CACHE_DIR"):
        runpy.run_path(str(path), run_name="__main__")


@pytest.mark.parametrize("suffix", ["select", "transpose"])
def test_r3_cli_rejects_conflicting_cache_override_before_clearing(monkeypatch, tmp_path, suffix):
    import torch
    import tilelang
    from tilelang.ascend.a2a3.kernel_cache import A2A3CythonKernelCache

    path = Path(__file__).parents[1] / f"language/test_tilelang_ascend_language_{suffix}.py"
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


@pytest.mark.parametrize("suffix", ["select", "transpose"])
def test_original_cli_help_remains_available_without_cache_or_device(monkeypatch, capsys, suffix):
    path = Path(__file__).parents[1] / f"language/test_tilelang_ascend_language_{suffix}.py"
    monkeypatch.delenv("TILELANG_CACHE_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", [str(path), "--help"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(path), run_name="__main__")
    assert result.value.code == 0
    output = capsys.readouterr().out
    assert "--m" in output and "--n" in output and "--dtype" in output and "--target" in output
