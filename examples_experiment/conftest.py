"""Use the same scoped A2/A3 example fixtures in the experimental namespace."""

import importlib.util
import pytest
from pathlib import Path

_fixture_path = Path(__file__).resolve().parents[1] / "examples" / "ascend_a2a3" / "conftest.py"
_spec = importlib.util.spec_from_file_location("_a2a3_common_example_fixtures", _fixture_path)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)

pytest_configure = _fixtures.pytest_configure
_require_a2a3 = _fixtures._require_a2a3
_restore_example_cache_state = _fixtures._restore_example_cache_state
isolated_a2a3_cache = _fixtures.isolated_a2a3_cache
isolated_npu_default_device = _fixtures.isolated_npu_default_device
disable_a2a3_cache = _fixtures.disable_a2a3_cache


@pytest.fixture
def isolated_autotune_results(isolated_a2a3_cache, monkeypatch):
    """Scope tuner results without clearing a caller's original dictionary."""
    from tilelang.autotuner.tuner import AutoTuner

    # The existing cache fixture supplies the dynamic disk namespace. Only the
    # shared result dictionary needs its own temporary object here.
    monkeypatch.setattr(AutoTuner, "_memory_cache", {})
    yield
