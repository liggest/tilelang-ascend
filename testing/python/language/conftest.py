"""
Pytest configuration and shared fixtures for language API tests.

Fixtures defined here are automatically available to all test files
in testing/python/language/ and its subdirectories — no import needed.
"""

import pytest
import torch
import tilelang


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def disable_tilelang_cache():
    """Disable tilelang JIT cache for the entire test session."""
    tilelang.disable_cache()


@pytest.fixture(autouse=True)
def random_seed():
    """Set deterministic random seed for reproducibility."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Marker registration
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """Register custom markers for test classification."""
    config.addinivalue_line("markers", "l0: Gate test — must pass to merge")
    config.addinivalue_line("markers", "l1: Functional test — must pass to merge")
    config.addinivalue_line("markers", "l2: Boundary/exception test — recommended")
    config.addinivalue_line("markers", "compile_time: Compile-time test, no NPU needed")


def pytest_generate_tests(metafunc):
    """Parametrize 'dtype' from the test class's op spec.

    Each base test class declares a _dtype_source attribute (e.g.
    "supported_dtypes" or "boundary_dtypes") that points to a list on
    the BinaryOpSpec.  This hook reads that list and feeds it to pytest.
    """
    if "dtype" not in metafunc.fixturenames or metafunc.cls is None:
        return
    op = getattr(metafunc.cls, "op", None)
    if op is None:
        return
    source = getattr(metafunc.cls, "_dtype_source", "supported_dtypes")
    dtypes = getattr(op, source, None)
    if dtypes:
        metafunc.parametrize("dtype", dtypes)
