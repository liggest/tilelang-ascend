"""Compile-boundary oracle contracts; no original numerical/runtime verdicts."""

import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "examples_experiment/deepseek_nsa/example_tilelang_nsa_fwd/test_example_tilelang_nsa_fwd.py"
PREFIX = "A2A3_ODD_G_RESULT "
DIAGNOSTIC = "CompareScalar alignment error: size=544, element_bytes=4, total_bytes=2176 byte is not 256-byte aligned."


@pytest.fixture
def oracle():
    spec = importlib.util.spec_from_file_location("_a2a3_nsa_oracle_contract", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(**changes):
    return {
        "phase": "target_compile",
        "kind": "expected_compile_rejection",
        "error_type": "InternalError",
        "message": DIAGNOSTIC,
        **changes,
    }


def _output(record):
    return (PREFIX + json.dumps(record) + "\n").encode()


@pytest.mark.parametrize(
    "status,records,expected",
    [
        (71, [_record()], True),
        (0, [_record()], False),
        (1, [_record()], False),
        (-11, [_record()], False),
        (71, [], False),
        (71, [_record(), _record()], False),
        (71, [{**_record(), "phase": "setup_imports"}], False),
        (71, [{**_record(), "phase": "setup_CPU"}], False),
        (71, [{**_record(), "kind": "unexpected_compile_acceptance"}], False),
        (71, [{**_record(), "error_type": "ValueError"}], False),
        (71, [{**_record(), "message": "CompareScalar alignment error: size=512"}], False),
        (71, [{**_record(), "message": DIAGNOSTIC.replace("total_bytes=2176", "total_bytes=21760")}], False),
        (71, [{**_record(), "message": DIAGNOSTIC.replace("size=544,", "size=5440,")}], False),
        (71, [{**_record(), "message": DIAGNOSTIC.replace("256-byte aligned", "512-byte aligned")}], False),
        (71, [{**_record(), "message": None}], False),
        (71, [{**_record(), "unreviewed": True}], False),
        (71, [["unexpected JSON list"]], False),
    ],
)
def test_only_explicit_single_compile_rejection_can_pass(oracle, status, records, expected):
    result = SimpleNamespace(returncode=status, stdout=b"ordinary compiler log\n" + b"".join(_output(r) for r in records))
    assert oracle._odd_gqa_compile_result_is_expected(result) is expected


@pytest.mark.parametrize("output", [None, b"A2A3_ODD_G_RESULT {not-json}\n", b"A2A3_ODD_G_RESULT \xff\n"])
def test_missing_or_malformed_marker_fails(oracle, output):
    assert not oracle._odd_gqa_compile_result_is_expected(SimpleNamespace(returncode=71, stdout=output))


def test_child_protocol_uses_real_sibling_argv_cwd_and_original_timeout(oracle, monkeypatch):
    calls = []

    def capture(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=71, stdout=_output(_record()), stderr=b"")

    monkeypatch.setattr(subprocess, "run", capture)
    assert oracle._run_odd_gqa_compile_probe()
    argv, options = calls[0]
    assert argv[:2] == [sys.executable, "-c"]
    assert argv[3] == str(SOURCE.resolve().parent)
    assert options == {"capture_output": True, "timeout": 120, "cwd": str(SOURCE.resolve().parent)}


def test_timeout_remains_a_failure(oracle, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 120)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        oracle._run_odd_gqa_compile_probe()


def test_missing_sibling_fails_before_same_name_decoy_import(oracle, tmp_path):
    directory = tmp_path / "partial"
    directory.mkdir()
    (directory / "example_tilelang_nsa_fwd.py").write_text("raise AssertionError('partial kernel was imported')\n")
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    marker = tmp_path / "decoy-imported"
    (decoy / "test_example_tilelang_nsa_fwd.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    result = subprocess.run(
        [sys.executable, "-c", oracle._odd_gqa_compile_script(), str(directory)],
        env=dict(os.environ, PYTHONPATH=str(decoy)),
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 72
    assert not oracle._odd_gqa_compile_result_is_expected(result)
    assert not marker.exists()
    record = json.loads(next(line[len(PREFIX) :] for line in result.stdout.decode().splitlines() if line.startswith(PREFIX)))
    assert record["phase"] == "setup_imports"
    assert record["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize("mode", ["accept", "other_compile_error", "setup_import_error"])
def test_child_classifies_control_failures_before_h2d(oracle, tmp_path, mode):
    # This fixture tests only oracle control flow. It returns no tensor/kernel
    # output and runs no original golden. CPU preparation uses real helpers.
    helper = "import torch\n\n" + inspect.getsource(oracle.prepare_inputs) + "\n" + inspect.getsource(oracle._to_3d_inputs)
    (tmp_path / "test_example_tilelang_nsa_fwd.py").write_text(helper)
    if mode == "setup_import_error":
        kernel = "raise ImportError('explicit oracle import fixture')\n"
    else:
        kernel = (
            "import torch\ndef forbid_h2d(*args, **kwargs):\n raise AssertionError('oracle reached forbidden H2D')\n"
            "torch.Tensor.npu = forbid_h2d\ndef native_sparse_attention(**kwargs):\n"
            + (" raise ValueError(" + repr(DIAGNOSTIC) + ")\n" if mode == "other_compile_error" else " return object()\n")
        )
    (tmp_path / "example_tilelang_nsa_fwd.py").write_text(kernel)
    # The parent preload does not cross exec. Supply the real OFF loader in
    # this control-only fixture; the production child still fails on setup
    # errors and is never taught to accept missing-native environments.
    bootstrap = (
        "import os\nif os.environ.get('SKIP_LOADING_TILELANG_SO') == '1':\n"
        " import tvm, ctypes\n"
        " ctypes.CDLL(os.path.join(os.environ['TVM_LIBRARY_PATH'], 'libtilelang.so'), mode=ctypes.RTLD_GLOBAL)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap + oracle._odd_gqa_compile_script(), str(tmp_path)],
        cwd=tmp_path,
        env=dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        timeout=60,
    )
    (tmp_path / "child_probe_result.json").write_text(
        json.dumps(
            {
                "returncode": result.returncode,
                "stdout": result.stdout.decode(errors="replace"),
                "stderr": result.stderr.decode(errors="replace"),
            },
            indent=2,
        )
    )
    assert result.returncode == 72, result.stdout + result.stderr
    assert not oracle._odd_gqa_compile_result_is_expected(result)
    output = result.stdout.decode()
    record = json.loads(next(line[len(PREFIX) :] for line in output.splitlines() if line.startswith(PREFIX)))
    assert "forbidden H2D" not in record["message"]
    if mode == "accept":
        assert record["phase"] == "target_compile"
        assert record["kind"] == "unexpected_compile_acceptance"
    elif mode == "other_compile_error":
        assert record["phase"] == "target_compile"
        assert record["kind"] == "unexpected_error"
        assert record["error_type"] == "ValueError"
    else:
        assert record["phase"] == "setup_imports"
        assert record["error_type"] == "ImportError"
