"""Hardware-free checks of the actual TA shell collector and exit/log contract."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = [
    "examples/bench_test.sh",
    "scripts/ci/path_layout.py",
    "scripts/ci/ci_scope.py",
    "scripts/ci/resolve_operator_tests.py",
    "ci/pytest-ascend.ini",
    "ci/operator_test_manifest.yaml",
]


def _repo(tmp_path):
    repo = tmp_path / "repo"
    for name in FILES:
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    for name in ["examples/ascend_a2a3/ops", "examples_experiment", "testing/ascend_a2a3/python/refactor"]:
        (repo / name).mkdir(parents=True, exist_ok=True)
    (repo / "ci/operator_test_manifest.yaml").write_text("version: 1\nmappings:\n  examples/missing.py: examples/test_missing.py\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    return repo


def _env(output):
    return dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"], TA_CI_ARTIFACT_DIR=str(output))


def test_final_layout_helpers_and_no_main_a5_fallback(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "scripts/ci"))
    import path_layout as layout

    assert layout.pytest_targets(repo) == ["testing/ascend_a2a3/python"]
    assert layout.resolve_path(repo, "examples_experiment/gdn/case.py") == "examples_experiment/gdn/case.py"
    assert layout.resolve_path(repo, "examples_experiment/ascend_a2a3/gdn/case.py") == "examples_experiment/gdn/case.py"
    for relative in [
        "codegen/test_a2a3_codegen.py",
        "language/_a2a3_source_kernels.py",
        "fixtures/a2a3_shape.py",
        "op/a2a3_op_schema.json",
    ]:
        assert layout.resolve_path(repo, "testing/ascend/" + relative) == "testing/ascend_a2a3/python/refactor/" + relative
    with pytest.raises(layout.LayoutError, match="outside"):
        layout.resolve_path(repo, "testing/ascend/runtime/test_dynamic_outputs.py")
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/ci/resolve_operator_tests.py"), "--repo-root", str(repo), "list-tests", "--require-nonempty"],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1 and not result.stdout and "empty Pytest targets" in result.stderr


def test_real_shell_runs_complete_queue_retains_both_streams_and_true_exit(tmp_path):
    repo = _repo(tmp_path)
    scripts = {
        "a.py": "print('stdout retained')\nprint('Kernel Output Match!')\n",
        "b.py": "import sys\nprint('stderr retained', file=sys.stderr)\nprint('Test Passed!', file=sys.stderr)\n",
        "c.py": "import sys\nprint('Kernel Output Match!')\nsys.exit(7)\n",
        "d.py": "print('no success marker')\n",
    }
    for name, text in scripts.items():
        (repo / "examples/ascend_a2a3/ops" / name).write_text(text)
    output = tmp_path / "logs"
    result = subprocess.run(
        ["bash", str(repo / "examples/bench_test.sh"), "--skip-pytest", "--dirs", "ops"],
        env=_env(output),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    records = [json.loads(path.read_text()) for path in sorted((output / "tasks").glob("*/result.json"))]
    assert len(records) == 4
    assert [row["execution_exit_code"] for row in records] == [0, 0, 7, 0]
    assert [row["original_predicate_passed"] for row in records] == [True, True, False, False]
    assert "stdout retained" in (output / "tasks/task-0001/stdout.log").read_text()
    assert "stderr retained" in (output / "tasks/task-0002/stderr.log").read_text()
    assert json.loads((output / "bench_result.json").read_text()) == {"total": 4, "passed": 2, "failed": 2, "bench_exit_code": 1}


def test_public_shell_does_not_mask_pytest_nonzero_with_tee(tmp_path):
    repo = _repo(tmp_path)
    (repo / "examples/ascend_a2a3/ops/a.py").write_text("print('Kernel Output Match!')\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake = fake_bin / "pytest"
    fake.write_text("#!/bin/sh\nprintf '1 failed\\n'\nexit 7\n")
    fake.chmod(0o755)
    env = _env(tmp_path / "logs")
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    result = subprocess.run(
        ["bash", str(repo / "examples/bench_test.sh"), "--dirs", "ops", "--pytest-markers", "not ci_skip"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 7, result.stdout + result.stderr


def test_real_pytest_xml_retains_passed_streams_and_xfail_distinctly(tmp_path):
    repo = _repo(tmp_path)
    script = tmp_path / "test_streams.py"
    script.write_text(
        "import sys,pytest\n"
        "def test_pass():\n print('passed stdout')\n print('passed stderr',file=sys.stderr)\n"
        "def test_fail():\n print('failure stdout')\n assert False\n"
        "def test_xfail():\n pytest.xfail('known original limitation')\n"
    )
    xml = tmp_path / "results.xml"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(repo / "ci/pytest-ascend.ini"), str(script), "--junitxml=" + str(xml)],
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    spec = importlib.util.spec_from_file_location("ci_driver_under_test", ROOT / "maint/ascend_a2a3/ci_batch_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    summary = driver.xml_nodes(xml, tmp_path / "nodes.jsonl")
    assert summary["passed"] == 1 and summary["failure"] == 1 and summary["xfail"] == 1 and summary["skipped"] == 0
    nodes = [json.loads(line) for line in (tmp_path / "nodes.jsonl").read_text().splitlines()]
    passed = next(row for row in nodes if row["name"] == "test_pass")
    assert "passed stdout" in passed["stdout"] and "passed stderr" in passed["stderr"]
    assert next(row for row in nodes if row["outcome"] == "xfail")["verdict_details"][0]["attributes"]["type"] == "pytest.xfail"


def test_process_table_rows_do_not_alias_physical_zero_board_row():
    spec = importlib.util.spec_from_file_location("ci_health_under_test", ROOT / "maint/ascend_a2a3/ci_batch_driver.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    text = (
        "| 0 Ascend910 | OK | power temp |\n"
        "| 0 0 | BUS | 0 memory hbm |\n"
        "| NPU Chip | Process id | Process name |\n"
        "| No running processes found in NPU 0 |\n"
        "| 7 0 | 9000 | unrelated process |\n"
    )
    assert driver.selected_health(text, [0])[0]["physical"] == 0
