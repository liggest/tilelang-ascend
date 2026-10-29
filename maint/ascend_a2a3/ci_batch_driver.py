"""Plan or run the complete TA daily CI queue on the migrated local checkout.

Planning and module import use stdlib only. Execution requires a matching root
READY file and the previously audited owned_process.py/outcomes.py directory.
The pytest plugin registers real TorchNPU but never selects a device or creates
an NPU tensor; it only records whether original collection initialized runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def fact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


def _observation(stage: str, **details) -> None:
    folder = os.environ.get("TA_CI_PYTEST_OBSERVATIONS")
    if folder:
        write(
            Path(folder) / f"{os.getpid()}-{stage}.json", {"pid": os.getpid(), "worker": os.environ.get("PYTEST_XDIST_WORKER"), **details}
        )


def native_observation() -> dict:
    module = sys.modules.get("tilelang")
    maps = Path("/proc/self/maps").read_text()
    loaded = {}
    for line in maps.splitlines():
        if any(
            name in line
            for name in (
                "libtilelang.so",
                "libtvm_compiler.so",
                "libtvm_runtime.so",
                "tilelang_cython_wrapper",
                "libruntime.so",
                "libascendcl.so",
            )
        ):
            fields = line.split(maxsplit=5)
            path = Path(fields[5])
            loaded[str(path)] = {**fact(path), "mapped_inode": int(fields[4]), "current_inode": path.stat().st_ino}
    return {"tilelang_python": getattr(module, "__file__", None), "actual_native_libraries": loaded}


def pytest_load_initial_conftests(early_config, parser, args):
    # Real package registration, as in TA CI; no metadata-only stand-in.
    import torch
    import torch_npu
    import tilelang  # noqa: F401  # CPU-side source/native identity, no device action.

    _observation(
        "registration",
        torch=torch.__version__,
        torch_npu=torch_npu.__file__,
        initialized_after_real_registration=bool(torch.npu._initialized),
        **native_observation(),
    )


def pytest_collection_modifyitems(items):
    # Metadata only: preserve the exact full parametrized node identity in XML.
    for item in items:
        item.user_properties.append(("tilelang_ascend_ci_nodeid", item.nodeid))


def pytest_collection_finish(session):
    import torch

    _observation(
        "collection",
        nodes=len(session.items),
        selected_nodeids=[item.nodeid for item in session.items],
        initialized_by_original_collection=bool(torch.npu._initialized),
        **native_observation(),
    )


def pytest_deselected(items):
    _observation(f"deselected-{time.time_ns()}", deselected_nodeids=[item.nodeid for item in items])


def selected_health(text: str, physical_ids: list[int]) -> list[dict]:
    """Parse management rows, without creating a runtime/NPU context."""
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if "Process id" in line and "NPU" in line]
    assert len(headers) == 1, "missing or ambiguous management process table"
    board_lines = lines[: headers[0]]
    processes = []
    for line in lines:
        match = re.match(r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|", line)
        if match:
            processes.append({"card": int(match[1]), "chip": int(match[2]), "pid": int(match[3])})
    result = []
    for physical in physical_ids:
        matches = [i for i, line in enumerate(board_lines) if re.match(rf"^\|\s+\d+\s+{physical}\s+\|", line)]
        assert len(matches) == 1, f"cannot identify physical {physical}"
        index = matches[0]
        name, row = lines[index - 1].split("|"), lines[index].split("|")
        card, chip = int(name[1].split()[0]), int(row[1].split()[0])
        assert name[2].strip() == "OK", (physical, name)
        assert float(row[3].split()[0]) == 0, f"physical {physical} is busy after owned cleanup"
        assert any(f"No running processes found in NPU {card}" in line for line in lines) or any(p["card"] == card for p in processes)
        selected = [p for p in processes if p["card"] == card and p["chip"] == chip]
        assert not selected, (physical, "external or remaining processes", selected)
        result.append(
            {
                "physical": physical,
                "management_card": card,
                "management_chip": chip,
                "health": "OK",
                "AICore_percent": 0,
                "selected_processes": selected,
            }
        )
    return result


def xml_nodes(path: Path, output: Path) -> dict:
    """Retain per-node captured streams; a Pytest outcome is not an OS exit code."""
    root = ET.parse(path).getroot()
    rows = []
    for case in root.findall(".//testcase"):
        status = "passed"
        for key in ("failure", "error", "skipped"):
            if case.find(key) is not None:
                status = key
                break
        skipped = case.find("skipped")
        if skipped is not None and skipped.get("type") == "pytest.xfail":
            status = "xfail"
        classname = case.get("classname", "")
        scope = (
            "operator_registry"
            if path.parent.name == "operator"
            else ("refactor_additional" if "refactor" in classname.replace("/", ".").split(".") else "TA_adopted_python")
        )
        rows.append(
            {
                "nodeid": next(
                    (p.get("value") for p in case.findall("properties/property") if p.get("name") == "tilelang_ascend_ci_nodeid"), None
                ),
                "classname": case.get("classname"),
                "name": case.get("name"),
                "outcome": status,
                "scope": scope,
                "verdict_details": [
                    {"kind": node.tag, "attributes": node.attrib, "text": node.text}
                    for node in case
                    if node.tag in {"failure", "error", "skipped"}
                ],
                "seconds": case.get("time"),
                "stdout": "\n".join(node.text or "" for node in case.findall("system-out")),
                "stderr": "\n".join(node.text or "" for node in case.findall("system-err")),
                "node_OS_exit_code": None,
                "node_OS_exit_code_note": "Original pytest reports outcomes, not a per-node shell exit code; actual batch code is recorded separately.",
            }
        )
    output.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "nodes": len(rows),
        "scopes": {scope: sum(row["scope"] == scope for row in rows) for scope in sorted({row["scope"] for row in rows})},
        **{name: sum(row["outcome"] == name for row in rows) for name in ("passed", "failure", "error", "skipped", "xfail")},
    }


def _layout(root: Path):
    sys.path.insert(0, str(root / "scripts/ci"))
    try:
        return importlib.import_module("path_layout")
    finally:
        sys.path.pop(0)


def _capture(command: list[str], *, cwd: Path, env: dict) -> str:
    done = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True)
    if done.returncode:
        raise RuntimeError(f"plan command failed ({done.returncode}): {command}\n{done.stderr}")
    return done.stdout


def build_plan(root: Path, python: Path, output: Path, visible: str) -> dict:
    root, python, output = root.resolve(), python.absolute(), output.resolve()
    assert python.parent.name == "bin" and (python.parent.parent / "pyvenv.cfg").is_file(), "provide the dedicated modern virtualenv Python"
    env = dict(os.environ, PATH=str(python.parent) + os.pathsep + os.environ["PATH"], PYTHONPATH=str(root), TL_ROOT=str(root))
    queue = output / "bench_queue.json"
    _capture(["bash", str(root / "examples/bench_test.sh"), "--collect-only", "--collect-json", str(queue)], cwd=root, env=env)
    jobs = json.loads(queue.read_text())
    assert jobs["bench_tasks"] == 180, ("review unexpected full bench queue", jobs["bench_tasks"])
    resolver = root / "scripts/ci/resolve_operator_tests.py"
    text = _capture([str(python), str(resolver), "--repo-root", str(root), "list-tests", "--require-nonempty"], cwd=root, env=env)
    operators = text.splitlines()
    assert len(operators) == 27 and all(value.strip() for value in operators), (
        "expected 27 active operator files; never invoke pytest with no targets"
    )
    targets = _layout(root).pytest_targets(root)
    assert targets == ["testing/ascend_a2a3/python"]
    config = root / "ci/pytest-ascend.ini"
    plugin = ["-p", "maint.ascend_a2a3.ci_batch_driver"]
    common = [str(python), "-m", "pytest", "--forked", "-c", str(config)]
    tail = ["-v", "-n", "8", "-m", "not ci_skip", *plugin]
    commands = {
        "bench": ["bash", str(root / "examples/bench_test.sh"), "--skip-pytest"],
        "testing": [*common, *[str(root / value) for value in targets], *tail, "--junitxml=" + str(output / "testing/results.xml")],
        "operator": [*common, *[str(root / value) for value in operators], *tail, "--junitxml=" + str(output / "operator/results.xml")],
    }
    metadata_code = "import importlib.metadata as m,json,sys; print(json.dumps({'prefix':sys.prefix,'packages':{n:m.version(n) for n in ['pytest','pytest-xdist','pytest-forked','torch','torch-npu','numpy']}}))"
    metadata = json.loads(_capture([str(python), "-I", "-B", "-c", metadata_code], cwd=root, env=env))
    assert Path(metadata["prefix"]).resolve() == python.parent.parent.resolve()
    assert metadata["packages"]["torch"] == "2.10.0+cpu" and metadata["packages"]["torch-npu"] == "2.10.0"
    head = _capture(
        ["git", "-c", "protocol.allow=never", "rev-parse", "HEAD"], cwd=root, env=dict(env, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0")
    ).strip()
    wrappers = list((root / "build/lib").glob("tilelang_cython_wrapper*.so"))
    assert len(wrappers) == 1, "resolve the actual root wrapper before granting CI"
    sdk = Path(os.environ["ASCEND_HOME_PATH"]).resolve()
    compiler = Path(shutil.which("bisheng", path=env["PATH"]))
    frozen = [
        fact(root / "build/lib/libtilelang.so"),
        fact(wrappers[0]),
        fact(compiler),
        fact(sdk / "lib64/libruntime.so"),
        fact(sdk / "lib64/libascendcl.so"),
    ]
    plan = {
        "root": str(root),
        "head": head,
        "python": str(python),
        "output": str(output),
        "metadata": metadata,
        "physical_visible_ids": visible,
        "bench_tasks": 180,
        "bench_counts": jobs["counts"],
        "operator_files": operators,
        "pytest_target": targets,
        "testing_pythonpath": str(root / targets[0]) + os.pathsep + str(root),
        "pytest_node_count": "discovered by real pytest; new refactor tests are additional to original TA CI",
        "daily_marker": "not ci_skip",
        "low_priority_included": True,
        "commands": commands,
        "stage_cwd": {"bench": str(root / "examples"), "testing": str(root / "examples/ascend_a2a3"), "operator": str(root / "examples")},
        "frozen_native": frozen,
        "original_CI_controls": {
            "bench_MAX_JOBS": 8,
            "pytest_xdist": 8,
            "pytest_forked": True,
            "TILELANG_AUTO_TUNING_CPU_COUNTS": 4,
            "TILELANG_AUTO_TUNING_MAX_CPU_COUNT": 4,
            "autotuner_values_apply_to": "bench and testing only",
        },
        "operator_parent_environment": {
            name: os.environ.get(name) for name in ["TILELANG_AUTO_TUNING_CPU_COUNTS", "TILELANG_AUTO_TUNING_MAX_CPU_COUNT"]
        },
        "host_limits": {
            name: os.environ.get(name) for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]
        },
        "host_limits_note": "Recorded current host limits, not claimed to be original TA CI settings; this driver adds no CPU thread cap.",
        "environment": {
            "PYTHONPATH": str(root),
            "TL_ROOT": str(root),
            "TL_PLATFORM": "A3",
            "TILELANG_TARGET": "ascendc",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "1",
            "A2A3_TORCH_INTEROP_VENV": metadata["prefix"],
            "ASCEND_HOME_PATH": str(sdk),
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        },
        "failure_policy": "Ordinary task/pytest nonzero does not stop later tasks or stages. Stop only on timeout, health/exclusivity failure, cancellation or provenance failure.",
        "overall_timeout_seconds": 21600,
        "individual_node_timeout_seconds": None,
        "execution_equivalence": "The public bench entry can still run --pytest-markers normally. This driver splits bench --skip-pytest, identical full testing, then original operator registry into three commands solely to retain independent exits; testing inherits the bench's two autotuner CPU=4 exports; operator inherits the frozen original parent environment, because the original bench child cannot export back to its workflow parent.",
        "original_CI_reference": "approximately 57min plus additional migrated refactor scope",
        "task_execution_started": False,
        "stage3_accepted": False,
    }
    write(output / "plan.json", plan)
    return plan


def verify_frozen(plan: dict) -> None:
    root = Path(plan["root"])
    head = subprocess.check_output(["git", "-c", "protocol.allow=never", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert head == plan["head"], "checkout changed after CI scope review"
    for row in plan["frozen_native"]:
        assert fact(Path(row["path"])) == row, "native/wrapper changed after review"


def execute(plan: dict, grant_path: Path, owned_dir: Path) -> int:
    grant = json.loads(grant_path.read_text())
    assert grant["authorization"] == "ROOT_READY_FULL_ASCEND_CI"
    assert grant["plan_sha256"] == fact(Path(plan["output"]) / "plan.json")["sha256"]
    assert grant["physical_visible_ids"] == plan["physical_visible_ids"]
    output, root = Path(plan["output"]), Path(plan["root"])
    sys.path.insert(0, str(owned_dir.resolve()))
    owned = importlib.import_module("owned_process")
    env = dict(os.environ, **plan["environment"], PATH=str(Path(plan["python"]).parent) + os.pathsep + os.environ["PATH"])
    env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    env.pop("TVM_IMPORT_PYTHON_PATH", None)
    env.pop("TL_TEMPLATE_PATH", None)
    env["ASCEND_RT_VISIBLE_DEVICES"] = plan["physical_visible_ids"]
    physical = [int(value) for value in plan["physical_visible_ids"].split(",")]
    assert physical and len(set(physical)) == len(physical) and min(physical) >= 0
    record = {"stage3_accepted": False, "status": "started", "stages": [], "grant": fact(grant_path)}

    def persist():
        write(output / "result.json", record)

    def run(command, folder, timeout, child_env, cwd=None):
        folder.mkdir(parents=True, exist_ok=False)
        with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, cwd=cwd or root, env=child_env, stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                return process.wait(timeout=timeout)
            finally:
                owned.cleanup(process, folder / "cleanup.json")

    def health(label):
        folder = output / ("health_" + label)
        code = run(["/usr/local/bin/npu-smi", "info"], folder, 60, env)
        assert code == 0
        write(folder / "selected.json", selected_health((folder / "stdout.log").read_text(), physical))

    persist()
    primary = None
    deadline = time.monotonic() + plan["overall_timeout_seconds"]
    with owned.subreaper(), owned.controller_signals():
        try:
            verify_frozen(plan)
            health("before")
            for name in ("bench", "testing", "operator"):
                verify_frozen(plan)
                folder = output / name
                child_env = dict(
                    env,
                    TA_CI_ARTIFACT_DIR=str(folder),
                    TILELANG_CACHE_DIR=str(folder / "cache"),
                    TORCH_EXTENSIONS_DIR=str(folder / "torch"),
                    TMPDIR=str(folder / "tmp"),
                    TA_CI_PYTEST_OBSERVATIONS=str(folder / "observations"),
                )
                if name == "testing":
                    # Original testing runs inside bench's helper-path export;
                    # the separate workflow operator step does not inherit it.
                    child_env["PYTHONPATH"] = plan["testing_pythonpath"]
                for key, inherited in plan["operator_parent_environment"].items():
                    if name in ("bench", "testing"):
                        child_env[key] = "4"
                    elif inherited is None:
                        child_env.pop(key, None)
                    else:
                        child_env[key] = inherited
                # Parent owns these roots; run() creates the stage directory.
                scratch = output / (name + "_scratch")
                scratch.mkdir()
                child_env["TMPDIR"] = str(scratch)
                row = {"stage": name, "argv": plan["commands"][name], "cwd": plan["stage_cwd"][name], "started_ns": time.time_ns()}
                record["stages"].append(row)
                persist()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("original full CI overall 360-minute budget exhausted")
                child_env["PWD"] = plan["stage_cwd"][name]
                code = run(plan["commands"][name], folder, remaining, child_env, cwd=Path(plan["stage_cwd"][name]))
                row.update(exit_code=code, finished_ns=time.time_ns())
                if name != "bench":
                    try:
                        row["pytest"] = xml_nodes(folder / "results.xml", folder / "nodes.jsonl")
                    except (OSError, ET.ParseError) as error:
                        row["XML_observation_error"] = repr(error)
                persist()
                health("after_" + name)
                # A normal nonzero is evidence, never a reason to prune CI.
            record["status"] = "all_three_stages_finished"
            persist()
        except BaseException as error:
            primary = error
            record["status"] = "stopped_infrastructure_or_device_" + repr(error)
            persist()
            raise
        finally:
            try:
                with owned.shield_cleanup_signals():
                    health("final")
            except BaseException as error:
                record["post_health_failure"] = repr(error)
                if primary is None:
                    record["status"] = "stopped_post_health_failure"
                    persist()
                    raise
                persist()
    return next((row["exit_code"] for row in record["stages"] if row["exit_code"]), 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--grant-file", type=Path)
    parser.add_argument("--owned-process-dir", type=Path)
    args = parser.parse_args(argv)
    if args.execute:
        if args.grant_file is None or args.owned_process_dir is None:
            parser.error("execution requires --grant-file and --owned-process-dir from the reviewed root window")
        plan = json.loads((args.output / "plan.json").read_text())
        assert plan["root"] == str(args.repo_root.resolve()) and plan["python"] == str(args.python.absolute())
        assert plan["physical_visible_ids"] == args.visible_devices
        return execute(plan, args.grant_file, args.owned_process_dir)
    args.output.mkdir(parents=True, exist_ok=False)
    plan = build_plan(args.repo_root, args.python, args.output, args.visible_devices)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
