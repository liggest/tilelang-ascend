#!/bin/bash
# Read-only modern A2/A3 checkout/metadata/tool discovery. Never install or fix env.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$PROJECT_ROOT"
python - <<'PY_CHECK'
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
failures = []
print("checkout:", root)
for name in ("torch", "torch-npu", "apache-tvm-ffi"):
    try:
        package = metadata.distribution(name)
        print(name, package.version, package.locate_file(""))
        if name == "torch-npu":
            print("declared requirements:", json.dumps(package.requires or []))
    except metadata.PackageNotFoundError:
        failures.append("missing package metadata: " + name)
for name in ("tvm", "catlass", "shmem", "pto-isa"):
    result = subprocess.run(["git", "submodule", "status", "3rdparty/" + name],
                            capture_output=True, text=True, check=False)
    print("pin", name, result.stdout.strip() or result.stderr.strip())
    if result.returncode or not result.stdout or result.stdout[0] in "-+U":
        failures.append("check recorded submodule pin: " + name)
print("dev library candidates:", [str(p) for folder in (root / "build/lib", root / "build/tvm")
                                  if folder.is_dir() for p in folder.glob("*.so")])
print("CANN:", os.environ.get("ASCEND_HOME_PATH", "unset"))
print("bisheng:", shutil.which("bisheng") or "not on PATH; check configured CANN compiler path")
print("File discovery does not prove native/Cython import or device readiness.")
if failures:
    print("FAILED checks:", json.dumps(failures))
    raise SystemExit(1)
print("Metadata and recorded source pins checked; native import/NPU checks remain separate.")
PY_CHECK
