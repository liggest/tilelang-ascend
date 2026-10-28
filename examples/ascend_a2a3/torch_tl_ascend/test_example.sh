#!/usr/bin/env bash

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Run the original fixed Torch/source tests after versioned package installation."
    echo "Use an explicit task-owned A2A3_TORCH_INTEROP_VENV and TILELANG_CACHE_DIR."
    exit 0
fi

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
python - <<'PYGUARD' || exit $?
import os
import sys
from pathlib import Path

owned = os.environ.get("A2A3_TORCH_INTEROP_VENV")
if not owned or sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != Path(owned).resolve():
    raise SystemExit("Set A2A3_TORCH_INTEROP_VENV to the active dedicated virtualenv; refusing package changes")
requested = os.environ.get("TILELANG_CACHE_DIR")
if not requested:
    raise SystemExit("Set a dedicated TILELANG_CACHE_DIR before compiling/installing this example")
import tilelang
if Path(tilelang.env.TILELANG_CACHE_DIR).resolve() != Path(requested).resolve():
    raise SystemExit("TILELANG_CACHE_DIR conflicts with the active cache-directory override; refusing package changes")
PYGUARD

ensure_torch_tl_ascend() {
    local output
    if ! output=$(python -m pip show torch_tl_ascend 2>&1); then
        echo "torch_tl_ascend is not installed. Installing with python setup.py install ..."
        python setup.py install || { echo "Failed to install package"; exit 1; }
        echo "Installed torch_tl_ascend"
        return 0
    fi

    local expected_version=$(tr -d ' \t\r\n' < src/torch_tl_ascend/VERSION)  # read local VERSION file and remove whitespace/newlines
    local installed_version=$(echo "$output" | awk -F': ' '/^Version:/{print $2}' | tr -d ' \t\r\n')  # find installed version

    if [ "$installed_version" != "$expected_version" ]; then
        echo "torch_tl_ascend v$installed_version is installed. Expected v$expected_version. Reinstalling ..."
        echo "Cleaning previous build artifacts ..."
        rm -rf build/ dist/
        rm -rf src/*.egg-info/
        python -m pip uninstall -y torch_tl_ascend || { echo "Failed to uninstall outdated package"; exit 1; }
        python setup.py install || { echo "Failed to install package"; exit 1; }
        echo "Installed torch_tl_ascend v$expected_version"
        return 0
    fi

    echo "torch_tl_ascend v$installed_version is already installed"
    return 0
}

ensure_torch_tl_ascend

echo "Running test_torch.py"
python test_torch.py || { echo "python test_torch.py failed"; exit 1; }
echo "Running test_source.py"
python test_source.py || { echo "python test_source.py failed"; exit 1; }

echo "All Test Passed!"
