"""Verify the actual trap subprocess imports register NPU without launching it."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_trap_script_registers_torch_npu_without_autoload():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("the original trap subprocess requires torch_npu")
    path = Path(__file__).parents[1] / "language" / "test_tilelang_ascend_language_copy_ub_to_gm_alignment.py"
    module = ast.parse(path.read_text())
    assignment = next(
        node
        for node in module.body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_TRAP_SCRIPT" for target in node.targets)
    )
    script = ast.parse(ast.literal_eval(assignment.value))
    prefix = []
    for statement in script.body:
        if not isinstance(statement, (ast.Import, ast.ImportFrom)):
            break
        prefix.append(ast.unparse(statement))
    # Run only the actual imports: do not load the kernel library, allocate an
    # NPU tensor or execute the intentional hardware trap in this host test.
    probe = "\n".join(prefix) + "\nassert 'torch_npu' in sys.modules\nassert callable(torch.Tensor.npu)\nprint('NPU_REGISTERED')\n"
    result = subprocess.run(
        [sys.executable, "-c", probe], env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NPU_REGISTERED" in result.stdout
