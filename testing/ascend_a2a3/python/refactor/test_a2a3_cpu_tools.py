"""Portable CLI tests for the original TA preprocessor and performance plot."""

import argparse
import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
TOOLS = ROOT / "examples/ascend_a2a3/flash_attention/fa_opt"
PREPROCESS = TOOLS / "expert_v3/tools/preprocess.py"
PLOT = TOOLS / "plot.py"
SUGAR = """# preserve plain source
LOCK Q 1 MTE2 MTE1 2
LOCK X 5 V MTE3
INIT_LOCKS Q, X
ACQ MTE1 Q(i)
REL MTE1 Q(i)
LOCKED MTE3 X: result = data
DESTROY_LOCKS Q, X
"""
EXPANDED = """# preserve plain source
T.set_flag("MTE1", "MTE2", 1)
T.set_flag("MTE1", "MTE2", 2)
T.set_flag("MTE3", "V", 5)
T.wait_flag("MTE2", "MTE1", 1 + i)
T.set_flag("MTE1", "MTE2", 1 + i)
T.wait_flag("V", "MTE3", 5)
result = data
T.set_flag("MTE3", "V", 5)
T.wait_flag("MTE1", "MTE2", 1)
T.wait_flag("MTE1", "MTE2", 2)
T.wait_flag("MTE3", "V", 5)
"""
CSV = "B,S,Q_H,KV_H,D,Mode,AC_Time(us),TL_Time(us),Ratio(AC/TL %),Result_Status\n1,128,12,1,128,test,12.5,10,125,fixture\n2,512,12,1,128,test,0,20,0,fixture\n"
HAS_PLOT_DEPS = all(importlib.util.find_spec(name) for name in ("pandas", "matplotlib"))


class CpuToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        self.env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(self.cwd / "mpl"), PYTHONDONTWRITEBYTECODE="1")

    def cli(self, tool, *args):
        return subprocess.run(
            [sys.executable, str(tool), *map(str, args)],
            cwd=self.cwd,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=30,
        )

    def test_imports_do_not_parse_cli_or_load_plot_or_accelerator_dependencies(self):
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name.split(".")[0] in {"pandas", "matplotlib", "torch", "torch_npu", "tilelang"}:
                raise AssertionError(f"unexpected import: {name}")
            return original_import(name, *args, **kwargs)

        with (
            patch.object(builtins, "__import__", guarded),
            patch.object(argparse.ArgumentParser, "parse_args", side_effect=AssertionError("parsed CLI on import")),
        ):
            for index, path in enumerate((PREPROCESS, PLOT)):
                spec = importlib.util.spec_from_file_location(f"tool_import_{index}", path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        self.assertEqual(list(self.cwd.iterdir()), [])

    def test_preprocessor_all_sugar_forms_stdout_and_output_file(self):
        source = self.cwd / "source.tl"
        source.write_text(SUGAR)
        stdout = self.cli(PREPROCESS, source)
        self.assertEqual(stdout.returncode, 0, stdout.stderr)
        self.assertEqual(stdout.stdout, EXPANDED)
        output = self.cwd / "expanded.py"
        result = self.cli(PREPROCESS, source, "-o", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(output.read_bytes(), EXPANDED.encode())

    def test_preprocessor_preserves_plain_macro_and_empty_input(self):
        source = self.cwd / "source.tl"
        plain = "@T.macro\ndef f(a):\n    T.copy(a, b)\n"
        source.write_text(plain)
        result = self.cli(PREPROCESS, source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, plain)
        source.write_text("")
        result = self.cli(PREPROCESS, source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "\n")

    def test_preprocessor_unknown_lock_fails_before_overwriting_output(self):
        source = self.cwd / "invalid.tl"
        source.write_text("ACQ V missing\n")
        output = self.cwd / "existing.py"
        output.write_text("keep existing")
        result = self.cli(PREPROCESS, source, "-o", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("KeyError", result.stderr)
        self.assertEqual(output.read_text(), "keep existing")

    def test_cli_help_does_not_run_tools(self):
        for tool in (PREPROCESS, PLOT):
            result = self.cli(tool, "--help")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("usage:", result.stdout)
        self.assertEqual(list(self.cwd.iterdir()), [])

    @unittest.skipUnless(HAS_PLOT_DEPS, "CPU plotting requires existing pandas and matplotlib; no installation is performed")
    def test_plot_real_csv_png_with_explicit_and_original_literal_default_paths(self):
        source = self.cwd / "performance.csv"
        source.write_text(CSV)
        output = self.cwd / "explicit.png"
        result = self.cli(PLOT, source, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        literal = self.cwd / ".log/run_cmp_performance_${RUN_ID}/comparison_report_${RUN_ID}.csv"
        literal.parent.mkdir(parents=True)
        literal.write_text(CSV)
        result = self.cli(PLOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        default = self.cwd / "time_tl-ascend_vs_ascendc.png"
        self.assertEqual(default.read_bytes(), output.read_bytes())

    @unittest.skipUnless(HAS_PLOT_DEPS, "CPU plotting requires existing pandas and matplotlib; no installation is performed")
    def test_plot_missing_source_columns_fails_without_fake_success(self):
        source = self.cwd / "invalid.csv"
        source.write_text("S,TL_Time(us)\n128,10\n")
        output = self.cwd / "invalid.png"
        result = self.cli(PLOT, source, "-o", output)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("AC_Time(us)", result.stderr)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
