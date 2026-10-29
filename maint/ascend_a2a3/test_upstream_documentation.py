"""Versioned documentation/profiling assets; synthetic CPU CSV checks only."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / ".agents/skills/tilelang-op-profiling"
PIN = "463c234ce3087205c092e931207ca87b4cd5494c"
BASE = "0deb8b93da11b7f0becbcded732d9633b7ca4e34"
GUIDE = "docs/TileLang-Ascend Programming Guide.md"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load(Path(__file__).with_name("audit_documentation.py"), "upstream_docs_audit")


def source_script():
    return subprocess.check_output(
        [
            "git",
            "-c",
            "protocol.allow=never",
            "-C",
            os.environ["A2A3_TA_SOURCE_REPO"],
            "show",
            PIN + ":.agents/skills/tilelang-op-profiling/scripts/perf_summary.py",
        ],
        env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0"),
    )


class VersionedAssets(unittest.TestCase):
    def test_current_and_original_guide_sources_and_main_skills(self):
        source = Path(os.environ["A2A3_TA_SOURCE_REPO"]) if os.environ.get("A2A3_TA_SOURCE_REPO") else None
        result = audit.audit(ROOT, source)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["source_counts_by_pin"], {BASE: 224, PIN: 7})
        self.assertEqual(result["historical_source_versions_checked"], 1)
        self.assertEqual(result["skills"], 22)
        self.assertEqual(result["protected_main_skills"], 22)

    def test_new_guide_pin_cannot_be_changed_back_to_old_global_pin(self):
        data = json.loads((ROOT / "maint/ascend_a2a3/docs_manifest.json").read_text())
        next(entry for entry in data["entries"] if entry["source_path"] == GUIDE)["source_pin"] = BASE
        with patch.object(audit.json, "loads", return_value=data):
            result = audit.audit(ROOT)
        self.assertTrue(any("Wrong per-entry source pin" in error for error in result["errors"]))

    def test_old_guide_history_cannot_be_dropped(self):
        data = json.loads((ROOT / "maint/ascend_a2a3/docs_manifest.json").read_text())
        next(entry for entry in data["entries"] if entry["source_path"] == GUIDE)["source_history"] = []
        with patch.object(audit.json, "loads", return_value=data):
            result = audit.audit(ROOT)
        self.assertTrue(any("retain the original 0deb guide archive" in error for error in result["errors"]))


@unittest.skipUnless(os.environ.get("A2A3_TA_SOURCE_REPO"), "optional raw source/CPU CSV pair needs A2A3_TA_SOURCE_REPO")
class ProfilingParser(unittest.TestCase):
    def setUp(self):
        self.raw = source_script()
        self.original = types.ModuleType("original_TA_profiling_parser")
        exec(compile(self.raw, "fixed_TA_perf_summary.py", "exec"), self.original.__dict__)
        self.current = load(SKILL / "scripts/perf_summary.py", "migrated_A2A3_profiling_parser")

    @staticmethod
    def fixture(directory):
        directory.mkdir()
        tables = {
            "OpBasicInfo.csv": [
                ["Op Name", "Op Type", "Task Duration(us)", "Block Dim"],
                ["main_kernel", "Vector", 10.5, 24],
                ["main_kernel", "Vector", 12.5, 24],
            ],
            "PipeUtilization.csv": [["Op Name", "aiv_time(us)", "aiv_vec_ratio", "aiv_mte2_ratio"], ["main_kernel", 10.0, 0.5, 0.25]],
            "Memory.csv": [["Op Name", "aiv_gm_read_bytes", "aiv_gm_write_bytes"], ["main_kernel", 1024, 512]],
        }
        for name, rows in tables.items():
            with (directory / name).open("w", newline="") as stream:
                csv.writer(stream).writerows(rows)

    def test_complete_parser_AST_is_original(self):
        self.assertEqual(
            ast.dump(ast.parse(self.raw), include_attributes=False),
            ast.dump(ast.parse((SKILL / "scripts/perf_summary.py").read_bytes()), include_attributes=False),
        )

    def test_same_synthetic_CSV_has_exact_same_tables_summary_and_refusals(self):
        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "synthetic_input"
            self.fixture(data)
            old_tables, old_name = self.original.load_kernel_tables(str(data), "main_kernel")
            new_tables, new_name = self.current.load_kernel_tables(str(data), "main_kernel")
            self.assertEqual((old_tables, old_name), (new_tables, new_name))
            self.assertEqual(
                self.original.generate_summary(old_tables, "same_archive", "main_kernel", old_name),
                self.current.generate_summary(new_tables, "same_archive", "main_kernel", new_name),
            )
            for module in [self.original, self.current]:
                with self.assertRaises(ValueError):
                    module.unique_match("main_kernel", ["main_kernel_a", "main_kernel_b"], "synthetic_fixture")
                with self.assertRaises(ValueError):
                    module.load_kernel_tables(str(data), "missing_kernel")
                second = Path(td) / module.__name__
                second.mkdir()
                self.fixture(second / "one")
                self.fixture(second / "two")
                with self.assertRaises(ValueError):
                    module.resolve_profile_dir(str(second))

    def test_both_CLI_archive_same_synthetic_CSV_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "synthetic_input"
            self.fixture(data)
            source = root / "original/perf_summary.py"
            source.parent.mkdir()
            source.write_bytes(self.raw)
            outputs = []
            for name, script in [("old", source), ("new", SKILL / "scripts/perf_summary.py")]:
                target = root / name
                command = [
                    sys.executable,
                    "-B",
                    str(script),
                    str(data),
                    "--output-dir",
                    str(target),
                    "--kernel-name",
                    "main_kernel",
                    "--command",
                    "synthetic CSV parser test only; no profiler executed",
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads((target / "metadata.json").read_text())
                metadata.pop("created_at_utc")
                outputs.append((metadata, (target / "summary.txt").read_text().replace(str(target), "COMMON_ARCHIVE")))
                before = {p.name: p.read_bytes() for p in target.iterdir()}
                refused = subprocess.run(command, capture_output=True, text=True, check=False)
                self.assertNotEqual(refused.returncode, 0)
                self.assertEqual(before, {p.name: p.read_bytes() for p in target.iterdir()})
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
