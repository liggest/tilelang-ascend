# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Candidate safety and source-preservation checks; no TileLang/NPU imports."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("a2a3_ecosystem", Path(__file__).with_name("ecosystem.py"))
eco = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eco)


class ImportRulesTest(unittest.TestCase):
    def test_namespace_has_no_fallback_and_rejects_unsafe_paths(self):
        self.assertEqual(eco.destination_for("examples/gemm/x.py"), "examples/ascend_a2a3/gemm/x.py")
        self.assertEqual(eco.destination_for("examples_experiment/x.py"), "examples_experiment/x.py")
        self.assertEqual(eco.destination_for("testing/python/x.py"), "testing/ascend_a2a3/python/x.py")
        for path in (
            "/examples/x.py",
            "examples/../x.py",
            "examples//x.py",
            "examples/./x.py",
            "examples/ascend_a2a3/x.py",
            "examples\\x.py",
            "docs/x.py",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                eco.destination_for(path)

    def test_aliases_and_layout_only_change_import_spans(self):
        source = """# keep tilelang.language in comments
import tilelang.language as DSL
from tilelang.language.pto import tile as pto_tile
from tilelang.intrinsics import make_zn_layout as zn, make_nz_layout
@jit(pass_configs={"x": True}, compile_flags=["-O3"])
def kernel(real_shape=[2, 4]):
    DSL._src_code("tilelang.language; { raw };")
    return DSL.alloc_ub(real_shape)
"""
        result, audit, issues = eco.rewrite_imports(source)
        self.assertEqual(len(audit), 3)
        self.assertEqual(issues, [])
        self.assertIn("import tilelang.ascend.a2a3.language as DSL", result)
        self.assertIn("from tilelang.ascend.a2a3.layout import make_zn_layout as zn, make_nz_layout", result)
        self.assertEqual(result[result.index("@jit") :], source[source.index("@jit") :])
        self.assertEqual(result.splitlines()[0], source.splitlines()[0])

    def test_mixed_import_keeps_generic_names_and_comments(self):
        text = "from tilelang import language as T, DataType  # keep\n"
        result, _, issues = eco.rewrite_imports(text)
        self.assertFalse(issues)
        self.assertEqual(result, "from tilelang.ascend.a2a3 import language as T; from tilelang import DataType  # keep\n")
        text = "from tilelang import (\n    language as T, # important\n    DataType,\n)\n"
        result, audit, issues = eco.rewrite_imports(text)
        self.assertEqual(result, text)
        self.assertFalse(audit)
        self.assertTrue(issues)

    def test_utf8_columns_and_nested_imports(self):
        text = 'title = "中文"; import tilelang.language as DSL\nif ready:\n    from tilelang import language as T, DataType\n'
        result, _, _ = eco.rewrite_imports(text)
        ast.parse(result)
        self.assertTrue(result.startswith('title = "中文"; import tilelang.ascend.a2a3.language as DSL'))
        self.assertIn("    from tilelang.ascend.a2a3 import language as T; from tilelang import DataType", result)

    def test_bare_unknown_old_ir_runtime_are_not_silently_rewritten(self):
        source = """import tilelang.language
from tilelang.language.frame import KernelFrame
from tilelang.intrinsics import make_zn_layout, unknown_cuda_op
from tilelang.utils.target import determine_platform
from tvm import tir
from tilelang.engine.phase import LowerAndLegalize
x = tir.AllocateConst
"""
        result, audit, issues = eco.rewrite_imports(source)
        self.assertEqual(result, source)
        self.assertFalse(audit)
        self.assertGreaterEqual(len(issues), 7)

    def test_nonstandard_import_is_reported_instead_of_claimed_changed(self):
        source = "from " + chr(92) + "\n tilelang.language import Kernel\n"
        result, audit, issues = eco.rewrite_imports(source)
        self.assertEqual(result, source)
        self.assertFalse(audit)
        self.assertTrue(any("nonstandard" in issue["reason"] for issue in issues))

    def test_guard_precedes_effects_and_preserves_legal_futures(self):
        for source in (
            '"文档"; touches.append(1)\n',
            "from __future__ import annotations; touches.append(1)\n",
            '"doc"\nfrom __future__ import (annotations,)\ntouches.append(1)\n',
            "# header\ntouches.append(1)\n",
        ):
            touches = []
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "requires manual review"):
                exec(compile(eco.guarded(source), "candidate", "exec"), {"touches": touches})
            self.assertEqual(touches, [])


class CandidateIOTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.ta = self.base / "ta"
        self.ta.mkdir()
        self.path = "examples/unit/example.py"
        p = self.ta / self.path
        p.parent.mkdir(parents=True)
        p.write_text("from tilelang import language as T\nraise AssertionError('source must never run')\n")

        def run(*args):
            return subprocess.check_output(["git", "-C", str(self.ta), *args])

        run("init", "-q")
        run("add", self.path)
        run("-c", "user.name=unit", "-c", "user.email=unit@example.invalid", "commit", "-qm", "fixture")
        pin = run("rev-parse", "HEAD").decode().strip()
        self.entry = {
            "path": self.path,
            "kind": "blob",
            "git_blob": run("rev-parse", f"HEAD:{self.path}").decode().strip(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "classification": "A2A3",
            "classification_basis": "fixture",
        }
        self.addCleanup(patch.stopall)
        patch.object(eco, "BASELINE", pin).start()
        patch.object(eco, "load_manifest", return_value={self.path: self.entry}).start()
        self.out = self.base / "candidates"

    def test_provenance_guard_mapping_and_repeated_output_refused(self):
        record = eco.prepare(self.ta, self.out, [self.path])["entries"][0]
        self.assertEqual(record["destination_path"], "examples/ascend_a2a3/unit/example.py")
        self.assertFalse(record["installed"])
        self.assertEqual(record["state"], "blocked_candidate")
        candidate = self.out / record["candidate_path"]
        with self.assertRaisesRegex(RuntimeError, "requires manual review"):
            exec(compile(candidate.read_text(), str(candidate), "exec"), {})
        original = candidate.read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            eco.prepare(self.ta, self.out, [self.path])
        self.assertEqual(candidate.read_bytes(), original)
        self.assertTrue(candidate.with_name(candidate.name + ".json").is_file())

    def test_drift_hash_and_wrong_pin_reject_before_output(self):
        (self.ta / self.path).write_text("changed = True\n")
        with self.assertRaisesRegex(ValueError, "drift"):
            eco.prepare(self.ta, self.out, [self.path])
        self.assertFalse(self.out.exists())
        self.entry["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            eco.prepare(self.ta, self.out, [self.path])
        with patch.object(eco, "BASELINE", "0" * 40), self.assertRaisesRegex(ValueError, "HEAD"):
            eco.prepare(self.ta, self.out, [self.path])

    def test_missing_selection_duplicate_and_whole_batch_preflight(self):
        for paths in ([], [self.path, self.path], [self.path, "examples/absent.py"]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                eco.prepare(self.ta, self.out, paths)
            self.assertFalse(self.out.exists())

    def test_unknown_classification_needs_specific_reason_and_a5_always_refused(self):
        self.entry["classification"] = "uncertain"
        for decisions in (None, {self.path: {"decision": "a2a3", "reason": " "}}):
            with self.assertRaisesRegex(ValueError, "decision/reason"):
                eco.prepare(self.ta, self.out, [self.path], decisions)
        decision = {self.path: {"decision": "a2a3", "reason": "fixture uses only source A2A3 launch"}}
        eco.prepare(self.ta, self.out, [self.path], decision)
        self.entry["classification"] = "A5"
        with self.assertRaisesRegex(ValueError, "A5"):
            eco.prepare(self.ta, self.base / "a5", [self.path], decision)

    def test_reserved_autotune_and_malformed_decisions_are_refused(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            eco.prepare(self.ta, self.out, [self.path], [])
        path = "examples/autotune/example_gemm_autotune.py"
        with patch.object(eco, "load_manifest", return_value={path: self.entry}), self.assertRaisesRegex(ValueError, "autotuner owner"):
            eco.prepare(self.ta, self.out, [path])
        self.assertFalse(self.out.exists())

    def test_no_output_in_source_or_through_symlink(self):
        with self.assertRaisesRegex(ValueError, "read-only TA"):
            eco.prepare(self.ta, self.ta / "bad", [self.path])
        link = self.base / "linked"
        link.symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            eco.prepare(self.ta, link / "bad", [self.path])
        (self.ta / self.path).unlink()
        (self.ta / self.path).symlink_to(self.base / "not-source")
        with self.assertRaisesRegex(ValueError, "symlink"):
            eco.prepare(self.ta, self.out, [self.path])


class FixedManifestTest(unittest.TestCase):
    def test_full_manifest_pin_and_hash_shapes(self):
        entries = eco.load_manifest()
        self.assertEqual(len(entries), 632)
        self.assertEqual(sum(p.endswith(".py") for p in entries), 340)
        self.assertEqual(sum(e["classification"] == "A5" for e in entries.values()), 5)
        for entry in entries.values():
            self.assertRegex(entry["sha256"], "^[0-9a-f]{64}$")
            self.assertRegex(entry["git_blob"], "^[0-9a-f]{40}$")
        self.assertEqual(sum(e["classification"] == "uncertain" for e in entries.values()), 196)


if __name__ == "__main__":
    unittest.main()
