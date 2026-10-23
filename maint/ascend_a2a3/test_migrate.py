# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Hardware-free behavioral checks for migration preparation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("ascend_a2a3_migrate", Path(__file__).with_name("migrate.py"))
migrate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migrate)


class MigrationTests(unittest.TestCase):
    def test_cpp_preserves_comments_user_strings_and_renames_qualified_types(self):
        original = """// tir::Block and "tl.transform.HostLegalize" in a comment
#include <tvm/tir/stmt.h>
namespace tvm::tir { BlockRealizeNode* node; }
const char* diagnostic = "Block tir::LetStmt";
const char* ffi_name = "tl.transform.HostLegalize";
"""
        candidate, issues = migrate.transform_cpp(original, {"tl.transform.HostLegalize": "tl.transform.A2A3HostLegalize"})
        self.assertIn('// tir::Block and "tl.transform.HostLegalize" in a comment', candidate)
        self.assertIn('"Block tir::LetStmt"', candidate)
        self.assertIn("namespace tvm::tirx { SBlockRealizeNode* node; }", candidate)
        self.assertIn("<tvm/tirx/stmt.h>", candidate)
        self.assertIn('"tl.transform.A2A3HostLegalize"', candidate)
        self.assertEqual(issues, [])

    def test_structural_rewrite_requires_review(self):
        candidate, issues = migrate.transform_cpp("Stmt x = LetStmt(v, n, body);", {})
        self.assertIn("Bind(v, n, body)", candidate)
        self.assertTrue(any(issue["node"] == "LetStmt" for issue in issues))

    def test_python_unsafe_candidate_stops_before_body_even_with_future_import(self):
        original = """from __future__ import annotations
# original body must not execute
print("UNSAFE BODY EXECUTED")
x = tir.LetStmt(var, value, body)
"""
        candidate, issues = migrate.transform_python(original, {})
        self.assertTrue(issues)
        guarded = migrate.guard_python(candidate)
        process = subprocess.run([sys.executable, "-c", guarded], text=True, capture_output=True)
        self.assertNotEqual(process.returncode, 0)
        self.assertNotIn("UNSAFE BODY EXECUTED", process.stdout)
        self.assertIn("requires manual structural review", process.stderr)
        self.assertNotIn("SyntaxError", process.stderr)

    def test_python_tvm_alias_and_direct_import_are_explicitly_blocked(self):
        for original in ("from tvm import tir as ir\nx = ir.Block\n", "from tvm.tir import Block\nx = Block()\n"):
            with self.subTest(original=original):
                _, issues = migrate.transform_python(original, {})
                self.assertTrue(issues)
                self.assertTrue(any("import" in issue["reason"] for issue in issues))

    def test_realistic_language_frame_and_op_fragments_are_accounted_for(self):
        original = """from tvm.script.ir_builder.tir.frame import TIRFrame, BlockFrame
class Scope(BlockFrame):
    pass
value = T.call_intrin("handle", "tir.tvm_access_ptr")
T.block_attr({"scope": "C"})
"""
        candidate, issues = migrate.transform_python(original, {})
        self.assertIn("class Scope(SBlockFrame):", candidate)
        self.assertIn("T.sblock_attr", candidate)
        self.assertTrue(any("legacy builder" in issue["reason"] for issue in issues))
        self.assertTrue(any(issue["node"] == "tir.tvm_access_ptr" for issue in issues))

    def test_python_known_tvm_binding_and_ffi_wrapper(self):
        original = """from tvm import tir
# tir.Block must remain in this comment
text = "description: tir.Block"
x = tir.BlockRealize
return_value = _ffi_api.HostLegalize()
"""
        candidate, issues = migrate.transform_python(original, {"tl.transform.HostLegalize": "tl.transform.A2A3HostLegalize"})
        self.assertFalse(issues)
        self.assertIn("from tvm import tirx", candidate)
        self.assertIn("tirx.SBlockRealize", candidate)
        self.assertIn("_ffi_api.A2A3HostLegalize()", candidate)
        self.assertIn("# tir.Block must remain in this comment", candidate)
        self.assertIn('text = "description: tir.Block"', candidate)

    def test_native_registration_collision_blocks_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "src/existing.cc").write_text(
                'TVM_FFI_STATIC_INIT_BLOCK() { refl::GlobalDef().def("tl.transform.A2A3HostLegalize", Factory); }'
            )
            original = root / "host.cc"
            original.write_text('TVM_REGISTER_GLOBAL("tl.transform.HostLegalize").set_body_typed(Factory);')
            output = root / "candidate.cc"
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(migrate.__file__)),
                    "migrate",
                    "--input",
                    str(original),
                    "--output",
                    str(output),
                    "--main-root",
                    str(root),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(process.returncode, 1)
            self.assertIn("collision", process.stderr)
            self.assertFalse(output.exists())
            self.assertEqual(
                (root / "src/existing.cc").read_text(),
                'TVM_FFI_STATIC_INIT_BLOCK() { refl::GlobalDef().def("tl.transform.A2A3HostLegalize", Factory); }',
            )

    def test_existing_candidate_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate"
            path.write_text("reviewer changes")
            with self.assertRaises(FileExistsError):
                migrate.write_new(path, "replacement")
            self.assertEqual(path.read_text(), "reviewer changes")

    def test_duplicate_destination_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.json"
            path.write_text(json.dumps({"mappings": [{"source": "a", "target": "same"}, {"source": "b", "target": "same"}]}))
            with self.assertRaisesRegex(ValueError, "duplicate destination"):
                migrate.load_mapping(path)

    def test_checked_in_inventory_covers_proposal_and_wrapper_aliases(self):
        data = json.loads(Path(migrate.__file__).with_name("ffi_map.json").read_text())
        by_name = {item["source"]: item for item in data["mappings"]}
        # AnnotateDeviceRegions is inactive but explicitly retained by proposal.
        self.assertEqual(by_name["tl.transform.AnnotateDeviceRegions"]["scope"], "proposal_retained")
        for name in (
            "LayoutInference",
            "LowerTileOp",
            "LegalizeVectorizedLoop",
            "LegalizeSafeMemoryAccess",
            "PipelinePlanning",
            "InjectSoftwarePipeline",
            "ConfigIndexBitwidth",
            "FlattenBuffer",
            "VectorizeLoop",
            "AnnotateDeviceRegions",
        ):
            self.assertIn("tl.transform." + name, by_name)
        source_files = {item["source_path"] for item in data["mappings"]}
        self.assertIn("src/transform/allocate_tmp_buffer.cc", source_files)
        self.assertIn("src/transform/cross_core_pipeline.cc", source_files)
        self.assertEqual(by_name["tl.transform.InferAllocScope"]["uses"][0]["python_name"], "AscendInferBufferScope")
        self.assertEqual(by_name["tl.transform.HostLegalize"]["uses"][0]["python_name"], "HostProcesser")
        self.assertEqual(by_name["target.build.tilelang_ascend_pto"]["target"], "target.build.tilelang_pto_a2a3")
        self.assertNotIn("tl.transform.AnnotateDeviceRegions", data["excluded_registrations"])
        self.assertEqual(len(by_name), 36)

    def test_host_calibration_refuses_source_drift(self):
        with self.assertRaisesRegex(ValueError, "source drift"):
            migrate.calibrate_host("// shortened replacement pass\n")


if __name__ == "__main__":
    unittest.main()
