"""Full source contracts and CPU helpers for the 463c SiTU/Kpool increment.

No TileLang, TVM, TorchNPU, compilation or device operation is imported/run.
These checks do not stand in for the original device pytest/CLI matrices.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("upstream_situ_kpool_ports.json")


def fingerprint(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def original_node(node):
    node = copy.deepcopy(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name == "_original_main":
            node.name = "main"
        node.decorator_list = [d for d in node.decorator_list if ast.unparse(d) != "pytest.mark.a2a3_device"]
    return node


def pure_scope(path, names, torch):
    definitions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    scope = {
        "torch": torch,
        "GLM53_HEAD_DIM": 128,
        "GLM53_POOL_SIZE": 4,
        "E4M3_MAX": 448.0,
        "HADAMARD128_SCALE": 1.0 / math.sqrt(128),
        "SUB_M": 32,
        "TORCH_TO_TL_DTYPE": {torch.float32: "float", torch.float16: "float16", torch.bfloat16: "bfloat16"},
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), scope)
    return scope


class SourceContracts(unittest.TestCase):
    def test_every_original_definition_and_global_is_retained(self):
        data = json.loads(MANIFEST.read_text())
        for entry in data["entries"]:
            target = ROOT / entry["destination_path"]
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), entry["destination_sha256"])
            if target.suffix != ".py":
                continue
            tree = ast.parse(target.read_text())
            has_original_main = any(isinstance(node, ast.FunctionDef) and node.name == "_original_main" for node in tree.body)
            definitions = {
                original_node(node).name: original_node(node)
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and not (node.name == "main" and has_original_main)
            }
            if not has_original_main:
                definitions.pop("main", None)
            for name, digest in entry["original_definition_AST_sha256"].items():
                with self.subTest(file=target.name, definition=name):
                    self.assertIn(name, definitions)
                    self.assertEqual(fingerprint(definitions[name]), digest)
            assignments = {ast.unparse(node.targets[0]): node for node in tree.body if isinstance(node, ast.Assign)}
            for name, digest in entry["original_global_AST_sha256"].items():
                with self.subTest(file=target.name, global_name=name):
                    self.assertEqual(fingerprint(assignments[name]), digest)

    def test_no_import_time_device_cache_or_cli_effects(self):
        data = json.loads(MANIFEST.read_text())
        for entry in data["entries"]:
            path = ROOT / entry["destination_path"]
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
                    continue
                text = ast.unparse(node)
                for forbidden in ["clear_cache(", "disable_cache(", "parse_args(", ".npu(", "set_default_device(", "import torch_npu"]:
                    self.assertNotIn(forbidden, text, str(path))
                self.assertNotIn("sys.path.insert", text, str(path))

    @unittest.skipUnless(os.environ.get("A2A3_TA_SOURCE_REPO"), "optional fixed-TA Git recheck needs A2A3_TA_SOURCE_REPO")
    def test_manifest_original_bytes_match_fixed_upstream_commit(self):
        data = json.loads(MANIFEST.read_text())
        for entry in data["entries"]:
            content = subprocess.check_output(
                [
                    "git",
                    "-c",
                    "protocol.allow=never",
                    "-C",
                    os.environ["A2A3_TA_SOURCE_REPO"],
                    "show",
                    data["source_pin"] + ":" + entry["source_path"],
                ],
                env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0"),
            )
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["source_sha256"])


class CPUContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.dict(os.environ, {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}):
            import torch
        cls.torch = torch
        cls.situ = pure_scope(
            ROOT / "examples/ascend_a2a3/situ_glu/test_situ_glu.py",
            [
                "golden_situ_glu",
                "get_precision",
                "check_precision",
                "make_input",
                "make_special",
                "make_paired",
                "_pick_blocks",
                "run_situ_glu",
            ],
            torch,
        )
        cls.kpool = pure_scope(
            ROOT / "examples/ascend_a2a3/kpool/example_glm53_kpool_compress.py",
            ["hadamard128_reference", "glm53_kpool_reference", "_ordered_codes", "_assert_fp8_within_one_ulp", "_validate_active_locs"],
            torch,
        )

    def test_situ_full_dtype_gate_and_up_reference_branches(self):
        t = self.torch
        for dtype in [t.float32, t.float16, t.bfloat16]:
            x = t.tensor([[1e-8, -1.0, 0.0, 4.0, 1e8, -3.0, 0.25, -2.0]], dtype=dtype)
            for left in [False, True]:
                for beta in [1.0, 4.0]:
                    for linear in [0.0, -1.0, 2.0]:
                        gate, up = x.double().chunk(2, dim=-1)
                        if not left:
                            gate, up = up, gate
                        if linear > 0:
                            up = linear * t.tanh(up / linear)
                        expected = (beta * t.tanh(gate / beta) * t.sigmoid(gate) * up).to(dtype)
                        actual = self.situ["golden_situ_glu"](x, beta=beta, linear_beta=linear, activate_left=left)
                        t.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        self.assertEqual(self.situ["get_precision"]("int32"), (0.0, 0.0, 0.0, 1.0))
        self.assertFalse(self.situ["check_precision"](t.tensor([float("inf")]), t.tensor([-float("inf")]), "float32")[0])
        self.assertFalse(self.situ["check_precision"](t.tensor([0.0]), t.tensor([float("nan")]), "float32")[0])

    def test_situ_metadata_failures_precede_any_factory_call(self):
        t = self.torch

        def forbidden(*args, **kwargs):
            raise AssertionError("no kernel factory is allowed in CPU metadata checks")

        self.situ["situ_glu"] = forbidden
        for x, kwargs, text in [
            (t.zeros(2, 3), {}, "even"),
            (t.zeros(2, 4, dtype=t.int32), {}, "unsupported dtype"),
            (t.zeros(2, 4), {"beta": 0.0}, "non-zero"),
            (t.zeros(2, 4), {"dim": 2}, "dim should"),
            (t.zeros(2, 4).t(), {}, "contiguous"),
            (t.tensor(1.0), {}, "rank"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                self.situ["run_situ_glu"](x, **kwargs)
        self.assertEqual(self.situ["run_situ_glu"](t.empty(0, 8)).shape, (0, 4))

    def test_kpool_hadamard_is_the_original_sylvester_transform(self):
        t = self.torch
        h = t.ones((1, 1))
        while h.shape[0] < 128:
            h = t.cat((t.cat((h, h), dim=1), t.cat((h, -h), dim=1)), dim=0)
        x = t.arange(256, dtype=t.float32).reshape(2, 128) / 128
        expected = x @ (h / math.sqrt(128))
        t.testing.assert_close(self.kpool["hadamard128_reference"](x), expected, rtol=2e-6, atol=2e-5)

    def test_kpool_uint8_golden_and_location_rejections(self):
        t = self.torch
        k = t.zeros((7, 4, 128), dtype=t.bfloat16)
        scores = t.zeros_like(k)
        ape = t.zeros((4, 128))
        for mode in [False, True]:
            codes, scale = self.kpool["glm53_kpool_reference"](k, scores, ape, round_scale=mode)
            self.assertEqual(codes.dtype, t.uint8)
            self.assertTrue(t.equal(codes, t.zeros_like(codes)))
            self.assertTrue(bool((scale > 0).all()))
        guard = self.kpool["_validate_active_locs"]
        for values in [[], [1, 2, 5], [1, 100000]]:
            guard(t.tensor(values, dtype=t.int64), 100001)
        for values, slots, text in [
            ([1, 2, 2], 8, "unique"),
            ([1, 100000, 100000], 100001, "unique"),
            ([-1], 8, "in \\[0, 8\\)"),
            ([8], 8, "in \\[0, 8\\)"),
        ]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, text):
                guard(t.tensor(values, dtype=t.int64), slots)
        self.kpool["_assert_fp8_within_one_ulp"](
            t.tensor([0, 1, 126, 128, 129, 254], dtype=t.uint8), t.tensor([0, 1, 126, 128, 129, 254], dtype=t.uint8)
        )
        with self.assertRaises(AssertionError):
            self.kpool["_assert_fp8_within_one_ulp"](t.tensor([3], dtype=t.uint8), t.tensor([0], dtype=t.uint8))


if __name__ == "__main__":
    unittest.main()
