#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Pinned, non-executable TA ecology migration candidates, never an installer."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

BASELINE = "0deb8b93da11b7f0becbcded732d9633b7ca4e34"
MANIFEST = Path(__file__).with_name("ecosystem_manifest.json")
LANGUAGE_MODULES = {
    "ascend",
    "ascend_tile",
    "allocate",
    "kernel",
    "copy_op",
    "reduce_ascend",
    "customize",
    "warpgroup",
    "pto",
}
LAYOUT_NAMES = {"make_zn_layout", "make_nz_layout", "make_col_major_layout"}
GUARD_MESSAGE = "A2A3 ecology candidate requires manual review; not an installed migration"
RESERVED = "examples/autotune/"


def destination_for(source_path: str) -> str:
    """Map an original TA example/test path to the current adopted layout.

    Pure path mapping only: the caller must check the source manifest and review
    classification. In particular, a mapped A5 path is NOT permission to migrate it.
    CI resolvers must never fall back to a same-named main CUDA/A5 source.
    """
    path = PurePosixPath(source_path)
    if str(path) != source_path or path.is_absolute() or ".." in path.parts or "\\" in source_path:
        raise ValueError(f"not a normalized relative source path: {source_path!r}")
    if len(path.parts) < 2 or path.parts[0] not in {"examples", "examples_experiment", "testing"}:
        raise ValueError(f"no adopted example/test namespace for {source_path!r}")
    if path.parts[1] == "ascend_a2a3":
        raise ValueError("input must be the original TA path, not an adopted path")
    # Mainline had no experimental tree: preserve the original TA hierarchy.
    if path.parts[0] == "examples_experiment":
        return source_path
    return str(PurePosixPath(path.parts[0], "ascend_a2a3", *path.parts[1:]))


def load_manifest(path: Path = MANIFEST) -> dict[str, dict]:
    data = json.loads(path.read_text())
    if data["schema_version"] != 1 or data["source_pin"] != BASELINE:
        raise ValueError("unsupported manifest schema or source pin")
    entries = {entry["path"]: entry for entry in data["entries"]}
    if len(entries) != len(data["entries"]):
        raise ValueError("duplicate source paths in manifest")
    for source_path in entries:
        p = PurePosixPath(source_path)
        if p.is_absolute() or ".." in p.parts or str(p) != source_path or "\\" in source_path:
            raise ValueError("unsafe manifest path")
    return entries


def git(ta_root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ta_root), *args])


def pinned_source(ta_root: Path, entry: dict) -> bytes:
    if git(ta_root, "rev-parse", "HEAD").decode().strip() != BASELINE:
        raise ValueError("TA HEAD differs from the fixed migration pin")
    path = entry["path"]
    if entry["kind"] != "blob":
        raise ValueError("gitlinks are inventory entries, not candidate source files")
    blob = git(ta_root, "show", f"{BASELINE}:{path}")
    digest = hashlib.sha256(blob).hexdigest()
    blob_id = git(ta_root, "rev-parse", f"{BASELINE}:{path}").decode().strip()
    if digest != entry["sha256"] or blob_id != entry["git_blob"]:
        raise ValueError(f"manifest hash mismatch for {path}")
    working = ta_root / path
    if working.is_symlink() or not working.is_file() or working.resolve() != ta_root.resolve() / path:
        raise ValueError(f"source is missing or traverses a symlink: {path}")
    if working.read_bytes() != blob:
        raise ValueError(f"source working tree drift: {path}")
    return blob


def _offsets(text: str):
    # AST columns are UTF-8 byte offsets, not Python character offsets.
    lines = text.encode().splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    return starts


def rewrite_imports(text: str) -> tuple[str, list[dict], list[dict]]:
    """Replace approved import spans only; never kernel bodies or strings."""
    tree = ast.parse(text)
    starts = _offsets(text)
    raw = text.encode()
    edits, audit, issues = [], [], []

    def issue(node, reason):
        issues.append({"line": node.lineno, "reason": reason})

    def replace(node, replacement):
        begin = starts[node.lineno - 1] + node.col_offset
        end = starts[node.end_lineno - 1] + node.end_col_offset
        original = raw[begin:end].decode()
        edits.append((begin, end, replacement.encode()))
        audit.append({"line": node.lineno, "before": original, "after": replacement})

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # An unaliased import binds the first component (tilelang), so a
            # module-only rewrite would change attribute resolution semantics.
            for alias in node.names:
                module = alias.name
                if module == "tilelang.language" or module.startswith("tilelang.language."):
                    suffix = module.removeprefix("tilelang.language").removeprefix(".")
                    if suffix and suffix not in LANGUAGE_MODULES:
                        issue(node, f"unreviewed language submodule: {module}")
                    elif alias.asname is None:
                        issue(node, f"bare module import needs binding review: {module}")
                    else:
                        begin = starts[alias.lineno - 1] + alias.col_offset
                        end = begin + len(module.encode())
                        changed = "tilelang.ascend.a2a3.language" + ("." + suffix if suffix else "")
                        edits.append((begin, end, changed.encode()))
                        audit.append({"line": alias.lineno, "before": module, "after": changed})
                elif module.startswith(("tvm", "tilelang.engine", "tilelang.jit.adapter", "tilelang.cache")):
                    issue(node, f"IR/runtime import requires API review: {module}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            segment = ast.get_source_segment(text, node)
            replacement_module = None
            if node.level:
                issue(node, "relative import needs destination dependency review")
            elif module == "tilelang" and any(a.name == "language" for a in node.names):
                if len(node.names) == 1:
                    replacement_module = "tilelang.ascend.a2a3"
                elif node.lineno == node.end_lineno and "#" not in segment:
                    lang = [ast.unparse(a) for a in node.names if a.name == "language"]
                    other = [ast.unparse(a) for a in node.names if a.name != "language"]
                    replace(node, "from tilelang.ascend.a2a3 import " + ", ".join(lang) + "; from tilelang import " + ", ".join(other))
                else:
                    issue(node, "mixed language import with comments/multiple lines needs review")
            elif module == "tilelang.language" or module.startswith("tilelang.language."):
                suffix = module.removeprefix("tilelang.language").removeprefix(".")
                if suffix and suffix not in LANGUAGE_MODULES:
                    issue(node, f"unreviewed language submodule: {module}")
                else:
                    replacement_module = "tilelang.ascend.a2a3.language" + ("." + suffix if suffix else "")
                    if any(a.name == "*" for a in node.names):
                        issue(node, "star import export/consumer review required")
            elif module == "tilelang.intrinsics":
                if all(a.name in LAYOUT_NAMES for a in node.names):
                    replacement_module = "tilelang.ascend.a2a3.layout"
                else:
                    issue(node, "intrinsics import contains symbols outside the reviewed layout helpers")
            elif module.startswith(
                (
                    "tvm",
                    "tilelang.utils.target",
                    "tilelang.engine",
                    "tilelang.jit.adapter",
                    "tilelang.cache",
                    "tilelang.profiler",
                    "tilelang.carver",
                )
            ):
                issue(node, f"IR/platform/runtime import requires API review: {module}")
            if replacement_module:
                changed = re.sub(
                    r"^(from\s+)" + re.escape(module) + r"(?=\s+import\b)",
                    lambda m, target_module=replacement_module: m.group(1) + target_module,
                    segment,
                    count=1,
                )
                if changed == segment:
                    issue(node, "nonstandard import spelling needs review; no rewrite performed")
                else:
                    replace(node, changed)
        elif isinstance(node, ast.Attribute):
            if node.attr in {
                "LetStmt",
                "Allocate",
                "AllocateConst",
                "BufferRealize",
                "CustomizedCode",
                "ProducerRealize",
                "DeclBuffer",
                "get_source",
                "determine_platform",
                "get_core_num",
                "get_npu_core_num",
                "_ffi_api",
            }:
                issue(node, f"legacy API/IR scope contract requires manual review: {node.attr}")
            if ast.unparse(node) == "tilelang.language":
                issue(node, "qualified tilelang.language access needs explicit binding review")
        elif isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name.endswith((".clear_cache", ".disable_cache", ".set_device", ".manual_seed")):
                issue(node, f"side effect must be isolated, never import source blindly: {name}")
    # Edits apply to nonoverlapping import spans. Leave every other byte intact.
    edits.sort(reverse=True)
    limit = len(raw)
    for begin, end, replacement in edits:
        if end > limit:
            raise ValueError("overlapping import edits")
        raw = raw[:begin] + replacement + raw[end:]
        limit = begin
    candidate = raw.decode()
    ast.parse(candidate)
    return candidate, audit, issues


def guarded(text: str) -> str:
    tree = ast.parse(text)
    # Stop at the precise AST boundary, not its line: a future import or
    # docstring may share a line with the first effectful statement.
    boundary = 0
    starts = _offsets(text)
    for index, node in enumerate(tree.body):
        if (index == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ):
            boundary = starts[node.end_lineno - 1] + node.end_col_offset
        else:
            break
    raw = text.encode()
    prefix, suffix = raw[:boundary].decode(), raw[boundary:].decode()
    if suffix.lstrip(" \t").startswith(";"):
        suffix = suffix.lstrip(" \t")[1:].lstrip(" \t")
    if prefix and not prefix.endswith("\n"):
        prefix += "\n"
    return prefix + f"raise RuntimeError({GUARD_MESSAGE!r})\n" + suffix


def prepare(ta_root: Path, output: Path, paths: list[str], decisions: dict | None = None) -> dict:
    """Preflight every source/collision before writing a new candidate directory."""
    ta_root = ta_root.resolve()
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory already exists; choose a fresh batch path")
    if output.resolve().is_relative_to(ta_root):
        raise ValueError("output must not be inside the read-only TA source")
    for ancestor in output.parents:
        if ancestor.is_symlink():
            raise ValueError("output ancestors must not be symlinks")
    entries = load_manifest()
    if decisions is not None and not isinstance(decisions, dict):
        raise ValueError("decisions must be a per-source JSON object")
    decisions = decisions or {}
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("choose at least one unique source path")
    records, files = [], []
    for path in paths:
        if path not in entries:
            raise ValueError(f"source not in fixed ecology manifest: {path}")
        entry = entries[path]
        destination = destination_for(path)
        if path.startswith(RESERVED):
            raise ValueError("regular autotune examples belong to the autotuner owner")
        if entry["classification"] == "A5":
            raise ValueError(f"explicit A5 asset cannot be migrated to A2A3: {path}")
        decision = decisions.get(path)
        if entry["classification"] != "A2A3" and not (
            isinstance(decision, dict)
            and decision.get("decision") == "a2a3"
            and isinstance(decision.get("reason"), str)
            and decision["reason"].strip()
        ):
            raise ValueError(f"explicit per-file A2A3 decision/reason required: {path}")
        if Path(path).suffix != ".py":
            raise ValueError("non-Python assets remain inventory/manual candidates only")
        original = pinned_source(ta_root, entry)
        candidate, changes, issues = rewrite_imports(original.decode())
        issues.insert(0, {"line": None, "reason": "kernel/decorator/config/IR/source/SDK/device review required"})
        result = guarded(candidate)
        ast.parse(result)
        record = {
            "source_path": path,
            "source_pin": BASELINE,
            "source_git_blob": entry["git_blob"],
            "source_sha256": entry["sha256"],
            "classification_draft": entry["classification"],
            "classification_basis": entry["classification_basis"],
            "classification_decision": decision,
            "destination_path": destination,
            "candidate_path": destination + ".candidate.py",
            "candidate_sha256": hashlib.sha256(result.encode()).hexdigest(),
            "import_changes": changes,
            "manual_review": issues,
            "state": "blocked_candidate",
            "installed": False,
        }
        records.append(record)
        files.append((record["candidate_path"], result.encode()))
        files.append((record["candidate_path"] + ".json", (json.dumps(record, indent=2) + "\n").encode()))
    report = {"schema_version": 1, "source_pin": BASELINE, "state": "blocked_candidates_only", "entries": records}
    # Exclusive creation is intentional. No --force/--apply/guard-removal mode.
    output.mkdir(parents=True, exist_ok=False)
    for relative, data in files:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    with (output / "batch.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ta-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--path", action="append", required=True, help="original path; repeat for a reviewed batch")
    parser.add_argument("--decisions", type=Path, help='JSON {source: {"decision":"a2a3", "reason":"..."}}')
    args = parser.parse_args(argv)
    try:
        decisions = json.loads(args.decisions.read_text()) if args.decisions else None
        report = prepare(args.ta_root, args.output_dir, args.path, decisions)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ecology candidate rejected: {error}", file=sys.stderr)
        return 1
    print(f"Prepared {len(report['entries'])} blocked candidates; no files installed or source modules executed.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
