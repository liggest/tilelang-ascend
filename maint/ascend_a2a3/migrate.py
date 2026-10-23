#!/usr/bin/env python3
# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Conservative, source-preserving TA -> A2/A3 migration preparation.

This tool writes candidates only. Structural rewrites require review; a blocked
candidate is never emitted as a normal compilable source file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tokenize

HERE = Path(__file__).resolve().parent
BASELINE = "0deb8b93da11b7f0becbcded732d9633b7ca4e34"
TIRX_HISTORY = "b939fa0137086ee05729be2ce7a066d0e102159d"
# Proposal source inventory (analysis section 2.2). Keep explicit lists so an
# inactive proposal component cannot silently disappear from future inventories.
PROPOSAL_SHARED = {
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
}
PROPOSAL_EXCLUSIVE_FILES = {
    "ascend_storage_rewrite.cc",
    "ascend_lower_parallel_to_vector.cc",
    "ascend_sync_insert.cc",
    "cross_core_pipeline.cc",
    "ascend_workspace_reduction.cc",
    "ascend_combinecv.cc",
    "allocate_tmp_buffer.cc",
    "ascend_vid_reduction.cc",
    "ascend_memory_planning.cc",
    "ascend_sync_insert_vs.cc",
    "ascend_infer_buffer_scope.cc",
    "ascend_tail_mask_propagation.cc",
    "ascend_lower_opaque_block.cc",
    "ascend_collect_buffer_shape.cc",
    "ascend_pto_save_buffer_shape.cc",
    "ascend_host.cc",
    "ascend_vector_instruction_selection.cc",
    "ascend_vector_mask_legalize.cc",
}
# Global namespace strings and Python _ffi_api attributes are migrated. C++
# symbols and namespaces require review rather than an unsafe global rename.
REGISTRATION = re.compile(
    r'(?:TVM_REGISTER_GLOBAL\s*\(|\.def(?:_packed)?\s*\()\s*"'
    r'((?:tl\.transform\.|target\.build\.)[^"\n]+)"'
)
CPP_PARTS = re.compile(r'(/\*.*?\*/|//[^\n]*|R"([^ ()\\\t\n]{0,16})\(.*?\)\2"|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')', re.S)
NODE_NAMES = {
    "Block": "SBlock",
    "BlockNode": "SBlockNode",
    "BlockRealize": "SBlockRealize",
    "BlockRealizeNode": "SBlockRealizeNode",
    "LetStmt": "Bind",
    "LetStmtNode": "BindNode",
    "Allocate": "AllocBuffer",
    "AllocateNode": "AllocBufferNode",
    "BlockFrame": "SBlockFrame",
    "block_attr": "sblock_attr",
}
# b939fa01 shows these changing both shape and lifetime. Token rename alone is
# insufficient (Bind/AllocBuffer/DeclBuffer no longer own a body).
UNSAFE = {
    "LetStmt": "Bind has no body; review enclosing SeqStmt and variable lifetime",
    "LetStmtNode": "BindNode has no body; review all visitor and body accesses",
    "Allocate": "AllocBuffer takes a Buffer and has no body/condition",
    "AllocateNode": "AllocBufferNode has different fields and allocation lifetime",
    "AllocateConst": "AllocateConst requires manual constant allocation lowering",
    "AllocateConstNode": "AllocateConstNode requires manual constant allocation lowering",
    "DeclBuffer": "DeclBuffer no longer owns a body; review enclosing SeqStmt",
    "DeclBufferNode": "DeclBufferNode no longer owns a body",
    "ProducerRealize": "legacy producer nodes require manual lowering",
    "ProducerRealizeNode": "legacy producer nodes require manual lowering",
    "AssertStmt": "AssertStmt scope/body representation requires manual review",
    "AssertStmtNode": "AssertStmt scope/body representation requires manual review",
}


def source_parts(text: str):
    """Yield (kind, text), preserving comments and string contents byte for byte."""
    start = 0
    for match in CPP_PARTS.finditer(text):
        if match.start() > start:
            yield "code", text[start : match.start()]
        part = match.group(0)
        yield ("comment" if part.startswith(("//", "/*")) else "string"), part
        start = match.end()
    yield "code", text[start:]


def uncomment(text: str) -> str:
    return "".join("\n" * part.count("\n") if kind == "comment" else part for kind, part in source_parts(text))


def registrations(root: Path, ignored: set[Path] | None = None) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted((root / "src").rglob("*")):
        if path.suffix not in {".cc", ".h"} or path.resolve() in (ignored or set()):
            continue
        clean = uncomment(path.read_text())
        for match in REGISTRATION.finditer(clean):
            result.setdefault(match.group(1), []).append(f"{path.relative_to(root)}:{clean.count(chr(10), 0, match.start()) + 1}")
    return result


def inventory(ta_root: Path) -> dict:
    """Resolve Python wrappers, active pipeline calls, and native registrations."""
    dirty = subprocess.check_output(
        [
            "git",
            "-C",
            str(ta_root),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src/transform",
            "src/target",
            "tilelang/transform",
            "tilelang/engine",
        ],
        text=True,
    )
    if dirty.strip():
        raise ValueError("TA migration inputs differ from recorded HEAD: " + dirty.strip())
    wrappers = ast.parse((ta_root / "tilelang/transform/__init__.py").read_text())
    aliases = {}
    for func in wrappers.body:
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            aliases[func.name] = sorted(
                {
                    node.func.attr
                    for node in ast.walk(func)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "_ffi_api"
                }
            )
    calls = []
    for filename in ("tilelang/engine/phase.py", "tilelang/engine/lower.py"):
        tree = ast.parse((ta_root / filename).read_text())
        for func in tree.body:
            if not isinstance(func, ast.FunctionDef):
                continue
            for node in sorted(ast.walk(func), key=lambda n: getattr(n, "lineno", 0)):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if ast.unparse(node.func.value) != "tilelang.transform":
                    continue
                name = node.func.attr
                if name == "get_pass_context":
                    continue
                native = aliases.get(name)
                if not native:
                    raise ValueError(f"unresolved pass wrapper {name} at {filename}:{node.lineno}")
                calls.append(
                    {
                        "file": filename,
                        "function": func.name,
                        "line": node.lineno,
                        "python_name": name,
                        "ffi_names": ["tl.transform." + n for n in native],
                    }
                )
    registry = registrations(ta_root)
    phase_names = {name for call in calls if call["file"].endswith("phase.py") for name in call["ffi_names"]}
    engine_names = {name for call in calls if call["file"].endswith("lower.py") for name in call["ffi_names"]}
    proposal_names = {"tl.transform." + name for name in PROPOSAL_SHARED}
    found_files = {Path(path.split(":")[0]).name for paths in registry.values() for path in paths}
    missing_files = PROPOSAL_EXCLUSIVE_FILES - found_files
    if missing_files:
        raise ValueError("proposal-exclusive files missing registrations: " + str(sorted(missing_files)))
    proposal_names |= {
        name for name, paths in registry.items() if any(Path(path.split(":")[0]).name in PROPOSAL_EXCLUSIVE_FILES for path in paths)
    }
    required = phase_names | engine_names | proposal_names
    # Include new Ascend auxiliary registrations and optional passes, even if
    # not currently invoked by phase.py (e.g. terminal catalog, dynamic loops).
    required |= {name for name, paths in registry.items() if any(Path(path.split(":")[0]).name.startswith("ascend_") for path in paths)}
    required |= {"tl.transform.FrontendLegalize", "tl.transform.LoopVectorizeDynamic"}
    required |= {"target.build.tilelang_ascend", "target.build.tilelang_ascend_pto"}
    mappings = []
    for name in sorted(required):
        paths = registry.get(name, [])
        if len(paths) != 1:
            raise ValueError(f"registration must exist exactly once: {name}: {paths}")
        target = (
            "tl.transform.A2A3" + name.rsplit(".", 1)[1]
            if name.startswith("tl.transform.")
            else {
                "target.build.tilelang_ascend": "target.build.tilelang_ascend_a2a3",
                "target.build.tilelang_ascend_pto": "target.build.tilelang_pto_a2a3",
            }[name]
        )
        path = paths[0].split(":")[0]
        mappings.append(
            {
                "source": name,
                "target": target,
                "source_path": path,
                "source_sha256": hashlib.sha256((ta_root / path).read_bytes()).hexdigest(),
                "target_path": (
                    path.replace("src/target/", "src/ascend/a2a3/codegen/", 1)
                    if path.startswith("src/target/")
                    else path.replace("src/", "src/ascend/a2a3/", 1)
                ),
                "uses": [c for c in calls if name in c["ffi_names"]],
                "scope": (
                    "phase"
                    if name in phase_names
                    else "engine_helper"
                    if name in engine_names
                    else "proposal_retained"
                    if name in proposal_names
                    else "auxiliary_optional"
                ),
                "proposal": name in proposal_names,
                "kind": (
                    "auxiliary"
                    if name == "tl.transform.AscendVectorTerminalCatalog"
                    else "pass"
                    if name.startswith("tl.transform.")
                    else "build"
                ),
            }
        )
    # Also record all other transform registrations without declaring CUDA/HIP
    # leftovers to be required Ascend migration components.
    return {
        "ta_commit": git_head(ta_root),
        "tirx_history": TIRX_HISTORY,
        "proposal_exclusive_files": sorted(PROPOSAL_EXCLUSIVE_FILES),
        "proposal_shared_passes": sorted(PROPOSAL_SHARED),
        "mappings": mappings,
        "pipeline_calls": calls,
        "excluded_registrations": {k: v for k, v in registry.items() if k.startswith("tl.transform.") and k not in required},
    }


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def source_provenance(path: Path) -> dict:
    """Associate a real source path with Git; do not invent provenance for copies."""
    process = subprocess.run(["git", "-C", str(path.resolve().parent), "rev-parse", "--show-toplevel"], text=True, capture_output=True)
    if process.returncode:
        return {"status": "unverified_copy"}
    root = Path(process.stdout.strip())
    relative = path.resolve().relative_to(root).as_posix()
    original = subprocess.run(["git", "-C", str(root), "show", "HEAD:" + relative], capture_output=True)
    return {
        "repository": str(root),
        "commit": git_head(root),
        "path": relative,
        "matches_head": original.returncode == 0 and original.stdout == path.read_bytes(),
    }


def load_mapping(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    result = {}
    for item in data["mappings"]:
        old, new = item["source"], item["target"]
        if old in result:
            raise ValueError(f"duplicate source mapping: {old}")
        if new in result.values():
            raise ValueError(f"duplicate destination mapping: {new}")
        result[old] = new
    return result


def collision_report(mapping: dict[str, str], registry: dict[str, list[str]]) -> list[dict]:
    return [{"target": target, "locations": registry[target]} for target in sorted(set(mapping.values()) & set(registry))]


def transform_cpp(text: str, mapping: dict[str, str]) -> tuple[str, list[dict]]:
    issues = []
    result = []
    line = 1
    for kind, part in source_parts(text):
        if kind == "code":
            for match in re.finditer(r"\b(" + "|".join(UNSAFE) + r")\b", part):
                issues.append({"line": line + part.count("\n", 0, match.start()), "node": match.group(0), "reason": UNSAFE[match.group(0)]})
            part = re.sub(r"\btir\s*::", "tirx::", part)
            part = re.sub(r"\btvm\s*::\s*tir\b", "tvm::tirx", part)
            part = re.sub(r"\bnamespace\s+tir\b", "namespace tirx", part)
            part = re.sub(r"\b(" + "|".join(NODE_NAMES) + r")\b", lambda m: NODE_NAMES[m.group(0)], part)
        elif kind == "string":
            # Includes and exact FFI strings only; leave diagnostics/user strings.
            inner = part[1:-1]
            if inner in mapping:
                part = '"' + mapping[inner] + '"'
            elif re.fullmatch(r"tir\.[A-Za-z_]\w*", inner):
                issues.append(
                    {"line": line, "node": inner, "reason": "legacy op identifier string: check tirx Op registry and adapt explicitly"}
                )
        result.append(part)
        line += part.count("\n")
    output = "".join(result)
    output = re.sub(r'(#\s*include\s*[<"])(?:tvm/)?tir/', lambda m: m.group(0).replace("tir/", "tirx/"), output)
    # Source references private TVM headers that were moved rather than exposed.
    if "tvm/tirx/data_type_rewriter.h" in output:
        issues.append(
            {"node": "data_type_rewriter.h", "reason": "private header moved to TVM src/tirx/ir; resolve include relative to destination"}
        )
    for name in ("utils.h", "target_info.h"):
        if "tvm/tirx/" + name in output:
            issues.append(
                {"node": name, "reason": "legacy public header was removed; inspect used symbols rather than substituting a path"}
            )
    for name in ("virtual_thread", "buffer_dim_align", "volatile_scope"):
        if "tirx::attr::" + name in output:
            issues.append({"node": name, "reason": "attribute moved between s_tir/tl/tirx; adapt ownership explicitly"})
    if "TVM_REGISTER_GLOBAL" in output:
        issues.append(
            {"node": "TVM_REGISTER_GLOBAL", "reason": "manually migrate registration body to TVM_FFI_STATIC_INIT_BLOCK/GlobalDef"}
        )
    return output, issues


def transform_python(text: str, mapping: dict[str, str]) -> tuple[str, list[dict]]:
    tree = ast.parse(text)
    import_issues = []
    imported_tir = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.ImportFrom, ast.Import)):
            modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names]
            for module in modules:
                if module.startswith("tvm.script.ir_builder.tir"):
                    import_issues.append(
                        {
                            "line": node.lineno,
                            "node": module,
                            "reason": "legacy builder module moved to tvm.tirx.script.builder; review frame inheritance and imports",
                        }
                    )
        if isinstance(node, ast.Attribute) and node.attr in {"virtual_thread", "buffer_dim_align", "volatile_scope"}:
            import_issues.append(
                {"line": node.lineno, "node": node.attr, "reason": "attribute ownership moved between s_tir/tl/tirx; review explicitly"}
            )
        if isinstance(node, ast.ImportFrom) and node.module == "tvm":
            for item in node.names:
                if item.name == "tir":
                    imported_tir = True
                    if item.asname and item.asname != "tir":
                        import_issues.append(
                            {
                                "line": node.lineno,
                                "node": item.asname,
                                "reason": "aliased TVM TIR import: manually adapt node references through alias",
                            }
                        )
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tvm.tir"):
            import_issues.append(
                {
                    "line": node.lineno,
                    "node": node.module,
                    "reason": "direct TVM TIR symbol imports require adapting imported and bare node names",
                }
            )
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name.startswith("tvm.tir") and item.asname:
                    import_issues.append(
                        {
                            "line": node.lineno,
                            "node": item.asname,
                            "reason": "aliased TVM TIR module import requires manual reference review",
                        }
                    )
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    short = {a.rsplit(".", 1)[1]: b.rsplit(".", 1)[1] for a, b in mapping.items() if a.startswith("tl.transform.")}
    for i, token in enumerate(tokens):
        if token.type == tokenize.NAME:
            value = token.string
            if value == "tir":
                tokens[i] = token._replace(string="tirx")
                if not imported_tir and not (i >= 2 and tokens[i - 1].string == "." and tokens[i - 2].string == "tvm"):
                    import_issues.append(
                        {
                            "line": token.start[0],
                            "node": "tir",
                            "reason": "unresolved bare tir binding; review identifier before using candidate",
                        }
                    )
            elif value in {"BlockFrame", "block_attr"}:
                tokens[i] = token._replace(string=NODE_NAMES[value])
            elif i >= 2 and tokens[i - 1].string == "." and tokens[i - 2].string == "_ffi_api" and value in short:
                tokens[i] = token._replace(string=short[value])
            elif value in NODE_NAMES and i >= 2 and tokens[i - 1].string == "." and tokens[i - 2].string in {"tir", "tirx"}:
                tokens[i] = token._replace(string=NODE_NAMES[value])
        elif token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (SyntaxError, ValueError):
                continue
            if isinstance(value, str) and value in mapping:
                tokens[i] = token._replace(string=repr(mapping[value]))
            elif isinstance(value, str) and re.fullmatch(r"tir\.[A-Za-z_]\w*", value):
                import_issues.append(
                    {
                        "line": token.start[0],
                        "node": value,
                        "reason": "legacy op identifier string: check tirx Op registry and adapt explicitly",
                    }
                )
    issues = [
        {"line": t.start[0], "node": t.string, "reason": UNSAFE[t.string]}
        for t in tokenize.generate_tokens(io.StringIO(text).readline)
        if t.type == tokenize.NAME and t.string in UNSAFE
    ]
    return tokenize.untokenize(tokens), issues + import_issues


def guard_python(text: str) -> str:
    """Keep future imports legal, then stop the candidate before it executes."""
    tree = ast.parse(text)
    after = max((node.end_lineno for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"), default=0)
    lines = text.splitlines(keepends=True)
    lines.insert(after, "raise RuntimeError('A2A3 migration requires manual structural review')\n")
    return "".join(lines)


def write_new(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous candidate or report, even on reruns.
    with path.open("x") as stream:
        stream.write(content)


def calibrate_host(text: str) -> tuple[str, list[dict]]:
    """Reviewed interface adaptations of the complete, pinned TA host pass.

    The Bind visitor is safe here only because the original LetStmt visitor
    delegates to its base class and never inspects/rewrites a lexical body.
    This exception must not be applied to other passes.
    """
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != "d0b8a9e21b0360f7b9a23a44c24bebb6d10b82743c8e2c4fcdf9dd17b0c7f16e":
        raise ValueError(f"HostLegalize source drift ({digest}); review calibration before updating it")
    rules = [
        ("tvm/tir/", "tvm/tirx/", "public TIRX headers"),
        ("tir/analysis/", "tirx/analysis/", "private TIRX analysis header"),
        ("tir::", "tirx::", "TIRX namespace"),
        ("namespace tir;", "namespace tirx;\nusing namespace ffi;", "TIRX and FFI containers"),
        ("LetStmtNode", "BindNode", "reviewed forwarding-only visitor; no body access"),
        ("#include <tvm/tirx/utils.h>\n", "", "removed TVM header; no used symbols"),
        (
            '#include "../op/builtin.h"\n#include "./common/collector.h"',
            "#include <tvm/ffi/reflection/registry.h>\n#include <tvm/ir/attrs.h>\n#include <unordered_set>\n#include <utility>",
            "replace unused TA dependencies with explicit FFI/STL headers",
        ),
        ("namespace tl {", "namespace tl {\nnamespace a2a3 {", "isolate complete TA class and symbols"),
        ("substituter.VisitStmt(f->body)", "substituter(f->body)", "current mutator public entry point"),
        (
            'auto fn_attr = fptr->attrs.CopyOnWrite();\n    fn_attr->dict.Set("tiling_map", substituter.tiling_map_);\n    fn_attr->dict.Set("var_sequence", substituter.var_sequence_);',
            'f = WithAttr(std::move(f), "tiling_map", substituter.tiling_map_);\n    f = WithAttr(std::move(f), "var_sequence", substituter.var_sequence_);',
            "current immutable attribute API; preserve source collectors",
        ),
        (
            "std::unordered_set<const VarNode*> cmp_set_;",
            "std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> cmp_set_;",
            "retain variable identity with owning handles",
        ),
        ("this->cmp_set_.count(v)", "this->cmp_set_.count(GetRef<Var>(v))", "identity handle lookup"),
        ("this->cmp_set_.insert(var.get())", "this->cmp_set_.insert(var)", "identity handle insertion"),
        ("cmp_set_.insert((iv->var.get()))", "cmp_set_.insert(iv->var)", "thread binding identity"),
        ("HostLegalize()", "A2A3HostLegalize()", "A2/A3 C++ factory"),
        ('"tl.HostLegalize"', '"tl.A2A3HostLegalize"', "A2/A3 pass display name"),
        (
            'TVM_REGISTER_GLOBAL("tl.transform.HostLegalize")\n    .set_body_typed(HostLegalize);',
            'TVM_FFI_STATIC_INIT_BLOCK() {\n  namespace refl = tvm::ffi::reflection;\n  refl::GlobalDef().def("tl.transform.A2A3HostLegalize", A2A3HostLegalize);\n}',
            "current FFI registration API and isolated global name",
        ),
        ("\n}\n}", "\n} // namespace a2a3\n} // namespace tl\n} // namespace tvm\n", "close isolated namespace"),
    ]
    changes = []
    for old, new, reason in rules:
        count = text.count(old)
        if count == 0:
            raise ValueError(f"calibration rule no longer matches: {reason}")
        text = text.replace(old, new)
        changes.append({"reason": reason, "count": count})
    # Match the repository include grouping and formatting after native review.
    start = text.index('#include "arith/')
    end = text.index("\nnamespace tvm", start)
    text = (
        text[:start]
        + """#include <unordered_set>
#include <utility>

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/attrs.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "arith/ir_mutator_with_analyzer.h"
#include "tirx/analysis/var_use_def_analysis.h"
"""
        + text[end:]
    )
    text = text.replace("const tirx::VarNode* v", "const tirx::VarNode *v")
    text = text.replace("op->min) || isNeedTiling", "op->min) ||\n        isNeedTiling")
    text = text.replace("bool after_thread_flag = false;\n\n};", "bool after_thread_flag = false;\n};")
    changes.append({"reason": "reviewed include grouping and whitespace formatting", "count": 1})
    return text, changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory")
    inv.add_argument("--ta-root", type=Path, required=True)
    inv.add_argument("--main-root", type=Path)
    inv.add_argument("--output", type=Path, required=True)
    mig = sub.add_parser("migrate")
    mig.add_argument("--input", type=Path, required=True)
    mig.add_argument("--output", type=Path, required=True)
    mig.add_argument("--mapping", type=Path, default=HERE / "ffi_map.json")
    mig.add_argument("--main-root", type=Path, required=True)
    mig.add_argument(
        "--replacing",
        type=Path,
        action="append",
        default=[],
        help="explicit native file being replaced; excludes only this file from collision checks",
    )
    host = sub.add_parser("calibrate-host", help="reproduce the reviewed complete TA HostProcesser candidate")
    host.add_argument("--ta-root", type=Path, required=True)
    host.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "calibrate-host":
            original = args.ta_root / "src/transform/ascend_host.cc"
            candidate, changes = calibrate_host(original.read_bytes().decode("utf-8"))
            report_path = args.output.with_name(args.output.name + ".json")
            if args.output.exists() or report_path.exists():
                raise FileExistsError(f"candidate/report already exists: {args.output}")
            write_new(args.output, candidate)
            write_new(
                report_path,
                json.dumps(
                    {
                        "ta_commit": git_head(args.ta_root),
                        "expected_ta_commit": BASELINE,
                        "tirx_history": TIRX_HISTORY,
                        "source_path": "src/transform/ascend_host.cc",
                        "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                        "status": "reviewed_candidate_requires_native_and_behavior_validation",
                        "changes": changes,
                    },
                    indent=2,
                )
                + "\n",
            )
            print(f"reviewed full-source HostProcesser candidate: {args.output}")
            return 0
        if args.command == "inventory":
            data = inventory(args.ta_root)
            if args.main_root:
                data["destination_collisions"] = collision_report(
                    {x["source"]: x["target"] for x in data["mappings"]}, registrations(args.main_root)
                )
            write_new(args.output, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            print(f"{len(data['mappings'])} mappings; {len(data['pipeline_calls'])} pipeline/engine calls; {args.output}")
            return 0
        if args.input.resolve() == args.output.resolve():
            raise ValueError("source and candidate must be different files")
        text = args.input.read_bytes().decode("utf-8")
        mapping = load_mapping(args.mapping)
        mapping_data = json.loads(args.mapping.read_text())
        # Check only registrations emitted by this file, not unrelated entries.
        emitted = {name: mapping[name] for name in REGISTRATION.findall(uncomment(text)) if name in mapping}
        ignored = {(args.main_root / path).resolve() for path in args.replacing}
        collisions = collision_report(emitted, registrations(args.main_root, ignored))
        if collisions:
            raise ValueError("destination registration collision: " + json.dumps(collisions))
        if args.input.suffix == ".py":
            candidate, issues = transform_python(text, mapping)
        elif args.input.suffix in {".cc", ".h"}:
            candidate, issues = transform_cpp(text, mapping)
        else:
            raise ValueError("only .cc, .h and .py inputs are supported")
        unknown = [name for name in REGISTRATION.findall(uncomment(text)) if name not in mapping]
        issues += [{"node": name, "reason": "registration is outside the reviewed mapping"} for name in unknown]
        if args.input.suffix in {".cc", ".h"}:
            issues.append(
                {
                    "node": "native integration",
                    "reason": "review a2a3 C++ symbol namespace, relative includes, FFI/container APIs and target attributes; compile before integration",
                }
            )
        report = {
            "input": str(args.input),
            "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "ta_commit": mapping_data["ta_commit"],
            "source_provenance": source_provenance(args.input),
            "mapping_file": str(args.mapping),
            "source_associations": [
                {"source_path": item["source_path"], "source": item["source"], "target": item["target"]}
                for item in mapping_data["mappings"]
                if item["source"] in emitted
            ],
            "status": "manual_review_required" if issues else "candidate",
            "tirx_history": TIRX_HISTORY,
            "issues": issues,
        }
        # Preflight every output path to avoid producing a candidate without a report.
        report_path = args.output.with_name(args.output.name + ".json")
        if args.output.exists() or report_path.exists():
            raise FileExistsError(f"candidate/report already exists: {args.output}")
        if issues:
            candidate = (
                guard_python(candidate) if args.input.suffix == ".py" else "#error A2A3_MIGRATION_REQUIRES_MANUAL_REVIEW\n" + candidate
            )
        write_new(args.output, candidate)
        write_new(report_path, json.dumps(report, indent=2) + "\n")
        print(f"{report['status']}: {args.output}; report: {report_path}")
        return 2 if issues else 0
    except (ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
