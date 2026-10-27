#!/usr/bin/env python3
"""Verify the migrated TA documentation assets without importing SDKs or running examples."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(root: Path, source_repo: Path | None = None) -> dict:
    manifest = json.loads((root / "maint/ascend_a2a3/docs_manifest.json").read_text())
    errors = []
    entries = manifest["entries"]
    counts = Counter(entry["source_path"].split("/")[0] for entry in entries)
    if counts != {"docs": 98, ".agents": 122, ".opencode": 5}:
        errors.append(f"Incomplete source inventory: {dict(counts)}")
    sources = [entry["source_path"] for entry in entries]
    destinations = [entry["destination"] for entry in entries]
    if len(set(sources)) != 225 or len(set(destinations)) != 225:
        errors.append("Source/destination inventory contains duplicates or omissions")
    skills = [entry for entry in entries if entry["source_path"].endswith("/SKILL.md")]
    if len(skills) != 21:
        errors.append(f"Expected 21 complete source skills, found {len(skills)}")
    for path, digest in manifest["main_skill_sha256"].items():
        if not (root / path).is_file() or sha256((root / path).read_bytes()) != digest:
            errors.append(f"Protected main skill changed: {path}")
    for path, digest in manifest.get("protected_main_docs", {}).items():
        if not (root / path).is_file() or sha256((root / path).read_bytes()) != digest:
            errors.append(f"Protected main document/resource changed: {path}")
    python_files = 0
    shell_files = 0
    for entry in entries:
        path = root / entry["destination"]
        if not path.is_file():
            errors.append(f"Missing destination: {entry['destination']}")
            continue
        content = path.read_bytes()
        if sha256(content) != entry["destination_sha256"]:
            errors.append(f"Destination hash changed: {entry['destination']}")
        if entry.get("encoding") == "gzip":
            if int.from_bytes(content[4:8], "little") != 0:
                errors.append(f"Archive contains a variable timestamp: {entry['destination']}")
            if sha256(gzip.decompress(content)) != entry["source_sha256"]:
                errors.append(f"Archived control source bytes differ: {entry['source_path']}")
        if entry.get("original_archive"):
            archived = root / entry["original_archive"]
            if not archived.is_file():
                errors.append(f"Missing full original: {entry['original_archive']}")
            elif sha256(gzip.decompress(archived.read_bytes())) != entry["source_sha256"]:
                errors.append(f"Full original bytes differ: {entry['source_path']}")
        if path.suffix == ".py":
            ast.parse(content, filename=str(path))
            python_files += 1
        if path.suffix == ".sh":
            checked = subprocess.run(["bash", "-n", path], capture_output=True, text=True, check=False)
            if checked.returncode:
                errors.append(f"Shell syntax: {path}: {checked.stderr}")
            shell_files += 1
        if source_repo is not None:
            original = subprocess.check_output(["git", "show", f"{manifest['source_pin']}:{entry['source_path']}"], cwd=source_repo)
            if sha256(original) != entry["source_sha256"]:
                errors.append(f"Source pin hash differs: {entry['source_path']}")
    # These are intentionally inactive; no nested Sphinx or instruction entry is enabled.
    if (root / "docs/ascend_a2a3/conf.py").exists() or (root / "docs/ascend_a2a3/Makefile").exists():
        errors.append("Historical site controls must not be active")
    if (root / "AGENTS.md").exists() and not manifest.get("root_agents_activated", False):
        errors.append("Root AGENTS was activated before its manual policy gate")
    archives = manifest["archives"]
    for archive in archives:
        path = root / archive["path"]
        if not path.is_file():
            errors.append(f"Missing source archive: {archive['path']}")
            continue
        content = path.read_bytes()
        if sha256(content) != archive["archive_sha256"]:
            errors.append(f"Archive hash differs: {archive['path']}")
        if int.from_bytes(content[4:8], "little") or sha256(gzip.decompress(content)) != archive["source_sha256"]:
            errors.append(f"Archive source or mtime differs: {archive['path']}")
    missing_links = []
    pages = [root / entry["destination"] for entry in entries if entry["destination"].endswith(".md")]
    pages.extend((root / "docs/ascend_a2a3").glob("*.md"))
    for page in set(pages):
        content = page.read_text()
        targets = re.findall(r"!?\[[^\]\n]*\]\(([^\s)]+)\)", content)
        targets += re.findall(r"(?:```|~~~)\{(?:figure|image)\}\s+([^\s]+)", content)
        targets += re.findall(r"<img[^>]*?src=[\"']([^\"']+)", content)
        for raw in targets:
            link = urlsplit(raw.strip("<>"))
            if link.scheme or not link.path or link.path.startswith("/") or "{" in raw:
                continue
            target = page.parent / unquote(link.path)
            if not target.exists():
                missing_links.append({"page": str(page.relative_to(root)), "target": raw})
    if missing_links:
        errors.append(f"Missing local Markdown/figure links: {missing_links}")
    return {
        "source_pin": manifest["source_pin"],
        "source_counts": dict(counts),
        "skills": len(skills),
        "protected_main_skills": len(manifest["main_skill_sha256"]),
        "archives": len(archives),
        "python_ast": python_files,
        "shell_syntax": shell_files,
        "local_links_missing": missing_links,
        "errors": errors,
        "scope": "Static source/assets/policy audit only; no SDK import, example execution, external agent eval or NPU verification",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--source-repo", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.source_repo)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
