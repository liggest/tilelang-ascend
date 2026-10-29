#!/usr/bin/env python3
"""Verify the migrated TA documentation assets without importing SDKs or running examples."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


BASE_PIN = "0deb8b93da11b7f0becbcded732d9633b7ca4e34"
UPSTREAM_PIN = "463c234ce3087205c092e931207ca87b4cd5494c"
GUIDE = "docs/TileLang-Ascend Programming Guide.md"
PROFILING_FILES = {
    ".agents/skills/tilelang-op-profiling/" + suffix
    for suffix in (
        "SKILL.md",
        "evals/evals.json",
        "references/csv_fields_reference.md",
        "references/kernel-chain-measurement.md",
        "references/optimization_quickref.md",
        "scripts/perf_summary.py",
    )
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit(root: Path, source_repo: Path | None = None) -> dict:
    manifest = json.loads((root / "maint/ascend_a2a3/docs_manifest.json").read_text())
    errors = []
    entries = manifest["entries"]
    increment = bool(manifest.get("upstream_463c234c_documentation_increment"))
    expected_entries = 231 if increment else 225
    expected_skills = 22 if increment else 21
    counts = Counter(entry["source_path"].split("/")[0] for entry in entries)
    if counts != {"docs": 98, ".agents": 128 if increment else 122, ".opencode": 5}:
        errors.append(f"Incomplete source inventory: {dict(counts)}")
    if manifest["source_pin"] != BASE_PIN:
        errors.append("The original 0deb source baseline must not be replaced by the incremental pin")
    pins = Counter(entry.get("source_pin", manifest["source_pin"]) for entry in entries)
    expected_pins = {BASE_PIN: 224, UPSTREAM_PIN: 7} if increment else {BASE_PIN: 225}
    if pins != expected_pins:
        errors.append(f"Incomplete per-revision inventory: {dict(pins)}")
    new_paths = PROFILING_FILES | {GUIDE} if increment else set()
    for entry in entries:
        expected_pin = UPSTREAM_PIN if entry["source_path"] in new_paths else BASE_PIN
        if entry.get("source_pin", manifest["source_pin"]) != expected_pin:
            errors.append(f"Wrong per-entry source pin: {entry['source_path']}")
    sources = [entry["source_path"] for entry in entries]
    destinations = [entry["destination"] for entry in entries]
    if len(set(sources)) != expected_entries or len(set(destinations)) != expected_entries:
        errors.append("Source/destination inventory contains duplicates or omissions")
    skills = [entry for entry in entries if entry["source_path"].endswith("/SKILL.md")]
    if len(skills) != expected_skills:
        errors.append(f"Expected {expected_skills} complete source skills, found {len(skills)}")
    if increment and not PROFILING_FILES.issubset(sources):
        errors.append("The profiling increment must retain all six original assets")
    for path, digest in manifest["main_skill_sha256"].items():
        if not (root / path).is_file() or sha256((root / path).read_bytes()) != digest:
            errors.append(f"Protected main skill changed: {path}")
    for path, digest in manifest.get("protected_main_docs", {}).items():
        if not (root / path).is_file() or sha256((root / path).read_bytes()) != digest:
            errors.append(f"Protected main document/resource changed: {path}")
    python_files = 0
    shell_files = 0
    historical_versions = 0

    def original_record(record: dict, source_path: str, default_pin: str) -> None:
        archived_path = record.get("original_archive")
        if archived_path:
            archived = root / archived_path
            if not archived.is_file():
                errors.append(f"Missing full original: {archived_path}")
            else:
                packed = archived.read_bytes()
                if int.from_bytes(packed[4:8], "little") or sha256(gzip.decompress(packed)) != record["source_sha256"]:
                    errors.append(f"Full original bytes/mtime differ: {source_path}")
                if record.get("archive_sha256") and sha256(packed) != record["archive_sha256"]:
                    errors.append(f"Full original archive hash differs: {source_path}")
        if source_repo is not None:
            pin = record.get("source_pin", default_pin)
            try:
                original = subprocess.check_output(
                    ["git", "-c", "protocol.allow=never", "show", f"{pin}:{source_path}"],
                    cwd=source_repo,
                    env=dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0"),
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                errors.append(f"Fixed local source object unavailable: {pin}:{source_path}")
            else:
                if sha256(original) != record["source_sha256"]:
                    errors.append(f"Source pin hash differs: {pin}:{source_path}")

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
        original_record(entry, entry["source_path"], manifest["source_pin"])
        for previous in entry.get("source_history", []):
            historical_versions += 1
            if previous.get("source_path", entry["source_path"]) != entry["source_path"] or previous.get("source_pin") != BASE_PIN:
                errors.append(f"Historical source identity differs: {entry['source_path']}")
            original_record(previous, entry["source_path"], BASE_PIN)
        if path.suffix == ".py":
            ast.parse(content, filename=str(path))
            python_files += 1
        if path.suffix == ".sh":
            checked = subprocess.run(["bash", "-n", path], capture_output=True, text=True, check=False)
            if checked.returncode:
                errors.append(f"Shell syntax: {path}: {checked.stderr}")
            shell_files += 1
    if increment:
        guide = next((entry for entry in entries if entry["source_path"] == GUIDE), {})
        if not any(
            previous.get("source_pin") == BASE_PIN and previous.get("original_archive") for previous in guide.get("source_history", [])
        ):
            errors.append("The int4 guide increment must retain the original 0deb guide archive and provenance")
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
        "source_counts_by_pin": dict(pins),
        "historical_source_versions_checked": historical_versions,
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
