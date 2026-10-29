#!/usr/bin/env python3
"""Explicit user-approved CI scope exclusions, separate from test migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from path_layout import adopted_layout, canonical_path, resolve_path

# Keep the original manifest registration and pending status. This one source
# requires TA's Ascend Carver strategy, which the user deferred for this stage.
DEFERRED_SOURCES = {
    "examples/autotune/example_gemm_carver.py": (
        "User-approved deferred scope: TA Ascend Carver hint policy is not migrated; "
        "preserve the original asset, do not substitute static configs or count it passed."
    ),
}


def deferred_reason(repo_root: Path, path: str) -> str | None:
    if not adopted_layout(repo_root):
        return None
    canonical = canonical_path(path)
    if path != resolve_path(repo_root, canonical):
        return None
    return DEFERRED_SOURCES.get(canonical)


def deferred_cases(repo_root: Path, dirs: list[str] | None = None, experiment_dirs: list[str] | None = None) -> list[dict[str, str]]:
    if not adopted_layout(repo_root):
        return []
    full = dirs is None and experiment_dirs is None
    records = []
    for source, reason in DEFERRED_SOURCES.items():
        if not full and source.split("/")[1] not in (dirs or []):
            continue
        path = resolve_path(repo_root, source)
        if (repo_root / path).is_file():
            records.append({"path": path, "source": source, "reason": reason, "status": "deferred"})
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dirs", nargs="*", default=None)
    parser.add_argument("--experiment-dirs", nargs="*", default=None)
    parser.add_argument("--format", choices=["json", "tsv"], default="json")
    args = parser.parse_args()
    records = deferred_cases(args.repo_root.resolve(), args.dirs, args.experiment_dirs)
    if args.format == "tsv":
        for record in records:
            print(f"{record['path']}\t{record['reason']}")
    else:
        print(json.dumps({"deferred_count": len(records), "passed_count": 0, "cases": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
