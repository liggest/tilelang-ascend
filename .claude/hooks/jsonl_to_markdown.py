#!/usr/bin/env python3
"""Convert Claude transcript JSONL files into human-readable Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Claude CLI transcript JSONL file as Markdown.",
    )
    parser.add_argument(
        "jsonl_path",
        help="Path to the transcript .jsonl file to convert.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path for the Markdown file (defaults to input name with .md).",
    )
    return parser.parse_args()

def default_output_path(src_path: Path) -> Path:
    if src_path.suffix == ".jsonl":
        return src_path.with_suffix(".md")
    return src_path.with_suffix(src_path.suffix + ".md")


def load_entries(path: Path) -> List[Tuple[int, Any]]:
    entries: List[Tuple[int, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Failed to parse JSON on line {idx}: {exc}") from exc
            entries.append((idx, payload))
    return entries


def build_markdown(entries: Sequence[Tuple[int, Any]], label: str) -> str:
    lines: List[str] = [f"# Transcript: {label}", ""]
    for index, (line_no, entry) in enumerate(entries, 1):
        lines.extend(format_entry(index, line_no, entry))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_entry(index: int, line_no: int, entry: Any) -> List[str]:
    entry_type = entry.get("type") if isinstance(entry, dict) else None
    if entry_type in {"assistant", "user"}:
        return format_message_entry(index, line_no, entry)
    if entry_type == "file-history-snapshot":
        return format_snapshot_entry(index, line_no, entry)
    if entry_type == "summary":
        return format_summary_entry(index, line_no, entry)
    return format_generic_entry(index, line_no, entry)


def format_message_entry(index: int, line_no: int, entry: dict) -> List[str]:
    message = entry.get("message") or {}
    role = message.get("role") or entry.get("type") or "unknown"
    timestamp = entry.get("timestamp") or "timestamp unavailable"
    header = f"## [{index}] {role} ({entry.get('type')}) — {timestamp}"
    lines: List[str] = [header]

    metadata_pairs = [
        ("Entry type", entry.get("type")),
        ("Source line", line_no),
        ("UUID", entry.get("uuid")),
        ("Parent UUID", entry.get("parentUuid")),
        ("Agent", entry.get("agentId")),
        ("Session", entry.get("sessionId")),
        ("User type", entry.get("userType")),
        ("Is sidechain", entry.get("isSidechain")),
        ("Working dir", entry.get("cwd")),
        ("Git branch", entry.get("gitBranch")),
        ("Version", entry.get("version")),
        ("Message ID", message.get("id")),
        ("Model", message.get("model")),
        ("Stop reason", message.get("stop_reason")),
        ("Stop sequence", message.get("stop_sequence")),
        ("Usage", message.get("usage")),
    ]
    lines.extend(format_metadata(metadata_pairs))

    lines.extend(format_message_content(message))

    if "toolUseResult" in entry:
        lines.append("### toolUseResult")
        lines.extend(render_code_block(entry["toolUseResult"]))
        lines.append("")

    return lines


def format_message_content(message: dict) -> List[str]:
    content = message.get("content")
    lines: List[str] = []
    if isinstance(content, str):
        lines.append("### Message")
        lines.extend(render_blockquote(content))
        lines.append("")
        return lines

    if isinstance(content, list):
        text_count = 0
        tool_count = 0
        result_count = 0
        for chunk in content:
            chunk_type = chunk.get("type")
            if chunk_type == "text":
                text_count += 1
                suffix = f" ({text_count})" if text_count > 1 else ""
                lines.append(f"### Message{suffix}")
                lines.extend(render_blockquote(chunk.get("text", "")))
                lines.append("")
            elif chunk_type == "tool_use":
                tool_count += 1
                lines.append(f"### Tool Call {tool_count}")
                lines.append(f"- Name: `{chunk.get('name', 'unknown')}`")
                if chunk.get("id"):
                    lines.append(f"- Call ID: `{chunk['id']}`")
                lines.append("")
                lines.extend(render_code_block(chunk.get("input")))
                lines.append("")
            elif chunk_type == "tool_result":
                result_count += 1
                label = f"### Tool Result {result_count}"
                if chunk.get("tool_use_id"):
                    label += f" ← `{chunk['tool_use_id']}`"
                if chunk.get("is_error"):
                    label += " (error)"
                lines.append(label)
                normalized = normalize_tool_result_content(chunk.get("content"))
                lines.extend(render_blockquote(normalized))
                lines.append("")
            else:
                lines.append(f"### Content ({chunk_type or 'unknown'})")
                lines.extend(render_code_block(chunk))
                lines.append("")
        if not content:
            lines.append("### Message")
            lines.extend(render_blockquote(""))
            lines.append("")
        return lines

    if content is None:
        return lines

    lines.append("### Message")
    lines.extend(render_code_block(content))
    lines.append("")
    return lines


def normalize_tool_result_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                parts.append(item["text"])
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    if isinstance(content, dict):
        return json.dumps(content, indent=2, ensure_ascii=False)
    return str(content)


def format_snapshot_entry(index: int, line_no: int, entry: dict) -> List[str]:
    snapshot = entry.get("snapshot") or {}
    timestamp = snapshot.get("timestamp") or "timestamp unavailable"
    header = f"## [{index}] file-history-snapshot — {timestamp}"
    lines: List[str] = [header]
    metadata_pairs = [
        ("Source line", line_no),
        ("Message ID", entry.get("messageId")),
        ("Snapshot message ID", snapshot.get("messageId")),
        ("Is snapshot update", entry.get("isSnapshotUpdate")),
    ]
    lines.extend(format_metadata(metadata_pairs))

    backups = snapshot.get("trackedFileBackups") or {}
    lines.append("### Tracked File Backups")
    if backups:
        for path_str in sorted(backups):
            info = backups[path_str]
            info_bits = ", ".join(f"{key}={info[key]}" for key in sorted(info))
            lines.append(f"- `{path_str}`: {info_bits}")
    else:
        lines.append("- (none)")
    lines.append("")
    return lines


def format_summary_entry(index: int, line_no: int, entry: dict) -> List[str]:
    header = f"## [{index}] summary — leaf {entry.get('leafUuid', 'unknown')}"
    lines: List[str] = [header]
    lines.extend(format_metadata([("Source line", line_no)]) )
    lines.append("### Summary")
    lines.extend(render_blockquote(entry.get("summary", "")))
    lines.append("")
    return lines


def format_generic_entry(index: int, line_no: int, entry: Any) -> List[str]:
    entry_type = entry.get("type") if isinstance(entry, dict) else type(entry).__name__
    header = f"## [{index}] {entry_type}"
    lines: List[str] = [header, f"- Source line: {line_no}", ""]
    lines.extend(render_code_block(entry))
    lines.append("")
    return lines


def format_metadata(pairs: Iterable[Tuple[str, Any]]) -> List[str]:
    rendered: List[str] = []
    for label, value in pairs:
        if value is None:
            continue
        rendered.append(f"- {label}: {format_metadata_value(value)}")
    if rendered:
        rendered.append("")
    return rendered


def format_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_blockquote(text: str) -> List[str]:
    lines: List[str] = []
    text = text or ""
    if not text:
        return ["> (empty)"]
    for raw_line in text.splitlines():
        lines.append("> " + raw_line if raw_line else ">")
    return lines


def render_code_block(payload: Any, language: str | None = None) -> List[str]:
    if isinstance(payload, str):
        body = payload
        lang = language or "text"
    else:
        try:
            body = json.dumps(payload, indent=2, ensure_ascii=False)
            lang = language or "json"
        except TypeError:
            body = str(payload)
            lang = language or "text"
    return [f"```{lang}", body, "```"]


def main() -> None:
    args = parse_args()
    src_path = Path(args.jsonl_path).expanduser().resolve()
    if not src_path.is_file():
        raise SystemExit(f"Input file not found: {src_path}")

    entries = load_entries(src_path)
    markdown = build_markdown(entries, src_path.name)

    if args.output:
        out_path = Path(args.output).expanduser().resolve()
    else:
        out_path = default_output_path(src_path)

    out_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
