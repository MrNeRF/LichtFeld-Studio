#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Report remaining I/O trap-class sites: unbounded setg, text-mode structured I/O.

This is a conservative line-based heuristic, not a C++ parser. It scans
src/io, src/sequencer, src/visualizer/project, and src/training. Use an
allowlist of exact path:line exceptions. Findings print as file:line
messages and the process exits non-zero when any remain.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALLOWLIST = PROJECT_ROOT / "tools" / "io_discipline_allowlist.txt"
DEFAULT_ROOTS = (
    PROJECT_ROOT / "src" / "io",
    PROJECT_ROOT / "src" / "sequencer",
    PROJECT_ROOT / "src" / "visualizer" / "project",
    PROJECT_ROOT / "src" / "training",
)
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".cc"}
SETG_CALL = re.compile(r"\bsetg\s*\(")
STREAM_CTOR = re.compile(r"\bstd::(ifstream|ofstream)\s+[A-Za-z_]\w*\s*\(")
OPEN_CALL = re.compile(r"\b(open_file_for_read|open_file_for_write)\s*\(")

ALLOWLIST_HEADER = """\
# Reviewed exceptions for tools/check_io_discipline.py.
# Entries are <repo-relative-path>:<exact stripped source line>.
# A finding is allowed only when the file's repo-relative path equals the
# entry path exactly and the flagged line, stripped, equals the entry's line
# text exactly. Suffix path matching and substring token matching are not used.
# Blank lines and # comments are ignored.
#
# Regenerated with: python3 tools/check_io_discipline.py --write-allowlist
"""
KIND_SECTION_HEADERS = {
    "setg": (
        "# --- setg: windowed or bounded get areas "
        "(MSVC stores get-area length as int)"
    ),
    "text-mode": (
        "# --- text-mode: deliberate line-oriented or /proc reads "
        "(not structured binary)"
    ),
}


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    text: str


_WRITE_ALLOWLIST_TO_DEFAULT = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="source root to scan (repeatable; defaults to io/sequencer/project/training)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help=(
            "exact <repo-relative-path>:<stripped line> entries; "
            "blank lines and # comments are ignored"
        ),
    )
    parser.add_argument(
        "--write-allowlist",
        nargs="?",
        const=_WRITE_ALLOWLIST_TO_DEFAULT,
        default=None,
        metavar="FILE",
        help=(
            "write current findings as exact <path>:<stripped line> entries "
            "(default destination: --allowlist; does not apply the existing allowlist)"
        ),
    )
    return parser.parse_args()


def load_allowlist(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    entries: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if ":" not in entry:
            raise ValueError(
                "allowlist entries must be "
                "<repo-relative-path>:<exact stripped source line>, "
                f"got {entry!r}"
            )
        file_part, line_text = entry.split(":", 1)
        file_part = file_part.strip()
        line_text = line_text.strip()
        if not file_part or not line_text:
            raise ValueError(
                "allowlist entries must be "
                "<repo-relative-path>:<exact stripped source line>, "
                f"got {entry!r}"
            )
        entries.add((file_part, line_text))
    return entries


def allowlist_key(path: Path, text: str) -> tuple[str, str]:
    return path.relative_to(PROJECT_ROOT).as_posix(), text.strip()


def is_allowed(path: Path, line: str, allowlist: set[tuple[str, str]]) -> bool:
    return allowlist_key(path, line) in allowlist


def code_without_line_comment(line: str) -> str:
    in_string = False
    escape = False
    quote = ""
    for index, char in enumerate(line):
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "/" and index + 1 < len(line) and line[index + 1] == "/":
            return line[:index]
    return line


def argument_list(lines: list[str], line_index: int, open_col: int) -> str | None:
    depth = 0
    chunks: list[str] = []
    started = False
    for index in range(line_index, min(line_index + 16, len(lines))):
        text = code_without_line_comment(lines[index])
        start = open_col if index == line_index else 0
        for column, char in enumerate(text[start:], start=start):
            if char == "(":
                depth += 1
                started = True
                continue
            if char == ")":
                depth -= 1
                if started and depth == 0:
                    return "".join(chunks)
                continue
            if started and depth > 0:
                chunks.append(char)
    return None


def scan_file(path: Path) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    findings: list[Finding] = []
    for line_number, line in enumerate(lines, start=1):
        code = code_without_line_comment(line)
        stripped = code.lstrip()
        if stripped.startswith("#"):
            continue
        for _match in SETG_CALL.finditer(code):
            findings.append(Finding(path, line_number, "setg", line.strip()))
        for pattern in (STREAM_CTOR, OPEN_CALL):
            for match in pattern.finditer(code):
                args = argument_list(lines, line_number - 1, match.end() - 1)
                if args is None or "binary" in args:
                    continue
                findings.append(
                    Finding(path, line_number, "text-mode", line.strip())
                )
    return findings


def collect_findings(roots: list[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for root in roots:
        if not root.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"scan root must be inside the project: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            findings.extend(scan_file(path))
    return findings


def display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def allowlist_entry(finding: Finding) -> str:
    relative_path, text = allowlist_key(finding.path, finding.text)
    return f"{relative_path}:{text}"


def write_allowlist(path: Path, findings: list[Finding]) -> int:
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[str]] = {}
    for finding in findings:
        key = allowlist_key(finding.path, finding.text)
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(finding.kind, []).append(allowlist_entry(finding))

    chunks = [ALLOWLIST_HEADER.rstrip(), ""]
    kinds = [kind for kind in ("setg", "text-mode") if kind in grouped]
    kinds.extend(kind for kind in grouped if kind not in {"setg", "text-mode"})
    for kind in kinds:
        header = KIND_SECTION_HEADERS.get(kind, f"# --- {kind}")
        chunks.append(header)
        chunks.extend(grouped[kind])
        chunks.append("")
    path.write_text("\n".join(chunks).rstrip() + "\n", encoding="utf-8")
    return len(seen)


def main() -> int:
    args = parse_args()
    roots = [
        (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        for root in (args.root or DEFAULT_ROOTS)
    ]
    findings = collect_findings(roots)

    if args.write_allowlist is not None:
        dest = (
            args.allowlist
            if args.write_allowlist is _WRITE_ALLOWLIST_TO_DEFAULT
            else Path(args.write_allowlist)
        )
        try:
            count = write_allowlist(dest, findings)
        except OSError as error:
            print(
                f"error: cannot write allowlist '{dest}': {error}",
                file=sys.stderr,
            )
            return 2
        print(f"wrote allowlist: {display_path(dest)}", file=sys.stderr)
        print(f"{count} entries", file=sys.stderr)
        return 0

    allowlist = load_allowlist(args.allowlist)
    remaining = [
        finding
        for finding in findings
        if not is_allowed(finding.path, finding.text, allowlist)
    ]

    if remaining:
        allowlist_display = display_path(args.allowlist)
        print(f"I/O discipline violations: {len(remaining)}")
        for finding in remaining:
            relative_path = finding.path.relative_to(PROJECT_ROOT).as_posix()
            print(f"{relative_path}:{finding.line}: [{finding.kind}] {finding.text}")
            print(f"  allowlist as: {relative_path}:{finding.text}")
        print(
            "To allowlist a reviewed exception, add "
            "`<repo-relative-path>:<exact stripped source line>` "
            f"to {allowlist_display} (or regenerate with --write-allowlist)."
        )
        return 1
    print("No I/O discipline violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
