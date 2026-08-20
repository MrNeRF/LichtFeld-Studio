#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Report remaining I/O trap-class sites: unbounded setg, text-mode structured I/O.

This is a conservative line-based heuristic, not a C++ parser. It scans
src/io, src/sequencer, src/visualizer/project, and src/training. Use an
allowlist for reviewed exceptions. Findings print as file:line messages and
the process exits non-zero when any remain.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
LINE_COMMENT = re.compile(r"//.*$")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    kind: str
    text: str


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
        default=PROJECT_ROOT / "tools" / "io_discipline_allowlist.txt",
        help="file:token entries; blank lines and # comments are ignored",
    )
    return parser.parse_args()


def load_allowlist(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    entries: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if ":" not in entry:
            raise ValueError(f"allowlist entries must be file:token, got {entry!r}")
        file_part, token = entry.split(":", 1)
        entries.append((file_part, token))
    return entries


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


def is_allowed(path: Path, line: str, allowlist: list[tuple[str, str]]) -> bool:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    for file_part, token in allowlist:
        if relative != file_part and not relative.endswith("/" + file_part):
            continue
        if token in line:
            return True
    return False


def scan_file(path: Path, allowlist: list[tuple[str, str]]) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    findings: list[Finding] = []
    for line_number, line in enumerate(lines, start=1):
        code = code_without_line_comment(line)
        stripped = code.lstrip()
        if stripped.startswith("#"):
            continue
        for match in SETG_CALL.finditer(code):
            if is_allowed(path, line, allowlist):
                continue
            findings.append(Finding(path, line_number, "setg", line.strip()))
        for pattern in (STREAM_CTOR, OPEN_CALL):
            for match in pattern.finditer(code):
                args = argument_list(lines, line_number - 1, match.end() - 1)
                if args is None or "binary" in args:
                    continue
                if is_allowed(path, line, allowlist):
                    continue
                findings.append(
                    Finding(path, line_number, "text-mode", line.strip())
                )
    return findings


def main() -> int:
    args = parse_args()
    allowlist = load_allowlist(args.allowlist)
    findings: list[Finding] = []
    roots = [
        (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        for root in (args.root or DEFAULT_ROOTS)
    ]
    for root in roots:
        if not root.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"scan root must be inside the project: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            findings.extend(scan_file(path, allowlist))

    if findings:
        print(f"I/O discipline violations: {len(findings)}")
        for finding in findings:
            relative_path = finding.path.relative_to(PROJECT_ROOT).as_posix()
            print(f"{relative_path}:{finding.line}: [{finding.kind}] {finding.text}")
        return 1
    print("No I/O discipline violations found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
