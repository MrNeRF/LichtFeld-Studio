#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Report likely user-facing strings hardcoded in the GUI source tree.

This is a conservative heuristic, not a C++ parser. It scans only common UI
sinks and RML text nodes, so logs, protocol data, identifiers, and ordinary
implementation strings are not reported by default. Use an allowlist for
reviewed exceptions and enable --fail-on-candidates once the baseline is clean.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUI_ROOT = PROJECT_ROOT / "src" / "visualizer" / "gui"
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h"}
RML_SUFFIXES = {".rml"}
STRING_LITERAL = re.compile(r'"((?:\\.|[^"\\])*)"')
RML_TEXT = re.compile(r">([^<>{}][^<>{}]*[A-Za-z][^<>{}]*)<")
UI_SINK = re.compile(
    r"\b(SetText|SetInnerRML|body_rml|\.title\s*=|\.label\s*=|"
    r"\.stage\s*=|\.error\s*=|\bstatus\s*=|addError\(|fail_start\(|std::unexpected\()"
)
IGNORE_LINE = re.compile(r"\b(LOG_(?:TRACE|DEBUG|INFO|WARN|ERROR)|#include)\b")
TECHNICAL_LITERAL = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_./:-]*|[A-Z0-9_]{2,}|"
    r"(?:PLY|SOG|SPZ|USDZ?|RAD|COLMAP|HTML|CUDA|GPU|RML|LSP))$"
)


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    text: str
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=GUI_ROOT, help="GUI source root to scan")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=PROJECT_ROOT / "tools" / "ui_hardcoded_allowlist.txt",
        help="one exact text per line; blank lines and # comments are ignored",
    )
    parser.add_argument("--fail-on-candidates", action="store_true", help="exit 1 if findings remain")
    return parser.parse_args()


def load_allowlist(path: Path) -> tuple[set[str], list[re.Pattern[str]]]:
    if not path.exists():
        return set(), []
    exact: set[str] = set()
    patterns: list[re.Pattern[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if entry.startswith("re:"):
            patterns.append(re.compile(entry.removeprefix("re:")))
        else:
            exact.add(entry)
    return exact, patterns


def is_candidate(text: str, allowlist: set[str], allow_patterns: list[re.Pattern[str]]) -> bool:
    text = text.strip()
    if not text or text in allowlist or "LOC(" in text:
        return False
    if any(pattern.fullmatch(text) for pattern in allow_patterns):
        return False
    if text.startswith(("@tr:", "@")) or (text.startswith("&") and text.endswith(";")):
        return False
    if re.fullmatch(r"(?:\d+(?:\.\d+)?(?:x|\s*fps|\s*FPS|p)?|\d+x\d+|\{:[^}]+\}%?)", text):
        return False
    if TECHNICAL_LITERAL.fullmatch(text):
        return False
    return any(character.isalpha() for character in text)


def scan_source(path: Path, allowlist: set[str], allow_patterns: list[re.Pattern[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if IGNORE_LINE.search(line) or not UI_SINK.search(line) or "LOC(" in line:
            continue
        for match in STRING_LITERAL.finditer(line):
            text = bytes(match.group(1), "utf-8").decode("unicode_escape")
            if is_candidate(text, allowlist, allow_patterns):
                findings.append(Finding(path, line_number, text, "source"))
    return findings


def scan_rml(path: Path, allowlist: set[str], allow_patterns: list[re.Pattern[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for match in RML_TEXT.finditer(line):
            text = match.group(1).strip()
            if is_candidate(text, allowlist, allow_patterns):
                findings.append(Finding(path, line_number, text, "rml"))
    return findings


def main() -> int:
    args = parse_args()
    allowlist, allow_patterns = load_allowlist(args.allowlist)
    findings: list[Finding] = []
    for path in sorted(args.root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in SOURCE_SUFFIXES:
            findings.extend(scan_source(path, allowlist, allow_patterns))
        elif path.suffix in RML_SUFFIXES:
            findings.extend(scan_rml(path, allowlist, allow_patterns))

    if findings:
        print(f"Likely hardcoded UI strings: {len(findings)}")
        for finding in findings:
            relative_path = finding.path.relative_to(PROJECT_ROOT).as_posix()
            print(f"{relative_path}:{finding.line}: [{finding.kind}] {finding.text}")
    else:
        print("No likely hardcoded UI strings found.")
    return 1 if args.fail_on_candidates and findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
