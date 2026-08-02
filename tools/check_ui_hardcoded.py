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
PYTHON_GUI_ROOT = PROJECT_ROOT / "src" / "python" / "lfs_plugins"
SOURCE_SUFFIXES = {".cpp", ".hpp", ".h", ".py"}
RML_SUFFIXES = {".rml"}
STRING_LITERAL = re.compile(r'"((?:\\.|[^"\\])*)"')
RML_TEXT = re.compile(r">([^<>{}][^<>{}]*[A-Za-z][^<>{}]*)<")
RML_TEXT_ATTRIBUTE = re.compile(r'\b(?:title|placeholder|aria-label)\s*=\s*"((?:\\.|[^"\\])*)"')
UI_SINK = re.compile(
    r"\b(SetText|SetInnerRML|body_rml|\.title\s*=|\.label\s*=|bind_func\(|"
    r"\blabel\s*=|\.stage\s*=|\.error\s*=|\bstatus\s*=|_set_status\(|on_progress\(|"
    r"_set_scan_log_entry\(|addError\(|fail_start\(|std::unexpected\(|"
    r"set_text\(|set_inner_rml\(|message_dialog\()"
)
IGNORE_LINE = re.compile(r"\b(LOG_(?:TRACE|DEBUG|INFO|WARN|ERROR)|#include)\b")
TECHNICAL_LITERAL = re.compile(
    r"^(?:[A-Z][A-Z0-9_]{1,}|"
    r"(?:PLY|SOG|SPZ|USDZ?|RAD|COLMAP|HTML|CUDA|GPU|RML|LSP))$"
)
FORMAT_ONLY_LITERAL = re.compile(r"^\{[A-Za-z_][A-Za-z0-9_]*:[^{}]+\}$")
INTERPOLATION_ONLY_LITERAL = re.compile(r"^[^A-Za-z{}]*\{[^{}]+\}[^A-Za-z{}]*$")
STYLE_INTERPOLATION_LITERAL = re.compile(r"^\{[^{}]+\}(?:dp|px)$")
DYNAMIC_MODEL_FIELD_LITERAL = re.compile(r"^_?label_\{[^{}]+\}$")
LOCALIZATION_KEY = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
INTERNAL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
RML_ELEMENT_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)+$")
RML_BINDING = re.compile(r"^\{\{?[A-Za-z_][A-Za-z0-9_]*\}?\}$")
URL_LITERAL = re.compile(r"^(?:https?://|github:)[^\s]+$")
HTML_TAG_LITERAL = re.compile(r"^</?[A-Za-z][A-Za-z0-9-]*>$")


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    text: str
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        action="append",
        default=None,
        help="GUI source root to scan (repeatable; defaults to native GUI and Python GUI plugins)",
    )
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
    if not text or text in allowlist or LOCALIZATION_KEY.fullmatch(text):
        return False
    if (INTERNAL_IDENTIFIER.fullmatch(text) or RML_ELEMENT_ID.fullmatch(text)
            or RML_BINDING.fullmatch(text) or URL_LITERAL.fullmatch(text)
            or HTML_TAG_LITERAL.fullmatch(text)):
        return False
    # Numeric format specifications and composed status lines that already call
    # the localization API do not introduce user-facing hardcoded copy.
    if (FORMAT_ONLY_LITERAL.fullmatch(text) or INTERPOLATION_ONLY_LITERAL.fullmatch(text)
            or STYLE_INTERPOLATION_LITERAL.fullmatch(text)
            or DYNAMIC_MODEL_FIELD_LITERAL.fullmatch(text)):
        return False
    if any(pattern.fullmatch(text) for pattern in allow_patterns):
        return False
    if text.startswith(("@tr:", "@")) or (text.startswith("&") and text.endswith(";")):
        return False
    if re.fullmatch(r"(?:\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8}|\\x[0-9a-fA-F]{2})+", text):
        return False
    if re.fullmatch(r"(?:\d+(?:\.\d+)?(?:K|x|\s*fps|\s*FPS|p)?|\d+x\d+|\{:[^}]+\}%?)", text):
        return False
    if TECHNICAL_LITERAL.fullmatch(text):
        return False
    return any(character.isalpha() for character in text)


def scan_source(path: Path, allowlist: set[str], allow_patterns: list[re.Pattern[str]]) -> list[Finding]:
    findings: list[Finding] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if IGNORE_LINE.search(line) or not UI_SINK.search(line):
            continue
        for match in STRING_LITERAL.finditer(line):
            # Keep UTF-8 literals intact. Decoding the UTF-8 byte sequence with
            # ``unicode_escape`` turns symbols such as ° into ``Â°``, which then
            # looks like a textual candidate to the heuristic.
            text = match.group(1)
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
        for match in RML_TEXT_ATTRIBUTE.finditer(line):
            text = match.group(1).strip()
            if is_candidate(text, allowlist, allow_patterns):
                findings.append(Finding(path, line_number, text, "rml-attribute"))
    return findings


def main() -> int:
    args = parse_args()
    allowlist, allow_patterns = load_allowlist(args.allowlist)
    findings: list[Finding] = []
    roots = [
        (root if root.is_absolute() else PROJECT_ROOT / root).resolve()
        for root in (args.root or [GUI_ROOT, PYTHON_GUI_ROOT])
    ]
    for root in roots:
        if not root.is_relative_to(PROJECT_ROOT):
            raise ValueError(f"scan root must be inside the project: {root}")
        for path in sorted(root.rglob("*")):
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
