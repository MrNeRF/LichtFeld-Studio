#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject LOG_* placeholders that the file's logger does not format.

In .cu and .cuh files the logger formats with snprintf, so a {} placeholder is
printed literally and its argument is dropped. Those files need printf
conversions (%d, %zu, %s, %.3f). Everywhere else the logger uses std::format,
where a printf conversion is printed literally. Findings print as file:line
messages and the process exits non-zero when any remain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "tests")
CUDA_SUFFIXES = {".cu", ".cuh"}
FORMAT_SUFFIXES = {".cpp", ".hpp", ".h", ".cc", ".cxx", ".mm"}
LOG_CALL = re.compile(r'\bLOG_(?:TRACE|DEBUG|INFO|PERF|WARN|ERROR|CRITICAL)\s*\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
PLACEHOLDER = re.compile(r"\{[^{}]*\}")
PRINTF_CONVERSION = re.compile(r"(?<!%)%[-+#0]*\d*(?:\.\d+)?(?:hh|h|ll|l|z|j|t|L)?[diouxXfFeEgGcsp]")


def main() -> int:
    findings = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*")):
            cuda = path.suffix in CUDA_SUFFIXES
            if not cuda and path.suffix not in FORMAT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for call in LOG_CALL.finditer(text):
                fmt = "".join(STRING_LITERAL.findall(call.group(1)))
                line = text.count("\n", 0, call.start()) + 1
                if cuda and PLACEHOLDER.search(fmt):
                    findings.append(f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                                    "{} placeholder in a CUDA LOG call; use printf conversions")
                elif not cuda and PRINTF_CONVERSION.search(fmt.replace("%%", "")):
                    findings.append(f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                                    "printf conversion in a std::format LOG call; use {} placeholders")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
