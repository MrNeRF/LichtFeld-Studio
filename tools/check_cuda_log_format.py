#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reject std::format placeholders in LOG_* calls compiled by NVCC.

In .cu and .cuh files the logger formats with snprintf, so a {} placeholder is
printed literally and its argument is dropped. Those files need printf
conversions (%d, %zu, %s, %.3f). Findings print as file:line messages and the
process exits non-zero when any remain.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = (PROJECT_ROOT / "src", PROJECT_ROOT / "tests")
CUDA_SUFFIXES = {".cu", ".cuh"}
LOG_CALL = re.compile(r'\bLOG_(?:TRACE|DEBUG|INFO|PERF|WARN|ERROR|CRITICAL)\s*\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
STRING_LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')
PLACEHOLDER = re.compile(r"\{[^{}]*\}")


def main() -> int:
    findings = []
    for root in SCAN_ROOTS:
        for path in sorted(root.rglob("*")):
            if path.suffix not in CUDA_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            for call in LOG_CALL.finditer(text):
                fmt = "".join(STRING_LITERAL.findall(call.group(1)))
                if PLACEHOLDER.search(fmt):
                    line = text.count("\n", 0, call.start()) + 1
                    findings.append(f"{path.relative_to(PROJECT_ROOT)}:{line}: "
                                    "{} placeholder in a CUDA LOG call; use printf conversions")
    for finding in findings:
        print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
