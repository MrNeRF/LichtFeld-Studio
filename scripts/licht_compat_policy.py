#!/usr/bin/env python3
"""Reject undeclared writer-compatibility changes without matrix-test updates."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence


RISKY = re.compile(
    r"\b(?:min_safe_writer(?:_version)?|required_(?:reader|writer)_capabilities)\b"
)
WRITER_PATHS = (
    "src/io/project/project_writer.cpp",
    "src/io/project/project_document.cpp",
    "src/io/include/io/project_container.hpp",
)
REQUIRED_SYMBOLS = (
    "V10CorpusAndSyntheticNewerClassificationRefusePayloadAccess",
    "ProhibitedOldWriterMatrixHasNoWriteEffectForEveryProducer",
    "OpaqueAndRetainedJsonSurviveSafeAppendAndCompaction",
)


def _git_diff(repository: Path, base: str) -> str:
    paths = (*WRITER_PATHS, "tests/test_p8_compatibility.cpp",
             "scripts/licht_compat_matrix.py")
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--unified=0", base, "--", *paths],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if untracked.returncode != 0:
        raise RuntimeError(untracked.stderr.strip() or "git ls-files failed")
    diff = result.stdout
    for relative in untracked.stdout.splitlines():
        content = (repository / relative).read_text(encoding="utf-8")
        diff += (
            f"diff --git a/{relative} b/{relative}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{relative}\n"
            + "".join(f"+{line}\n" for line in content.splitlines())
        )
    return diff


def check_diff(diff: str) -> tuple[bool, str]:
    current_path = ""
    risky_lines: list[str] = []
    test_delta = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_path = line[6:]
            continue
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        if current_path in WRITER_PATHS and RISKY.search(line[1:]):
            risky_lines.append(f"{current_path}: {line}")
        if current_path in {"tests/test_p8_compatibility.cpp", "scripts/licht_compat_matrix.py"}:
            test_delta += line[1:] + "\n"
    if not risky_lines:
        return True, "no writer compatibility declaration changed"
    if not any(symbol in test_delta for symbol in REQUIRED_SYMBOLS):
        return (
            False,
            "writer compatibility declaration changed without a matching P8 matrix-symbol "
            "change:\n  " + "\n  ".join(risky_lines),
        )
    return (
        True,
        f"{len(risky_lines)} declaration line(s) paired with a P8 matrix-symbol change",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD", help="git diff base (default: HEAD)")
    parser.add_argument("--diff-file", type=Path, help="check a saved unified diff")
    args = parser.parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    try:
        diff = (
            args.diff_file.read_text(encoding="utf-8")
            if args.diff_file
            else _git_diff(repository, args.base)
        )
        safe, detail = check_diff(diff)
    except (OSError, RuntimeError) as error:
        print(f"compat-policy ERROR: {error}", file=sys.stderr)
        return 2
    print(f"compat-policy {'PASS' if safe else 'FAIL'}: {detail}")
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
