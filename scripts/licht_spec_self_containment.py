#!/usr/bin/env python3
"""Machine gate for the published 1.0 spec and outsider fixture bundle."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys


AUTHORITY_SHA = "8ca8028e6214b1f424c373b24d479cd90ff2e918"
FORBIDDEN_REFERENCES = (
    re.compile(r"PROJECT_FORMAT_PLAN(?:\.md)?"),
    re.compile(r"docs/licht_ownership_matrix\.md"),
    re.compile(r"(?:^|[\s`(])(?:src|tests|tools)/"),
    re.compile(r"`licht_format` branch"),
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    spec_path = repository / "docs/licht_format_spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    violations: list[str] = []
    for line_number, line in enumerate(spec.splitlines(), 1):
        for pattern in FORBIDDEN_REFERENCES:
            if pattern.search(line):
                violations.append(
                    f"{spec_path.relative_to(repository)}:{line_number}: {line.strip()}"
                )
    if violations:
        print(
            "spec-self-containment FAIL: non-published references:\n  "
            + "\n  ".join(violations),
            file=sys.stderr,
        )
        return 1

    required_text = (
        "self-contained on-disk contract",
        "AUDIT-ONLY accelerator",
        "never semantic authority",
        "| 112 | 8 | u64 | `preview_offset` |",
        "| 120 | 4 | u32 | `preview_bytes` |",
        "| 124 | 4 | u32 | `preview_format` |",
        "There are no unresolved byte-grammar questions in version 1.0",
    )
    missing = [phrase for phrase in required_text if phrase not in spec]
    if missing:
        print(
            "spec-self-containment FAIL: missing normative text: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    release = repository / "tests/fixtures/licht/release_corpus"
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("authority_sha") != AUTHORITY_SHA:
        print("spec-self-containment FAIL: release authority SHA mismatch", file=sys.stderr)
        return 1
    rows = {row["path"]: row for row in manifest.get("fixtures", [])}
    preview = release / "foreign-preview.licht"
    if "foreign-preview.licht" not in rows or not preview.is_file():
        print("spec-self-containment FAIL: foreign-preview corpus cell missing", file=sys.stderr)
        return 1

    # The compatibility job supplies the outsider acceptance proof: the frozen
    # package is import-isolated, the live package and byte-table verifier use
    # only the published fixture bytes, and the production C++ reader consumes
    # those same bytes. No plan or source-document input is passed to a parser.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/licht_compat_matrix.py",
        ],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            "spec-self-containment FAIL: outsider acceptance failed:\n"
            + (result.stdout + result.stderr).strip(),
            file=sys.stderr,
        )
        return 1
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(
        "spec-self-containment PASS: references=0 preview_audit_only=1 "
        f"release_rows={len(rows)} provenance={AUTHORITY_SHA} head={head}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
