#!/usr/bin/env python3
"""Run the four named P8 Linux durability cells at the pinned authority."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from licht_compat_matrix import verify_frozen_content


AUTHORITY_SHA = "8ca8028e6214b1f424c373b24d479cd90ff2e918"
CELLS = (
    "ProjectDocumentTest.AutosaveSigkillAtEveryBoundaryLeavesCompleteOldOrNewSidecar",
    "ProjectContainerWriter.ProcessKillCrashMatrixPublishesOnlyOldOrNew",
    "ProjectContainerWriter.RealEnospcUsesIsolatedTmpfs",
    "ProjectContainerWriter.CompactionSigkillAtEveryBoundaryPublishesOnlyOldOrVerifiedNew",
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        frozen = verify_frozen_content(repository, run_cpp_proof=True)
    except Exception as error:
        print(
            f"compat-recovery FAIL: frozen-content gate: {error}",
            file=sys.stderr,
        )
        return 1
    if os.name == "nt":
        for cell in CELLS:
            print(f"DEFERRED {cell}: Windows recovery matrix is owned by P0d")
        return 2
    binary = repository / "build/tests/lichtfeld_tests"
    if not binary.is_file():
        print("compat-recovery FAIL: test binary is not built", file=sys.stderr)
        return 1
    passed = 0
    for cell in CELLS:
        result = subprocess.run(
            [
                "nice",
                "-n",
                "15",
                str(binary),
                "--gtest_color=no",
                f"--gtest_filter={cell}",
            ],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        print(output, end="" if output.endswith("\n") else "\n")
        if result.returncode != 0 or "[  SKIPPED ]" in output or "0 tests" in output:
            print(f"compat-recovery FAIL: {cell}", file=sys.stderr)
            return 1
        passed += 1
    print(
        f"compat-recovery PASS: provenance={AUTHORITY_SHA} head={head} "
        f"manifest={frozen['manifest_root']} tagged_tree={frozen['tagged_tree']} "
        f"linux_cells={passed} skipped=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
