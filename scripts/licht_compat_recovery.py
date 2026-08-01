#!/usr/bin/env python3
"""Run the four named P8 Linux durability cells at the pinned authority.

Set LICHT_COMPAT_ASSUME_NO_USERNS=1 only to exercise the container skip policy.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from licht_compat_matrix import verify_frozen_content


AUTHORITY_SHA = "8ca8028e6214b1f424c373b24d479cd90ff2e918"
ASSUME_NO_USERNS_ENV = "LICHT_COMPAT_ASSUME_NO_USERNS"
REQUIRED = "required"
REQUIRES_USER_MOUNT_NAMESPACE = "required-where-capable:user-mount-namespace"
ENOSPC_NO_USERNS = "enospc:no-userns"
CELLS = (
    (
        "P8CompatibilityTest.AutosaveSigkillAtEveryBoundaryLeavesCompleteOldOrNewSidecar",
        REQUIRED,
    ),
    (
        "ProjectContainerWriter.ProcessKillCrashMatrixPublishesOnlyOldOrNew",
        REQUIRED,
    ),
    (
        "ProjectContainerWriter.RealEnospcUsesIsolatedTmpfs",
        REQUIRES_USER_MOUNT_NAMESPACE,
    ),
    (
        "ProjectContainerWriter.CompactionSigkillAtEveryBoundaryPublishesOnlyOldOrVerifiedNew",
        REQUIRED,
    ),
)


def _user_mount_namespaces_available(repository: Path) -> bool:
    override = os.environ.get(ASSUME_NO_USERNS_ENV)
    if override == "1":
        return False
    if override not in (None, "", "0"):
        raise ValueError(f"{ASSUME_NO_USERNS_ENV} must be 0 or 1")
    try:
        probe = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--mount", "true"],
            cwd=repository,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return False
    return probe.returncode == 0


def _environment_skip_reason(policy: str, repository: Path) -> str | None:
    if policy == REQUIRED:
        return None
    if policy == REQUIRES_USER_MOUNT_NAMESPACE:
        if not _user_mount_namespaces_available(repository):
            return ENOSPC_NO_USERNS
        return None
    raise AssertionError(f"unknown recovery-cell policy: {policy}")


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
        for cell, _policy in CELLS:
            print(f"DEFERRED {cell}: Windows recovery matrix is owned by P0d")
        return 2
    format_binary = repository / "build/tests/lichtfeld_format_tests"
    if not format_binary.is_file():
        print("compat-recovery FAIL: format test binary is not built", file=sys.stderr)
        return 1
    ran = 0
    environment_skips: list[str] = []
    for cell, policy in CELLS:
        try:
            skip_reason = _environment_skip_reason(policy, repository)
        except Exception as error:
            print(
                f"compat-recovery FAIL: {cell} environment probe: {error}",
                file=sys.stderr,
            )
            return 1
        if skip_reason is not None:
            environment_skips.append(skip_reason)
            print(
                f"ENV-SKIPPED {cell}: {skip_reason}; "
                "unshare user/mount namespace probe unavailable"
            )
            continue
        result = subprocess.run(
            [
                "nice",
                "-n",
                "15",
                str(format_binary),
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
        skipped = "[  SKIPPED ]" in output
        passed = f"[       OK ] {cell}" in output
        if result.returncode != 0 or skipped or "0 tests" in output or not passed:
            print(f"compat-recovery FAIL: {cell}", file=sys.stderr)
            return 1
        ran += 1
    environment_summary = str(len(environment_skips))
    if environment_skips:
        environment_summary += f"({','.join(environment_skips)})"
    print(
        f"compat-recovery PASS: provenance={AUTHORITY_SHA} head={head} "
        f"baseline_manifest={frozen['baseline_manifest_root']} "
        f"candidate_manifest={frozen['candidate_manifest_root']} "
        f"tagged_tree={frozen['tagged_tree']} "
        f"linux_cells={len(CELLS)} ran={ran} env_skipped={environment_summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
