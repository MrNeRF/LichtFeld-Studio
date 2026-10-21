# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply the pinned AMD PAL runtime patch before an opt-in SDK source build.

This patches source only. It does not modify an installed SDK or system DLL,
compile the SDK, or enable a workaround in LichtFeld.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve(strict=True)
    patches = Path(__file__).resolve().parents[2] / "cmake/patches/rocm"
    metadata = json.loads((patches / "pal-event-copy-order-12952.json").read_text())
    patch = patches / "pal-event-copy-order-12952.patch"
    if digest(patch) != metadata["runtime_patch_lf_sha256"]:
        raise RuntimeError("Patch digest differs from the pinned provenance")
    if (root / ".git").exists():
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if head != metadata["source_commit"]:
            raise RuntimeError("SDK source commit does not match the validated version")
    environment = dict(os.environ, GIT_CEILING_DIRECTORIES=str(root.parent))
    states = []
    for name, hashes in metadata["files"].items():
        actual = digest(root / name)
        states.append("before" if actual == hashes["before_sha256"] else
                      "after" if actual == hashes["after_sha256"] else "unknown")
    if all(state == "after" for state in states):
        print("Already applied: pinned PAL event/copy ordering patch")
        return
    if not all(state == "before" for state in states):
        raise RuntimeError("Modified, incompatible or partially patched SDK sources; refusing to apply")
    # A checkout may convert .patch to CRLF. Apply canonical bytes without
    # changing the tracked patch, and never inherit a parent checkout prefix.
    with tempfile.TemporaryDirectory(prefix="lfs-hip-sdk-patch-") as temporary:
        canonical_patch = Path(temporary) / patch.name
        canonical_patch.write_bytes(patch.read_bytes().replace(b"\r\n", b"\n"))
        subprocess.run(["git", "apply", "--check", str(canonical_patch)], cwd=root, env=environment, check=True)
        if args.check_only:
            print("PASS: pinned patch applies to exact source files; no files changed")
            return
        subprocess.run(["git", "apply", str(canonical_patch)], cwd=root, env=environment, check=True)
    for name, hashes in metadata["files"].items():
        if digest(root / name) != hashes["after_sha256"]:
            raise RuntimeError("Patched source hash mismatch; do not build this SDK")
    print("PASS: pinned PAL patch applied; SDK binary still requires a separate source build and tests")


if __name__ == "__main__":
    main()
