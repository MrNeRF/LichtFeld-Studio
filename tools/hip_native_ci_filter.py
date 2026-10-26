# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Select the isolated HIP compile job without changing other workflow jobs."""
import argparse
from pathlib import Path
import re
import subprocess


PREFIXES = ("cmake/hip-native/", "src/core/tensor/backend/hip/", "tests/hip/")
FILES = {
    "cmake/HipArchitectures.cmake", "cmake/HipOffloadCompression.cmake",
    ".github/workflows/ubuntu.yml", "tools/hip_native_ci_filter.py",
    "tests/python/test_hip_native_ci_filter.py",
    "tests/python/test_hip_runtime_configuration.py",
    "tests/python/test_hip_pointwise_generation.py",
    "src/core/include/core/detail/pointwise_ops.def",
    "src/core/include/core/detail/pointwise_lowering.hpp",
    "src/core/include/core/detail/tensor_functors.hpp",
}


def needs_compile(repo, event, base, head):
    if event == "workflow_dispatch":
        return True
    if event not in ("pull_request", "push"):
        raise ValueError("Unsupported HIP compile event")
    if event == "push" and base and not base.strip("0"):
        return True  # A new branch has no previous tree to compare.
    for revision in (base, head):
        if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
            raise ValueError("Expected full base and head commit hashes")
    if event == "pull_request":
        base = subprocess.check_output(
            ["git", "merge-base", base, head], cwd=repo, text=True).strip()
    # Disable rename detection so both the old and new paths are inspected.
    paths = subprocess.check_output(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head, "--"], cwd=repo)
    return any(path.decode("utf-8", errors="surrogateescape") in FILES
               or path.startswith(tuple(prefix.encode("utf-8") for prefix in PREFIXES))
               for path in paths.split(b"\0") if path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    changed = needs_compile(args.repo, args.event, args.base, args.head)
    with args.output.open("a", encoding="utf-8") as stream:
        stream.write(f"changed={'true' if changed else 'false'}\n")
    print("HIP native compile: " + ("required" if changed else "no relevant changes"))


if __name__ == "__main__":
    main()
