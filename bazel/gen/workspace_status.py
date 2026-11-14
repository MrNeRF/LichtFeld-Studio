# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bazel --workspace_status_command: git information for git_version.h.

Mirrors cmake/GenerateGitVersion.cmake. STABLE_* keys make dependent actions
rerun only when the values change.
"""

import os
import subprocess


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=30
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


commit = git("rev-parse", "--short", "HEAD")
tagged = git("describe", "--tags", "--dirty", "--always", "--match", "v*")

if not commit and os.environ.get("GITHUB_SHA"):
    commit = os.environ["GITHUB_SHA"][:7]
if not tagged:
    if os.environ.get("GITHUB_REF_TYPE") == "tag" and os.environ.get("GITHUB_REF_NAME"):
        tagged = os.environ["GITHUB_REF_NAME"]
    elif commit:
        tagged = commit

print(f"STABLE_GIT_COMMIT_HASH_SHORT {commit or 'unknown'}")
print(f"STABLE_GIT_TAGGED_VERSION {tagged or 'unknown'}")
