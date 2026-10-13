#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
# Publish formatting only while the branch still points at the triggering commit.
set -euo pipefail

branch=${1:?expected branch name}
expected=${2:?expected triggering commit}

remote_head() {
    git fetch --no-tags origin "refs/heads/$branch"
    git rev-parse FETCH_HEAD
}

current=$(remote_head)
if [[ "$current" != "$expected" ]]; then
    echo "Branch advanced; skipping obsolete formatting commit."
    exit 0
fi

if git push origin "HEAD:refs/heads/$branch"; then
    exit 0
fi

# Another commit can land between the fetch and push. Never force-push or
# replay stale formatting over it. Authentication and other failures stay red.
current=$(remote_head)
if [[ "$current" != "$expected" ]]; then
    echo "Branch advanced during push; skipping obsolete formatting commit."
    exit 0
fi
echo "Formatting push failed while the branch was unchanged." >&2
exit 1
