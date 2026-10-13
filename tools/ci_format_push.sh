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

if output=$(LC_ALL=C git push --porcelain origin "HEAD:refs/heads/$branch" 2>&1); then
    printf '%s\n' "$output"
    exit 0
fi
printf '%s\n' "$output" >&2

# Only Git's explicit non-fast-forward rejection can be treated as stale.
# A coincidental branch advance must not hide authentication/server failures.
while IFS=$'\t' read -r flag ref reason; do
    if [[ "$flag" == '!' && "$ref" == "HEAD:refs/heads/$branch" &&
          ( "$reason" == '[rejected] (fetch first)' || "$reason" == '[rejected] (non-fast-forward)' ) ]]; then
        current=$(remote_head)
        if [[ "$current" != "$expected" ]]; then
            echo "Branch advanced during push; skipping obsolete formatting commit."
            exit 0
        fi
    fi
done <<< "$output"
echo "Formatting push failed; no verified non-fast-forward rejection to skip." >&2
exit 1
