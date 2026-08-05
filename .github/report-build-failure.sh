#!/usr/bin/env bash
#
# Publish the tail of a failed build log where it can be read without an
# authenticated session.
#
# Workflow logs require a signed-in GitHub session: the REST logs endpoint
# returns 403 for anonymous callers and the web view hides them behind a login.
# That makes a red CI run undiagnosable from outside, so this posts the relevant
# lines as a pull-request comment and as ::error:: annotations, both of which
# are served by the public API.
#
# Usage: report-build-failure.sh <label> <logfile>
#
# Never fails the job: reporting a failure must not replace the failure.

set -uo pipefail

LABEL="${1:-Build}"
LOGFILE="${2:-}"

if [ -z "$LOGFILE" ] || [ ! -f "$LOGFILE" ]; then
    echo "report-build-failure: no log at '${LOGFILE}'" >&2
    exit 0
fi

# Compiler diagnostics first; fall back to the tail when the failure produced
# none (out-of-memory kills and full disks usually do not).
excerpt=$(grep -E "error:|Error [0-9]|Killed|No space left|virtual memory exhausted|Segmentation fault" "$LOGFILE" \
    | head -n 40)
if [ -z "$excerpt" ]; then
    excerpt=$(tail -n 40 "$LOGFILE")
fi

# Annotations are visible via /check-runs/{id}/annotations without auth.
printf '%s\n' "$excerpt" | head -n 10 | while IFS= read -r line; do
    echo "::error::${line}"
done

disk=$(df -h / 2>/dev/null || echo "df unavailable")

# A pull-request comment is the durable copy. Requires the event to be a PR and
# a token with write access; skipped silently otherwise.
PR_NUMBER=$(jq -r '.pull_request.number // empty' "${GITHUB_EVENT_PATH:-/dev/null}" 2>/dev/null || true)
if [ -z "${PR_NUMBER}" ] || [ -z "${GITHUB_TOKEN:-}" ]; then
    echo "report-build-failure: not a PR event or no token; annotations only" >&2
    exit 0
fi

body=$(printf '**%s failed** on `%s` (%s)\n\n```\n%s\n```\n\n<details><summary>disk</summary>\n\n```\n%s\n```\n</details>\n' \
    "$LABEL" "${GITHUB_JOB:-job}" "${GITHUB_SHA:0:8}" "$excerpt" "$disk")

payload=$(jq -nc --arg b "$body" '{body: $b}') || exit 0

curl -sS -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    "${GITHUB_API_URL:-https://api.github.com}/repos/${GITHUB_REPOSITORY}/issues/${PR_NUMBER}/comments" \
    -d "$payload" >/dev/null 2>&1 || true

exit 0
