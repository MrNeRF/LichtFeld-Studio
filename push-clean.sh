#!/usr/bin/env bash
# Publish a verified rev to origin/lfs-elite: content-scrubbed, messages cleaned.
# Usage: ./push-clean.sh <verified-rev>   (run by the repo owner)
set -euo pipefail
REV=${1:?verified rev required}
REPO=/home/gauss/projects/LichtFeld-Studio
cd "$REPO"
git worktree remove -f /tmp/lfs-publish-wt 2>/dev/null || true
git branch -f lfs-elite-publish "$REV"
git worktree add /tmp/lfs-publish-wt lfs-elite-publish
cd /tmp/lfs-publish-wt
# internal campaign material is not published at all
for d in perf_campaign; do
  if [ -d "$d" ]; then git rm -rq "$d"; fi
done
for f in SPEED_VRAM_OPTIMIZATION_PLAN.md TENSOR_LIB_FINDINGS.md HANDOFF.md; do
  if [ -f "$f" ]; then git rm -q "$f"; fi
done
# drop campaign-related lines the campaign added to .gitignore
sed -i '/perf_campaign/d; /push-clean/d; /Phase 0 bench outputs/d' .gitignore
git add -A
git diff --cached --quiet || git commit -m "chore: strip internal development material"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --msg-filter \
  'sed -e "/^Co-Authored-By: Claude/d" -e "/^Claude-Session:/d"' \
  -- origin/master..lfs-elite-publish
cd "$REPO"
git worktree remove -f /tmp/lfs-publish-wt
git push --force-with-lease origin lfs-elite-publish:refs/heads/lfs-elite
echo "pushed $(git rev-parse --short lfs-elite-publish) -> origin/lfs-elite"
