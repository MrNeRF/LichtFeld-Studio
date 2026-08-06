#!/usr/bin/env bash
# Concurrency-aware build wrapper (see RULES.md "Build discipline").
# Usage: ./perf_campaign/build.sh <build-dir> [extra cmake --build args...]
# Registers itself as an active builder and caps -j by how many builds run:
#   1 builder -> -j12, 2 -> -j8, 3+ -> -j4   (30 GB RAM box; oomd threshold 50%)
set -euo pipefail
SLOTS=/tmp/lfs-build-slots; mkdir -p "$SLOTS"
# prune dead slots
for f in "$SLOTS"/*; do [ -e "$f" ] || continue; kill -0 "$(basename "$f")" 2>/dev/null || rm -f "$f"; done
echo 1 > "$SLOTS/$$"; trap 'rm -f "$SLOTS/$$"' EXIT
N=$(ls "$SLOTS" | wc -l)
if   [ "$N" -ge 3 ]; then J=4
elif [ "$N" -eq 2 ]; then J=8
else                      J=12; fi
BUILD_DIR=$1; shift || true
echo "[build.sh] active_builders=$N -> -j$J ($BUILD_DIR)"
exec cmake --build "$BUILD_DIR" -j "$J" "$@"
