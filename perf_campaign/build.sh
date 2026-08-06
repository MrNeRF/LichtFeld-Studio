#!/usr/bin/env bash
# Concurrency-aware build wrapper (RULES.md "Build discipline").
#   ./perf_campaign/build.sh <build-dir> [args...]     — build (waits for one of 3 slots)
#   ./perf_campaign/build.sh --configure <src-dir>     — configure preset (also slot-gated)
# Slot count hard-capped at 3 concurrent builds machine-wide; -j per maintainer rule:
#   3 active -> -j4, 2 -> -j8, 1 -> -j12.
set -euo pipefail
SLOTDIR=/tmp/lfs-build-slots; mkdir -p "$SLOTDIR"
acquire() {
  while :; do
    for s in 1 2 3; do
      exec {fd}>"$SLOTDIR/slot$s" || continue
      if flock -n "$fd"; then echo "$fd"; return; fi
      exec {fd}>&-
    done
    sleep 15
  done
}
FD=$(acquire)
active() { c=0; for s in 1 2 3; do flock -n "$SLOTDIR/slot$s" -c true 2>/dev/null || c=$((c+1)); done; echo $c; }
N=$(active); if [ "$N" -ge 3 ]; then J=4; elif [ "$N" -eq 2 ]; then J=8; else J=12; fi
if [ "${1:-}" = "--configure" ]; then
  cd "$2"; echo "[build.sh] configure ($N active)"; exec cmake --preset build -DLFS_CUDA_COMPILER_CACHE=ccache -DENABLE_COMPILER_CACHE=ccache
fi
BUILD_DIR=$1; shift || true
echo "[build.sh] active=$N -> -j$J ($BUILD_DIR)"
exec cmake --build "$BUILD_DIR" -j "$J" "$@"
