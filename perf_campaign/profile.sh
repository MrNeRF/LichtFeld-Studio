#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Repeatable GPU profiling harness (nsys / ncu) for the perf campaign.
#
# Subcommands:
#   ./perf_campaign/profile.sh timeline <label>
#       nsys timeline of a steady-state training slice. The trainer is built
#       with env-gated hooks (see src/training/trainer.cpp StepProfilingHooks):
#       LFS_PROFILE_START_ITER / LFS_PROFILE_STOP_ITER call cudaProfilerStart/
#       Stop, and nsys --capture-range=cudaProfilerApi records exactly the
#       slice [START, STOP) — default iters 200..500 (300 steady iterations
#       after warmup). LFS_NVTX=1 wraps each iteration in an NVTX range
#       "train_step:<iter>" for per-iteration attribution.
#       Output: perf_campaign/profiles/<label>/timeline.nsys-rep (+ meta.json)
#
#   ./perf_campaign/profile.sh kernels <label>
#       Post-process the label's .nsys-rep: nsys stats CSVs (kernel summary,
#       memop time/size sums, gpu trace), launch-gap analysis from the
#       exported sqlite (perf_campaign/launch_gaps.py), and a compact
#       summary.md. CSVs + summary are commit-able; raw .nsys-rep/.sqlite are
#       gitignored.
#
#   ./perf_campaign/profile.sh ncu <label> <kernel-regex>
#       Per-kernel deep metrics (occupancy, registers, local-mem traffic,
#       scheduler/warp stalls, pipe utilization). Hardware counters are
#       admin-locked on this machine (RmProfilingAdminOnly=1), so this
#       runs `sudo ncu` if passwordless sudo is available; otherwise it
#       prints the exact command for the maintainer and exits 3.
#
# Env:
#   LFS_BENCH_DATASET   dataset dir (default /home/gauss/data/360_v2/bonsai;
#                       bicycle = /home/gauss/data/360_v2/bicycle)
#   LFS_PROF_START      first profiled iteration        (default 200)
#   LFS_PROF_STOP       one-past-last profiled iter     (default 500)
#   LFS_PROF_ITERS      total training iterations       (default STOP+20)
#   LFS_PROF_BUILD_DIR  build dir                       (default build/tests if present, else build)
#
# GPU measurement runs hold the machine-wide bench lock (flock /tmp/lfs-bench.lock).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILES_ROOT="$ROOT/perf_campaign/profiles"
BENCH_LOCK=/tmp/lfs-bench.lock

if [[ -n "${LFS_PROF_BUILD_DIR:-}" ]]; then
  BUILD_DIR="$LFS_PROF_BUILD_DIR"
elif [[ -d "$ROOT/build/tests" && -f "$ROOT/build/tests/build.ninja" ]]; then
  BUILD_DIR="$ROOT/build/tests"
else
  BUILD_DIR="$ROOT/build"
fi
BIN="$BUILD_DIR/LichtFeld-Studio"
[[ -x "$BIN" ]] || BIN="$BUILD_DIR/bin/LichtFeld-Studio"

DATASET="${LFS_BENCH_DATASET:-/home/gauss/data/360_v2/bonsai}"
START="${LFS_PROF_START:-200}"
STOP="${LFS_PROF_STOP:-500}"
ITERS="${LFS_PROF_ITERS:-$((STOP + 20))}"
MAX_CAP="${LFS_BENCH_MAX_CAP:-500000}"
STRATEGY="${LFS_BENCH_STRATEGY:-mrnf}"
IMAGES="${LFS_BENCH_IMAGES:-images_4}"

usage() { sed -n '5,45p' "$0"; exit 2; }

CMD="${1:-}"; shift || true
case "$CMD" in timeline|kernels|ncu) ;; *) usage ;; esac
LABEL="${1:-}"; shift || true
[[ -n "$LABEL" ]] || usage
DIR="$PROFILES_ROOT/$LABEL"

train_env() {
  # Env for the profiled training run: NVTX per-iteration ranges + the
  # cudaProfilerStart/Stop slice hooks.
  echo "LFS_NVTX=1 LFS_PROFILE_START_ITER=$START LFS_PROFILE_STOP_ITER=$STOP"
}

train_args() {
  echo "-d $DATASET -o $DIR/train_out --images $IMAGES --headless --iter $ITERS --max-cap $MAX_CAP --strategy $STRATEGY"
}

write_meta() {
  cat > "$DIR/meta.json" <<EOF
{
  "label": "$LABEL",
  "commit": "$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)",
  "branch": "$(git -C "$ROOT" branch --show-current 2>/dev/null || echo detached)",
  "dataset": "$DATASET",
  "images": "$IMAGES",
  "strategy": "$STRATEGY",
  "max_cap": $MAX_CAP,
  "iters_total": $ITERS,
  "profiled_slice": [$START, $STOP],
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)",
  "date_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tool": "$1"
}
EOF
}

case "$CMD" in
  timeline)
    [[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN (build first: ./perf_campaign/build.sh $BUILD_DIR)" >&2; exit 1; }
    mkdir -p "$DIR"
    echo "[timeline] label=$LABEL dataset=$DATASET slice=[$START,$STOP) iters=$ITERS"
    echo "[timeline] acquiring bench lock ($BENCH_LOCK)..."
    # shellcheck disable=SC2046
    flock "$BENCH_LOCK" \
      env $(train_env) \
      nsys profile \
        --trace=cuda,nvtx \
        --sample=none --cpuctxsw=none \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        --force-overwrite=true \
        --output "$DIR/timeline" \
        "$BIN" $(train_args) > "$DIR/train.log" 2>&1 || {
          echo "ERROR: nsys run failed; tail of log:" >&2; tail -30 "$DIR/train.log" >&2; exit 1; }
    write_meta nsys
    rm -rf "$DIR/train_out"
    echo "[timeline] wrote $DIR/timeline.nsys-rep"
    ;;

  kernels)
    REP="$DIR/timeline.nsys-rep"
    [[ -f "$REP" ]] || { echo "ERROR: $REP missing — run 'timeline $LABEL' first" >&2; exit 1; }
    echo "[kernels] nsys stats -> $DIR"
    nsys stats --force-export=true --format csv \
      -r cuda_gpu_kern_sum -r cuda_gpu_mem_time_sum -r cuda_gpu_mem_size_sum \
      -o "$DIR/stats" "$REP" > "$DIR/nsys_stats.log" 2>&1
    # gpu trace CSV is large; keep it on disk (gitignored) for ad-hoc digging.
    nsys stats --format csv -r cuda_gpu_trace -o "$DIR/stats" "$REP" >> "$DIR/nsys_stats.log" 2>&1 || true
    SQLITE="$DIR/timeline.sqlite"
    [[ -f "$SQLITE" ]] || nsys export --type=sqlite -o "$SQLITE" "$REP"
    python3 "$ROOT/perf_campaign/launch_gaps.py" "$SQLITE" "$DIR/gaps.json"
    python3 "$ROOT/perf_campaign/profile_summary.py" "$DIR" > "$DIR/summary.md"
    echo "[kernels] wrote $DIR/summary.md"
    ;;

  ncu)
    KREGEX="${1:-}"; [[ -n "$KREGEX" ]] || { echo "usage: profile.sh ncu <label> <kernel-regex>" >&2; exit 2; }
    [[ -x "$BIN" ]] || { echo "ERROR: binary not found: $BIN" >&2; exit 1; }
    mkdir -p "$DIR"
    SAN="$(echo "$KREGEX" | tr -c 'A-Za-z0-9_' '_' | cut -c1-48)"
    OUT="$DIR/ncu_${SAN}"
    # Deep-metric section set: launch config + registers, occupancy, scheduler
    # and warp-state stalls (barrier), memory workload (local-mem traffic),
    # SOL (XU/pipe utilization).
    NCU_CMD=(ncu
      --profile-from-start off
      --kernel-name "regex:$KREGEX"
      --launch-count 5
      --section LaunchStats --section Occupancy --section SchedulerStats
      --section WarpStateStats --section MemoryWorkloadAnalysis
      --section SpeedOfLight --section SpeedOfLight_RooflineChart
      --force-overwrite -o "$OUT")
    FULL=(env LFS_NVTX=1 LFS_PROFILE_START_ITER="$START" LFS_PROFILE_STOP_ITER="$STOP"
      "${NCU_CMD[@]}" "$BIN")
    # shellcheck disable=SC2046
    FULL+=($(train_args))
    if sudo -n true 2>/dev/null; then
      echo "[ncu] passwordless sudo OK; profiling '$KREGEX' (bench lock held)"
      flock "$BENCH_LOCK" sudo -E "${FULL[@]}" 2>&1 | tee "$DIR/ncu_${SAN}.log"
      sudo ncu --import "${OUT}.ncu-rep" --page details --csv > "$DIR/ncu_${SAN}.csv" || true
      write_meta ncu
      rm -rf "$DIR/train_out"
    else
      cat >&2 <<EOF
[ncu] Hardware counters are admin-locked (RmProfilingAdminOnly=1) and
passwordless sudo is unavailable. Maintainer: either run

  flock $BENCH_LOCK sudo -E ${FULL[*]}

or unlock counters machine-wide (persists until reboot; then plain ncu works):

  sudo systemctl stop nvidia-persistenced 2>/dev/null; \\
  sudo modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia; \\
  sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0

(or add 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' to
/etc/modprobe.d/nvidia-profiling.conf and reboot).
EOF
      exit 3
    fi
    ;;
esac
