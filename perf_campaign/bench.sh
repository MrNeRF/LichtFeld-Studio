#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Phase 0.3 bench gate — fixed-seed headless training smoke.
# Reports: wall, steady ms/iter (post-warmup), peak VRAM, allocs/iter
# (warmup vs steady), final loss/PSNR, and the training-state ledger.
#
# Usage:
#   ./perf_campaign/bench.sh              # single run, prints JSON path
#   ./perf_campaign/bench.sh --runs 3     # 3 runs (baseline collection)
#   LFS_BENCH_DATASET=/path ./perf_campaign/bench.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Prefer the tests-enabled build tree (has lichtfeld_tests + full app);
# fall back to plain build/.
if [[ -d "$ROOT/build/tests" && -f "$ROOT/build/tests/build.ninja" ]]; then
  BUILD_DIR="$ROOT/build/tests"
else
  BUILD_DIR="$ROOT/build"
fi

ITERS="${LFS_BENCH_ITERS:-2000}"
WARMUP="${LFS_BENCH_WARMUP:-200}"
MAX_CAP="${LFS_BENCH_MAX_CAP:-500000}"
STRATEGY="${LFS_BENCH_STRATEGY:-mrnf}"
IMAGES="${LFS_BENCH_IMAGES:-images_4}"
RUNS=1
OUT_ROOT="${LFS_BENCH_OUT:-$ROOT/perf_campaign/runs}"
SKIP_BUILD="${LFS_BENCH_SKIP_BUILD:-0}"

# Dataset discovery (non-interactive). Override with LFS_BENCH_DATASET.
DATASET="${LFS_BENCH_DATASET:-}"
if [[ -z "$DATASET" ]]; then
  for candidate in \
    "/home/gauss/data/360_v2/bonsai" \
    "/home/gauss/data/360_v2/counter" \
    "/home/gauss/data/360_v2/room" \
    "/home/gauss/data/360_v2/garden" \
    "$ROOT/data/bonsai" \
    "$ROOT/data/garden"; do
    if [[ -d "$candidate/sparse" || -d "$candidate/images_4" || -d "$candidate/images" ]]; then
      DATASET="$candidate"
      break
    fi
  done
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DATASET" || ! -d "$DATASET" ]]; then
  cat >&2 <<EOF
ERROR: No COLMAP/nerfstudio dataset found.
Set LFS_BENCH_DATASET to a scene directory, or place one under data/.
EOF
  exit 1
fi

BIN="$BUILD_DIR/LichtFeld-Studio"
if [[ ! -x "$BIN" ]]; then
  # Some builds place the binary under bin/
  if [[ -x "$BUILD_DIR/bin/LichtFeld-Studio" ]]; then
    BIN="$BUILD_DIR/bin/LichtFeld-Studio"
  fi
fi

echo "=== Phase 0.3 bench gate ==="
echo "build:    $BUILD_DIR"
echo "binary:   $BIN"
echo "dataset:  $DATASET"
echo "iters:    $ITERS  warmup: $WARMUP  max_cap: $MAX_CAP  strategy: $STRATEGY"
echo "runs:     $RUNS"
echo

if [[ "$SKIP_BUILD" == "1" ]]; then
  echo "[1/3] Skipping build (LFS_BENCH_SKIP_BUILD=1) — binary must already be current"
else
  echo "[1/3] Building via bounded ./perf_campaign/build.sh (never unbounded -j)..."
  ./perf_campaign/build.sh "$BUILD_DIR"
fi
if [[ ! -x "$BIN" ]]; then
  echo "ERROR: binary not found at $BIN after build" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
COMMIT="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
HOST="$(hostname)"
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"

REPORT_PATHS=()
for ((r=1; r<=RUNS; r++)); do
  RUN_DIR="$OUT_ROOT/${STAMP}_run${r}"
  mkdir -p "$RUN_DIR"
  echo
  echo "[2/3] Run $r/$RUNS → $RUN_DIR"
  # --perf-bench activates the in-process collector (writes perf_bench.json).
  # shellcheck disable=SC2086
  /usr/bin/time -f 'elapsed_sec=%e' -o "$RUN_DIR/time.txt" \
    "$BIN" \
      -d "$DATASET" \
      -o "$RUN_DIR" \
      --images "$IMAGES" \
      --headless \
      --iter "$ITERS" \
      --max-cap "$MAX_CAP" \
      --strategy "$STRATEGY" \
      --perf-bench \
      --perf-bench-warmup="$WARMUP" \
      2>&1 | tee "$RUN_DIR/train.log"

  if [[ ! -f "$RUN_DIR/perf_bench.json" ]]; then
    echo "ERROR: perf_bench.json missing after run $r — is --perf-bench wired?" >&2
    exit 1
  fi
  # Annotate with machine/commit metadata for BASELINE.md assembly.
  {
    echo "{"
    echo "  \"run\": $r,"
    echo "  \"commit\": \"$COMMIT\","
    echo "  \"host\": \"$HOST\","
    echo "  \"gpu\": \"$GPU\","
    echo "  \"dataset\": \"$DATASET\","
    echo "  \"images\": \"$IMAGES\","
    echo "  \"iters\": $ITERS,"
    echo "  \"warmup\": $WARMUP,"
    echo "  \"max_cap\": $MAX_CAP,"
    echo "  \"strategy\": \"$STRATEGY\","
    echo "  \"metrics\": $(cat "$RUN_DIR/perf_bench.json")"
    echo "}"
  } > "$RUN_DIR/bench_report.json"
  REPORT_PATHS+=("$RUN_DIR/bench_report.json")
  echo "  wrote $RUN_DIR/bench_report.json"
done

echo
echo "[3/3] Summary"
python3 - "$OUT_ROOT" "${REPORT_PATHS[@]}" <<'PY'
import json, sys, statistics, os
out_root = sys.argv[1]
paths = sys.argv[2:]
rows = []
for p in paths:
    with open(p) as f:
        rows.append(json.load(f))

def med(xs):
    return statistics.median(xs) if xs else float("nan")

def get(m, *keys, default=0):
    cur = m
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

print(f"{'run':>4} {'wall_s':>10} {'steady_ms':>10} {'dl_wait':>10} {'gt_cache':>10} {'alloc/s':>10} {'peak_MiB':>10} {'B/splat':>10} {'loss':>10}")
walls, steadies, dl_waits, gt_caches, allocs, peaks, bps = [], [], [], [], [], [], []
for r in rows:
    m = r["metrics"]
    wall = get(m, "wall_seconds")
    steady = get(m, "steady_ms_per_iter")
    # Prefer steady-state per-iter wait; fall back to overall per-iter.
    dl_wait = get(m, "steady_dataloader_wait_ms_per_iter")
    if not dl_wait:
        dl_wait = get(m, "dataloader_wait_ms_per_iter")
    gt_cache = get(m, "gt_cache_mib")
    if not gt_cache:
        gt_cache = get(m, "gt_cache_bytes") / (1024*1024)
    a = get(m, "steady_allocs_per_iter")
    peak = get(m, "peak_cuda_used_bytes") / (1024*1024)
    b = get(m, "ledger", "bytes_per_splat")
    loss = get(m, "last_loss")
    walls.append(wall); steadies.append(steady); dl_waits.append(dl_wait)
    gt_caches.append(gt_cache); allocs.append(a); peaks.append(peak); bps.append(b)
    print(f"{r['run']:4d} {wall:10.2f} {steady:10.3f} {dl_wait:10.3f} {gt_cache:10.1f} {a:10.2f} {peak:10.1f} {b:10.1f} {loss:10.5f}")

if len(rows) > 1:
    print("-" * 100)
    print(f"{'med':>4} {med(walls):10.2f} {med(steadies):10.3f} {med(dl_waits):10.3f} {med(gt_caches):10.1f} {med(allocs):10.2f} {med(peaks):10.1f} {med(bps):10.1f}")

# Always write a machine-readable aggregate next to the runs.
agg = {
    "commit": rows[0]["commit"] if rows else "",
    "host": rows[0]["host"] if rows else "",
    "gpu": rows[0]["gpu"] if rows else "",
    "dataset": rows[0]["dataset"] if rows else "",
    "n_runs": len(rows),
    "runs": rows,
    "median": {
        "wall_seconds": med(walls),
        "steady_ms_per_iter": med(steadies),
        "steady_dataloader_wait_ms_per_iter": med(dl_waits),
        "gt_cache_mib": med(gt_caches),
        "steady_allocs_per_iter": med(allocs),
        "peak_cuda_used_mib": med(peaks),
        "bytes_per_splat": med(bps),
    },
}
agg_path = os.path.join(out_root, "latest_aggregate.json")
with open(agg_path, "w") as f:
    json.dump(agg, f, indent=2)
print(f"\nAggregate: {agg_path}")
for p in paths:
    print(f"  report: {p}")
PY

echo
echo "Done."
