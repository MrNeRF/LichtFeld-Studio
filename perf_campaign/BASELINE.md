# Phase 0 baseline (gate for all later tasks)

Recorded on **2026-08-06**. Every subsequent campaign task must measure against
these numbers and report before/after. Do not silently regress wall-clock,
steady-state allocs/iter, peak VRAM, or quality (loss/PSNR).

## Machine

| Field | Value |
|---|---|
| Host | `machalonobis` |
| GPU | NVIDIA GeForce RTX 4080, 16376 MiB |
| Driver | 595.84 |
| CUDA (toolkit, build) | 13.3.73 (sm_89) |
| OS | Linux 7.0.0-29-generic x86_64 |
| CPUs | 24 |

## Software / commit

| Field | Value |
|---|---|
| Branch | `perf/spirulae-parity` |
| Commit (at baseline) | `e5506f39` (Phase 0.3; numbers collected just before this commit on dirty tree `f6b0d5b6`) |
| Binary | `build/tests/LichtFeld-Studio` |
| Bench script | `perf_campaign/bench.sh` |

## Workload

| Field | Value |
|---|---|
| Dataset | `/home/gauss/data/360_v2/bonsai` (MipNeRF360 bonsai, COLMAP) |
| Images | `images_4` |
| Strategy | `mrnf` |
| Iterations | 2000 |
| Warmup (excluded from steady) | 200 |
| max_cap | 500000 |
| Headless | yes |
| Eval / PSNR | off (`last_psnr = -1`) |
| Env | `LFS_PERF_BENCH=1` |

Raw per-run JSON: `perf_campaign/runs/20260806T171556Z_run{1,2,3}/bench_report.json`  
Aggregate: `perf_campaign/runs/latest_aggregate.json`

## Per-run numbers

| run | wall_s | steady_ms/iter | warmup_ms/iter | steady_allocs/iter | warmup_allocs/iter | peak_VRAM_MiB | B/splat | last_loss | final_splats |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.96 | 4.111 | ~2.75 | 5.05 | ~0.2+5 | 1156.3 | 429.0 | 0.04049 | ~404k |
| 2 | 9.00 | 4.132 | ~2.75 | 5.05 | ~0.2+5 | 1156.3 | 429.0 | 0.03800 | ~404k |
| 3 | 9.06 | 4.129 | ~2.75 | 5.05 | ~0.2+5 | 1156.3 | 429.0 | 0.03981 | ~404k |

Exact metrics from run 1 (`perf_bench.json`):

```
wall_seconds            8.96–9.06
steady_ms_per_iter      4.111 / 4.132 / 4.129
steady_allocs_per_iter  5.05 / 5.05 / 5.05
warmup_allocs_per_iter  (see JSON; dominated by sort-buffer growth)
peak_cuda_used_bytes    1_212_481_536  (~1156 MiB)
ledger.bytes_per_splat  429.0
ledger.params_bytes     ~100.1e6
ledger.optimizer_bytes  ~69.4e6
ledger.gradients_or_helpers_bytes  0
ledger.densify_aux_bytes ~3.63e6
last_loss               0.038–0.040
last_psnr               -1 (eval disabled)
```

## Medians (gate values)

| Metric | Median |
|---|---:|
| **G4 wall_seconds** | **9.00** |
| **G4 steady_ms_per_iter** | **4.129** |
| **G2 steady_allocs_per_iter** | **5.05** |
| **peak_cuda_used_MiB** | **1156.3** |
| **G1 bytes_per_splat** | **429.0** |

### Training-state ledger (run 1, representative)

| Bucket | Bytes | B/splat (@ ~404k live) |
|---|---:|---:|
| params | 100,127,224 | ~248 |
| optimizer | 69,442,220 | ~172 |
| gradients_or_helpers | 0 | 0 |
| densify_aux | 3,633,489 | ~9 |
| **total** | **173,202,933** | **429.0** |

Note: densify_aux is slightly above the pure 8 B/splat `densification_info`
because the soft-delete mask / other aux can be present; unit test with pure
`[2,N]` densify info matches 428 B/splat exactly (SH3).

## How to re-run

```bash
# Single run
./perf_campaign/bench.sh

# Three-run baseline (writes perf_campaign/runs/<stamp>_run*/ + latest_aggregate.json)
LFS_BENCH_ITERS=2000 LFS_PERF_BENCH_WARMUP=200 LFS_BENCH_MAX_CAP=500000 \
  LFS_BENCH_DATASET=/home/gauss/data/360_v2/bonsai \
  ./perf_campaign/bench.sh --runs 3
```

## North-star gate status at baseline

| Gate | Baseline (this machine) | Target |
|---|---|---|
| G1 B/splat (SH3 train) | **429** | ≤ 304 |
| G2 steady allocs/iter | **5.05** | 0 |
| G3 host syncs/iter | not yet instrumented | 0 hard |
| G4 wall / ms/iter | **9.00 s / 4.129 ms** | ≤ baseline per task |
| G5 peak refine VRAM | not isolated yet (peak 1156 MiB overall) | ≤ 1.1× steady |
| G6 quality | last_loss ≈ 0.04 (no PSNR yet) | no regression |
| G7 GUI additive VRAM | N/A (headless) | later |
