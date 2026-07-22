# Training-snapshot benchmark prototype (P0)

Standalone CUDA microbenchmark for **PROJECT_FORMAT_PLAN.md §3 / §7** training-snapshot
gates. It does **not** link LichtFeld Studio libraries — plain CUDA runtime + C++20 std.

**Orchestrator note:** this directory is write + compile-check only under GPU contention.
Run the binary later in an idle window; do not launch it while another heavy GPU job owns
the device.

## Build

```bash
./build.sh
# or:
# nvcc -O2 -std=c++20 -arch=native snapshot_bench.cu -o snapshot_bench
```

Optional: `NVCC=/path/to/nvcc ARCH_FLAG="-arch=sm_89" ./build.sh`

## Design contract (default) vs stress-overlap

Production rule (PROJECT_FORMAT_PLAN.md §3): **one snapshot + one writer job in flight;
newer requests coalesce**. The real system never begins a snapshot while the previous
background serialization/write job is still running.

| Mode | Flag | Behavior |
|------|------|----------|
| **Contract (default)** | *(none)* | Before each snapshot cycle, join the previous cycle’s background CRC+disk job. That wait is **not** part of the pause clock — it models request coalescing. Pause gates are defined over these contract-mode cycles. |
| **Stress-overlap** | `--stress-overlap` | Leave the previous writer running into the next pause so CRC/disk contend with D2H. Reference measurement only; intentionally violates the design contract. Do not treat stress-overlap pause numbers as gate evidence. |

A cold first cycle has no previous writer; under contract mode it is the same path production
takes when the writer queue is empty. Warm cycles must also be free of previous-writer
contention — that is what the default enforces.

## What it simulates

1. **Training state** — default ~10 GiB of device tensors (256 MiB buffers) plus a lightweight
   “optimizer step” kernel on a dedicated non-blocking stream.
2. **Snapshot cycle** at a safe point:
   - **contract mode:** join any in-flight writer (outside the pause clock);
   - sync every stream that mutates persisted state;
   - stamp one snapshot id into every tensor header;
   - banded D2H through a pinned ring (default 3 × 128 MiB, ceiling 512 MiB) into
     **pre-faulted pageable** host staging (touched once at alloc);
   - pinned→pageable drain uses `--drain-threads` workers (default 1; multi-thread H2H
     contending host DRAM bandwidth made pause/efficiency/step worse on 4090) and
     `--drain-copy` (default `nt`: AVX non-temporal streaming stores to avoid LLC
     pollution on write-once staging; `std` = libc memcpy);
   - **pause ends** when the last D2H CUDA event completes (`optimizer-may-mutate`);
   - background thread: CRC32 over staging + rate-limited disk write (`--disk`, default
     `/dev/null`) while training steps resume;
   - **contract mode:** join that writer before the next cycle’s snapshot (again outside
     the pause clock).
3. **Consistency proof** — every staging buffer carries the same snapshot stamp; post-resume
   device mutations (re-stamp) must not appear in host staging.

## CLI

| Flag | Default | Meaning |
|------|---------|---------|
| `--gib N` | 10 | Total device tensor bytes (GiB) |
| `--cycles N` | 20 | Snapshot cycles for pause percentiles |
| `--band-mib N` | 128 | Pinned ring band size |
| `--bands N` | 3 | Ring depth |
| `--disk-throttle-mbps N` | 0 | Background write throttle (0 = none) for the **primary** pass |
| `--disk PATH` | `/dev/null` | Serialization write target |
| `--drain-threads N` | 1 | Parallel workers for each pinned→pageable band drain |
| `--drain-copy nt\|std` | `nt` | Drain path: `nt` = AVX streaming stores; `std` = memcpy |
| `--stress-overlap` | off | Overlap previous writer with next pause (reference only) |
| `--json PATH` | — | Machine-readable results |
| `--quick` | off | Smoke: 1 GiB, 3 cycles |

Example (orchestrator full gate run — **contract mode**):

```bash
./snapshot_bench --json out.json
./snapshot_bench --quick --json smoke.json   # short path only
# Reference contention measurement (not for gates):
./snapshot_bench --stress-overlap --json stress.json
```

## Measured numbers

Over ≥20 cycles (unless `--quick`), plus the **cold first cycle** reported separately.
**Pause gates apply to contract-mode cycles only** (default; no `--stress-overlap`).

| Metric | Definition | Gate |
|--------|------------|------|
| `pause_ms` | Safe-point enter (before stream syncs) → last D2H event complete | p95 ≤ 750 ms; max ≤ 1000 ms (cold included in max) |
| `d2h_efficiency` | Mean banded-ring D2H GiB/s ÷ pinned-direct baseline (≤512 MiB large transfer at startup) | ≥ 80 % |
| `pinned_bytes_peak` | Pinned ring only (baseline pin is temporary and freed) | ≤ 512 MiB |
| `rss_delta` / extra host | `/proc/self/status` VmRSS after prefaulted staging minus before | rss_delta ≤ snapshot + 768 MiB |
| `step_time_regression_pct` | Mean step time over 100 post-resume iterations vs pre-snapshot baseline | ≤ 10 % |
| `disk_throttle_delta_ms` | \|mean pause unthrottled − mean pause @ 500 MB/s\| (short comparison pass) | ≤ 100 ms |

Also always: consistency/no-leak proof across cycles.

## Reading the output

Human table first, then a **Gates** section:

```
=== Gates ===
  [PASS] pause_p95_ms  p95=412.300 (gate ≤ 750)
  ...
Overall: PASS
```

Process exit code is `0` on overall PASS, `1` on any gate FAIL (or CUDA error → abort with
non-zero via `CHECK_CUDA`). **`Overall: FAIL` always exits 1.**

`--json out.json` mirrors every metric and per-gate `{pass, detail}` plus `overall_pass`.

### Interpreting a FAIL

- **pause_*** — PCIe/D2H path too slow or safe-point work too heavy; plan says stop training
  autosave and renegotiate (storage versioning / D2D), not ship the stall. If you used
  `--stress-overlap`, re-run in contract mode first — warm pauses under overlap are not
  design-valid samples.
- **d2h_efficiency** — ring/H2H pipeline under-utilizing the bus vs a single large pinned copy.
  Default is single-thread NT drain (`--drain-threads 1 --drain-copy nt`); multi-thread H2H
  can *worsen* efficiency via host DRAM contention. Compare `--drain-copy std` only as a
  reference if NT regresses.
- **pinned_bytes_peak** — ring config exceeds 512 MiB (`bands * band-mib`).
- **extra_host_ram** — staging alloc or bookkeeping blew past snapshot + 768 MiB RSS budget.
- **step_time_regression_pct** — background CRC/disk is contending CPU; production must
  self-throttle / nice the writer.
- **disk_throttle_delta_ms** — disk speed leaked into the pause path (must not: serialization
  is after last D2H, and the previous writer must be joined before the next pause in
  contract mode).
- **consistency_proof** — stamp mismatch or post-resume mutation visible in staging.

## Design notes (spec gaps)

- Device layout: `ceil(gib / 256 MiB)` buffers; last buffer may be shorter.
- Stamp: one `uint64_t` at the head of each buffer; optimizer kernel skips that header.
- Pause clock is host `steady_clock`; D2H baseline uses `cudaEvent` elapsed time.
- Final H2H from the last pin band(s) into pageable staging may finish **after** the pause
  mark (matches “last D2H event complete”); staging is complete before CRC/serialization.
- Throttle comparison always runs unthrottled vs 500 MB/s (up to 10 cycles each, or all
  cycles under `--quick`), independent of `--disk-throttle-mbps` on the primary pass, and
  uses the same contract/stress join policy as the primary pass.
- CRC32 is ISO/HDLC reflected (`0xEDB88320`), not CRC32C — integrity marker only.
