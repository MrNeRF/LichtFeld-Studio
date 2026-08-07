# Campaign Handoff — Speed & VRAM Optimization (2026-08-06 → 2026-08-07)

Branch: **`lfs-elite`** (local; clean published mirror at `origin/lfs-elite` via `./push-clean.sh <rev>`).
This document is INTERNAL (excluded from publication). Full evidence trail:
`perf_campaign/{PROGRESS,ISSUES,BASELINE,RULES}.md`, analysis reports in `docs/analysis/`,
worker outputs in `~/lfs-campaign-out/`.

---

## 1. Results (all 3-run medians, RTX 4080, measured — receipts in PROGRESS.md)

| Metric | Campaign start | Current | Δ |
|---|---:|---:|---|
| Bonsai steady ms/iter | 4.129 | **3.113** | **−25%** |
| Bonsai wall (2k iters) | 9.00 s | ~7.3 s | −19% |
| Bicycle-7k steady ms/iter | 3.290 | **2.737** | **−17%** |
| Bicycle-7k wall | 31.15 s | ~20.6 s | **−34%** |
| Persistent training state | 429 B/splat | **304.3 / 306.8 B/splat** | −29% — at/below the reference implementation (304) |
| Steady-state allocations/iter | 5.05 | ~0.10 | −98% |
| Hard host syncs/iter | 1 | 0 | eliminated |
| Dataloader wait (was invisible) | ~4.8 ms/img decode | ~0.005 ms | GT device cache |
| Bicycle final-loss band | 0.098–0.121 | **0.085–0.101** | quality BETTER than baseline |
| GUI startup reservation (5M cap) | 1183 MiB | 142 MiB | −1.04 GB |
| PLY save (1M splats) | 1.57 s | 0.70 s | 2.2× |

Peak VRAM: bonsai ~1495 MiB / bicycle ~1636 MiB *including* the opt-in GT device cache
(339/564 MiB, budget-gated, own ledger line, disable to reclaim).

## 2. What was done (by area)

**FastGS rasterizer (training hot path)**
- Persistent high-water sort buffers (was 5× cudaMallocAsync/free per step); async
  `n_instances` (the per-step hard `cudaStreamSynchronize` is gone); preflight
  pointer-validation debug-only.
- Background compose fused into `blend_cu`; backward unblend pass removed.
- `blend_backward` T_eff clamp (skips the dead splat tail; bit-identical math).
- **Warp-level sub-tile culling** in `blend_cu` (per the cited CGIT paper, doi:10.1145/3820019,
  using our own Vulkan shader's ellipse-exact subtile test): −18.5% kernel, bit-identical;
  batch size retuned 256→192. Backward port (WO-WARP-BWD) in flight.
- Philox RNG in all noise/relocation kernels — the old per-thread XORWOW init was 99.5%
  of the noise kernel (measured 1149 µs → 6.5 µs at N=400k, ~1.3 ms/iter recovered).

**Quantization (the memory prize)**
- Joint (u, log_s) Adam codec, 16-bit non-SH / 8-bit SH, per-256-block bounds (−20 B/splat).
- SH-rest 16-bit value quantization with decode-in-registers and single re-encode in the
  fused Adam tail; densify re-encode handles capacity growth (−102 B/splat). Runtime
  fallbacks: `LFS_SH_VALUE_QUANT`, `LFS_ADAM_LEGACY_CODEC`.
- Gradient-recovery + unfused `AdamOptimizer::step()` implemented for the joint codec
  (the ISS-015 root cause — found by bisect after numerical-gradient tests caught it).

**Tensor library**
- Host dispatch: lazy-IR recording now opt-in (was a global mutex + string per op);
  `has_lazy_expr()` lock-free; contiguous same-shape binary fast path; shared TensorState
  on copy + inline small-vector shapes.
- Kernels: binary(+reduce) fusion; dead coalesced/smem Channel3D broadcast kernels wired;
  `where` host clones removed; SM-capped grid-stride elementwise; device-side mean/prod
  finalize; fp16 API holes filled (reductions/unary/clamp/fill); half2 fast paths.
- Memory: zero-stride `expand`/`broadcast_to` views behind a correctness firewall
  (incl. materialize-on-raw-pointer-escape); strided reduce with per-shape heuristic
  (up to 5× vs transpose-copy); `out=` destination APIs; allocator hygiene (pooled
  metadata allocs, empty-slab reclaim, no 1-byte sentinels).
- Correctness: training-reachable strided-op bug set fixed with regression tests
  (legacy findings doc Theme A subset).

**Trainer / losses / strategies**
- Loss-workspace union (5 mutually-exclusive variants share one arena region);
  `zero_terms` (23.7 MiB of literal zeros) eliminated; fp16 SSIM partials everywhere the
  fused path already proved them; regularizer loss scalars folded into fused backward;
  persistent cropbox/frozen masks; mask-chain fusion; densification-info zeroed in-kernel.
- MRNF densify: fused free-slot writes, batched refine readbacks, selection-based medians,
  reusable child workspace, ≤2× bounded compaction, Adam capacity-invariant guard.
- GT device cache (opt-in by VRAM budget) — killed the single-threaded host JPEG-decode
  bottleneck (was 73% of the bicycle window). `dataloader_wait_ms` is now a bench metric.

**Gsplat path**: per-step isect buffer frees → persistent (0 steady allocs/forward).
**IO**: PLY save parallel-pack + buffered write (2.2×).
**GUI**: exportable splat block grows with live N (was max_cap upfront); CUDA-VMM/Vulkan
release-order bug fixed (the NVRM `invalid mmap context` errors); TLS buffer release hooks;
teardown-order release before pool shutdown (exit-139 class).
**Checkpoint/export**: resume fixed for joint codec + q16 (was a segfault); Sog export
dequantizes correctly; graph-capture device-fault test fixed.

## 3. How it is verified
- `perf_campaign/bench.sh` — dual-workload gate (bonsai 2k; bicycle 7k = quality canary,
  compare loss CURVES). `perf_campaign/profile.sh` — nsys steady-window kernel profiling
  (late window `LFS_PROF_START=1600` matters: one regression was SH-degree-gated and
  invisible early).
- Every change: fail-first test evidence + before/after numbers in `PROGRESS.md`.
  ~60 new tests (codecs, gradients numerical, leak-regression, teardown, views, resume).
- Full-suite gate mandatory since ISS-015 (loss curves alone missed a gradient bug).

## 4. What remains

**In flight right now**
- WO-WARP-BWD: warp-culling port into `blend_backward` (est. −0.2–0.5 ms/iter further).
- Adversarial review agent over the full diff → findings become fix orders / known-issues.

**Queued to review-ready**
1. Triage adversarial findings (blockers fixed, rest documented).
2. Final full-suite + dual-workload bench → final numbers table.
3. Rebase/merge onto current `origin/master` (small drift, ~1 commit behind at last check).
4. Publish final rev: `./push-clean.sh <rev>` (owner-run; strips internal material,
   cleans messages, force-pushes `origin/lfs-elite`).

**Open issues (documented, non-blocking)**
- ISS-007: 10-min MANUAL GUI check — zero-copy viewport through a densify grow (owner).
- ISS-016/017/019: pre-existing reds (video-extractor naming ×3, tensor-reserve overflow
  edge, python-integration visualizer) — independent of campaign.
- ncu hardware counters admin-locked; one-liner + reboot documented in profile.sh.
- allocs/iter ~0.10 vs the pure 0.05 ideal; ex-cache peak audit trail in ISSUES.md.

**Deferred backlog (analyst reports, ranked in `~/lfs-campaign-out/analysis/a-backlog.md`)**
- CUDA graphs (measured ≤0.2 ms upside here; revisit after WARP-BWD shrinks the step,
  and for Windows/WDDM where launch overhead is far higher).
- HP-2 parallel host decode (for datasets exceeding the GT-cache budget).
- Per-band SH bit budgets (16/12) — plan Phase 8, gate G6, worth ~−20 B/splat more.
- Remaining tensor-lib hardening themes B–F fix-specs; bucket-pool size-class tuning;
  Vulkan `Splat_2D_AlphaBlend` 128-bit packing audit; ellipse tile-intersect experiment (M).

## 5. Ops runbook
- Workers: grok CLI via systemd units in `lfs.slice` (MemoryHigh 14G), max 3 concurrent,
  queue runner + stall watchdog in `~/lfs-campaign-out/`. Build: `./perf_campaign/build.sh`
  (2-slot machine-wide semaphore — 30 GB RAM box, oomd kills the desktop above ~50%
  pressure; this was learned the hard way, see ISS-012 and RULES.md).
- Worktrees: `./perf_campaign/setup-worktree.sh` (submodules, ccache-not-sccache,
  VCPKG_ROOT, CUDA PATH, libtorch symlink — all five traps).
- GPU timing: always under `flock /tmp/lfs-bench.lock`, quiet machine for wall-clock.
- Publication: `./push-clean.sh <verified-rev>` — owner-run only.

## 6. Lessons that must survive
1. Measure before believing any ranking — the top-3 speed wins (RNG init, decode
  bottleneck, dead-tail walk) were all invisible to static analysis, and CUDA graphs
  (the assumed endgame) measured near-worthless here.
2. Loss curves are NOT a gradient-correctness gate; numerical-gradient suites are.
3. Fuses must be sized for the worst legitimate phase (builds), not the average.
4. Never edit a running bash script; never let workers read prompts from a tree that
  other workers mutate (bisect wiped the work orders once).
5. The bicycle canary earns its keep — but only past densification growth (7k iters).
