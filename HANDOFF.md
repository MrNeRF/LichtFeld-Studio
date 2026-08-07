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
| (Validity caveat: numbers verified on the DEFAULT path — FastGS+MRNF, quant ON. See §4 must-fix wave for non-default configs.) | | | |
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

**MUST-FIX WAVE (from the hostile final review, 2026-08-07 — full report:
`~/lfs-campaign-out/ADVERSARIAL-REVIEW.md`).** The review found a systemic blind spot:
strategy test suites forced quantization OFF, so the dual-representation state
(q16 SH codes + joint Adam moments) was only ever exercised through the fused FastGS
chokepoint. Outside it: 4 blockers + 5 majors of crash/silent-corruption on documented
configs (legacy-codec fallback flag, gut/gsplat mode, improved_gs_plus strategy,
checkpoint-after-real-steps), plus a PLY cancel-path UAF and a viewer-grow ordering bug
(sibling of the fixed NVRM issue). Two fix orders are dispatched and chained:
- WO-FIX-CODEC (in queue): BL-1/BL-2, MJ-1..MJ-4, bounds-family hardening + the missing
  quant-ON strategy/checkpoint-after-step test suites.
- WO-FIX-INTEG (chained): BL-3/BL-4, MJ-5/MJ-6/MJ-12, MJ-14/MJ-15 triage, MN-1/2/4.
IMPORTANT until these land: the DEFAULT bench path (FastGS+MRNF, quant ON) is validated;
gut mode, improved_gs_plus, and the LFS_ADAM_LEGACY_CODEC fallback are UNSAFE with
quantization enabled.

**In flight**
- WO-WARP-BWD: warp-culling port into blend_backward (est. −0.2–0.5 ms/iter further).

**Queued to review-ready**
1. FIX-CODEC → FIX-INTEG (above) with their new test suites.
2. Final full-suite + dual-workload bench → final numbers table.
3. Rebase onto current origin/master (MN-11; one commit behind, textual overlap with
   viewport work — do it deliberately).
4. Publish final rev: `./push-clean.sh <rev>` (owner-run).

**Ship-as-documented (filed with repro recipes from the review)**
- MJ-7 DLPack-of-expanded-view UAF (Python edge), MJ-8 RowProxy write-on-view,
  MJ-9 render-thread TLS caches retained for session, MJ-10/MJ-11 gsplat stream/teardown
  latents, MJ-13 GT cache invisible to OOM pressure, MN-1..MN-13 minors/nits.
- ISS-007: 10-min MANUAL GUI check (owner). ISS-016/019 pre-existing reds (attribution
  notes in the review). ncu counters admin-locked (one-liner + reboot in profile.sh docs).

**Deferred backlog** (unchanged): graphs (Windows/WDDM case), HP-2 parallel decode,
per-band SH bits, hardening themes B–F, bucket-pool tuning, Vulkan 128-bit packing audit.

**Process addition:** RULES.md now carries a red-provenance clause — a failing test may be
called "pre-existing" only with git-log or branch-point-run proof (a campaign-added
fail-loud guard was misclassified once; MJ-14).

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
