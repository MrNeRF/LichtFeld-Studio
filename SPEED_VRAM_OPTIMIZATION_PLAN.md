# Speed & VRAM Optimization Plan — match spirulae-splat's footprint, beat it on speed (2026-08-06)

Source: 8 parallel Grok deep-assessments of LichtFeld-Studio and the local spirulae-splat
checkout, cross-verified against the code by hand (key claims in `forward.cu`,
`sh_layout.cuh`, `regularization.cpp`, `adam_optimizer.hpp`, spirulae `Tensor.h`/`PoolSlots.h`
were re-read directly). Full evidence reports: `docs/analysis/spirulae-comparison/*.md`.
Scope: tensor lib + training (MRNF strategy + FastGS rasterizer) + GUI/headless, per
the guiding constraint **speed is king — nothing may regress, everything improves**.

---

## 0. What spirulae-splat actually does (verified, not folklore)

The maintainer's "memory allocation never fragments and keeps cache at minimum" claim
decomposes into three verified mechanisms (`spirulae-splat/src/core/Tensor.h`, `PoolSlots.h`):

1. **No dynamic allocator at all.** ~180 named buffers in one compile-time X-macro
   registry; each is a single `cudaMalloc` with **high-water-mark semantics** (grow-only,
   exact size, never shrinks, freed only at reset/shutdown). One shared monotonic
   `DeviceScratch` blob serves all CUB temporaries. Zero fragmentation because there is
   zero steady-state alloc/free churn — after warm-up a training step issues **0 allocations**.
2. **Fixed `max_num_splats` capacity; densification is 100% in-place.** New splats are
   written into pre-existing rows; `cur_num_splats` advances. No realloc, no old+new
   coexistence, ever.
3. **Aggressive quantization** (default `quantization_level=1` + FPBO):
   SH values 16-bit block-quantized (decode in registers), Adam state as joint
   `(u, log_s)` 16-bit (non-SH) / 8-bit (SH) codec with per-256-block bounds, world
   gradients **never allocated** (projection-backward fused into Adam = "FPBO"),
   grad-quant fallback uses a signed-symmetric codec with exact `decode(0)=0`.

Result: **~304 B/splat persistent training state** (SH3) vs their own fp32 baseline of 960.

What spirulae does **not** have (our headroom to beat them): CUDA graphs, multi-stream
overlap, incremental sort, capacity-amortized growth (they free+malloc on every new
high-water). They also carry a D2H sync for `nnz`/`n_isects` like we do.

## 0b. Where LFS actually stands (corrects several wrong assumptions)

- LFS Adam moments are **already uint8-quantized** (+ per-primitive fp32 scales) —
  `adam_optimizer.hpp:45-54`. Not fp32 m+v.
- LFS FastGS **already fuses Adam into the raster backward** and allocates **no persistent
  world gradients** — functional FPBO equivalent (`kernels_backward.cuh`, `backward.cu:117-168`).
- MRNF/MCMC params already pre-reserve `max_cap` via `zeros_direct` outside the pool.
- GUI is already zero-copy (one CUDA-VMM block imported into Vulkan; preview at 1 Hz).

**The measured gap (SH3, N=2M, capacity=live):**

| Category (B/splat) | LFS | spirulae L1+FPBO | Δ |
|---|---|---|---|
| Geometry params (fp32 both) | 56 | 56 | 0 |
| SH rest | **192** (fp32 swizzled, incl. 12 B float4 pad) | **90** (16-bit blocks) | **+102** |
| World grads (persistent) | 0 | 0 | 0 |
| Adam state | **172** (u8 + 48 B fp32 per-primitive scales) | **146** (joint (u,log_s)) | **+26** |
| Densify aux | 8 | 12 | −4 |
| **Persistent total** | **428** | **304** | **+124** (~0.25 GB @ 2M) |
| Transient bwd grad helpers (arena, per step) | +40 | ~0 | +40 peak |

So the footprint gap is **two concrete items** — SH-rest quantization and the Adam codec —
not architecture. Meanwhile our per-step speed waste (sort-buffer churn, hard sync,
duplicate reg reductions, bg double-touch) is where we both fix churn *and* gain speed.

---

## North-star gates (every phase measured against these)

| Gate | Baseline | Target |
|---|---|---|
| G1: persistent B/splat (SH3, train) | 428 | **≤ 304** (parity), stretch ≤ 290 |
| G2: allocations per steady-state training iteration (post-warmup) | ~7+ (5 sort + reg temps + clone) | **0** |
| G3: host sync points per iteration | ≥1 hard (`n_instances`) | **0 hard** (async/pipelined only) |
| G4: wall-clock / iteration (7k-iter fixed-seed smoke, 3 runs) | baseline | **strictly ≤ baseline** per task; net improvement per phase |
| G5: peak VRAM during MRNF refine event | up to ~3× largest tensor transient | **≤ 1.1× steady state** |
| G6: quality (PSNR/SSIM on standard eval scenes + exact-trajectory where applicable) | baseline | **no regression** |
| G7: GUI-mode additive VRAM @1080p, 5M max_cap | ~1.2 GB exportable + 384 MiB floor + RTs | **scales with live N; floor removed** |

Phase 0 builds the instrumentation that makes G1–G5 enforceable in CI, spirulae-style
(their `PoolSlots` VRAM category report is the model).

---

## Phase 0 — Measurement harness (prerequisite, small)

| # | Task | Anchors | Done when |
|---|---|---|---|
| 0.1 | **Per-iteration allocation counter**: count every `cudaMalloc`/`cudaMallocAsync`/pool-miss inside a training step; expose `LFS_ASSERT_ZERO_STEADY_ALLOCS` mode that logs (later: fails) on steady-state allocation | `memory_pool.hpp`, `memory_pressure.cpp:278-340`, arena | Counter in VramProfiler + perf log; baseline recorded |
| 0.2 | **Bytes-per-splat ledger** in VRAM HUD/profiler: params / grads / optim / aux / raster-transient buckets, like spirulae's `VramCategory` report | `vram_hud_overlay.cpp`, `VramProfiler` | Ledger matches §0b table on a live run |
| 0.3 | **Bench gate script**: fixed-seed 7k-iter smoke ×3 (headless), records wall, peak VRAM, PSNR, allocs/iter; used as the merge gate for every task below | existing ctest tiers | One command, stable numbers |

---

## Phase 1 — Hot-path speed, FastGS + trainer (no quality impact by construction)

All tasks are removal of dead work/churn; each is independently mergeable behind gate G4.

| # | Task | Evidence | Expected effect |
|---|---|---|---|
| 1.1 | **Persistent high-water sort buffers** in FastGS forward: keys×2, indices×2, CUB WS become grow-only members (spirulae slot-style), sized to max `n_instances` seen; drop the per-step 5× `cudaMallocAsync` + `cudaFreeAsync` | `fastgs/.../src/forward.cu:55`, `:261-300`, `:395-403` | Kills the dominant per-step allocator churn (G2), removes alloc latency + fragmentation |
| 1.2 | **Remove the `n_instances` hard sync**: keep the async D2H but consume it one step late for buffer sizing (high-water + headroom means the sort can launch on a conservative bound), or move range extraction fully device-side | `forward.cu:240-259` (verified `cudaStreamSynchronize`) | Removes the only hard pipeline stall per step (G3); biggest single latency win |
| 1.3 | **Fold reg losses into fused backward**: scale/opacity regularizer *gradients* are already fused into `preprocess_backward_cu`; accumulate their *loss scalars* there too (block reduce + atomic) and delete the loss-only kernels and their per-call `Tensor::empty({num_blocks})+empty({1})` pairs | `regularization.cpp:38-39,76-77,121-122,159-160` (verified), `trainer.cpp:3995-4027`, `kernels_backward.cuh:75-86` | −2 kernels −4 allocs per step |
| 1.4 | **Fuse background compose into `blend_cu`** and accumulate `grad_alpha` in `blend_backward_cu` without the unblend pass | `fast_rasterizer.cpp:476, 571-589, 662-663` | −2 to −3 full-image passes per step |
| 1.5 | **Preflight checks debug-only**: `cudaPointerGetAttributes` ×10 per forward behind `#ifndef NDEBUG` / once-per-topology-change | `rasterization_api.cu:125-180, 254-271` | Host latency per step |
| 1.6 | **Photometric loop hygiene**: drop `reduction_result.clone()` (write into persistent `loss_accumulator_`), skip `grad_img.zero_()` when backward overwrites every pixel, skip redundant `clamp_` when PPISP CRF already clamps | `ssim.cu:1916, 1956-1957`, `trainer.cpp:4389-4392` | −1 alloc, −1–2 full-image touches |
| 1.7 | **Persistent masks**: cache cropbox damping mask (rebuild only on cropbox/topology change) and keep frozen-range mask resident on GPU (no host vector + H2D inside `inject_noise`) | `trainer.cpp:1955-1972, 3936`, `strategy_utils.cpp:109-128`, `mcmc.cpp:664-667` | Removes per-step mask rebuild + H2D |
| 1.8 | **Fuse MRNF/MCMC noise injection** into one kernel (RNG + transform + add, honoring frozen mask); keep every-iteration semantics (quality-neutral) — frequency gating only as a separate opt-in experiment | `mcmc.cpp:641-678, 755-756`, `mrnf.cpp:1057-1078` | Removes `normal_()` full-buffer pass + extra kernel per step |
| 1.9 | **`densification_info` zeroing**: zero in the backward kernel that writes it (or zero only touched rows) instead of a full `[2,N]` memset every iteration | `mcmc.cpp:700-714`, `mrnf.cpp:573` | −1 full-N memset per step |

## Phase 2 — VRAM parity: close the 124 B/splat gap (MRNF + FastGS + optimizer)

This is the spirulae-learning core. Port the *codec designs* (not the code): endpoint-exact
linear block quantization, per-splat-block bounds, `log1p/expm1` 0↔0 fixed point, exact
`decode(0)=0` for anything gradient-shaped.

| # | Task | Design (from spirulae, verified) | Saves | Risk / gate |
|---|---|---|---|---|
| 2.1 | **SH-rest 16-bit block quantization**: store shN packed 2 B/cell + `float2` bounds per 256-splat block; decode in registers in FastGS forward/backward SH evaluation; re-encode in the fused Adam kernel after the update (single writer — same place spirulae's FPBO re-encodes) | `spirulae Tensor.h:1288-1376` (`QuantizedTensor<16,256>`), `EngineForward.cpp:62-108` decode-on-load; integrates with our swizzle or replaces it | **−102 B/splat** (192→~90) — the whole param gap | Largest task. Quality gate G6 on full eval suite; spirulae ships this as default, and 16-bit endpoint-exact is conservative. Fallback flag to fp32 |
| 2.2 | **Joint `(u, log_s)` Adam codec** replacing uint8+per-primitive-scales: 16-bit/cell non-SH, 8-bit/cell SH, `float4` bounds per 256-splat block. Eliminates the 48 B/splat of fp32 scales; better SNR than our per-primitive scale (spirulae measured +20–30 dB on post-step SNR vs linear) | `spirulae Tensor.h:916-1113`; our fused Adam already has the single read-modify-write site (`kernels_backward.cuh:59-96`) | **−26 B/splat**, likely quality-*positive* | Medium. Exact-trajectory won't hold (different rounding); gate on PSNR parity |
| 2.3 | **Reuse one arena workspace for bwd grad helpers** (mean2d/conic/depth/opacity/color) across steps instead of per-frame bump-alloc of fresh 40 B/splat | `rasterization_api.cu:288-293, 532-533` | −40 B/splat *peak* | Low |
| 2.4 | **SH swizzle tail pad**: measure dropping the 3-float float4 pad (192→180 fp32; post-2.1 the equivalent packed saving) vs its coalescing benefit — keep whichever benches faster, memory is secondary here | `sh_layout.cuh:26-46` (verified 12 float4 slots) | −12 B/splat if free | Only if G4 holds |
| 2.5 | (Stretch) **Grad-quant for non-fused paths**: where full fp32 grads still materialize (gsplat path, lazy `get_grad` consumers), use the signed-symmetric `decode(0)=0` codec (16-bit geom / 8-bit SH) | `spirulae GradQuant.cuh:1-68`; `adam_optimizer.cpp:425-447` | up to −175 B/splat on those paths | After 2.1/2.2 |

Post-Phase-2 ledger: 56 + ~90 + 0 + ~146 + 8 = **~300 B/splat** → G1 met, slightly under spirulae.

## Phase 3 — Tensor lib: zero-churn steady state (spirulae's allocator lesson, applied correctly)

We do **not** need spirulae's static-slot architecture globally — a general tensor lib needs a
real allocator. The lesson to port is the *discipline*: the steady-state training loop must
hit the allocator **zero times**, and everything long-lived grows by high-water mark.

| # | Task | Evidence | Notes |
|---|---|---|---|
| 3.1 | **Enforce G2 in CI**: after warmup, a training iteration performs 0 pool misses / driver allocs (counter from 0.1). Every remaining alloc site found becomes a bug with an owner | 0.1 harness | The single most valuable spirulae property |
| 3.2 | **Persistent partials for reductions**: warp-reduce already accepts `partial_buffer` (`tensor_warp_reduce.cu:1051-1059`) — pass it from callers; same for dot & fused-pointwise partials (today: per-call `cudaMallocAsync`, `tensor_dot_optimized.cu:130-133`, `tensor_fused_pointwise.cu:319`) | lfs-tensor-mem §4.3 | Hot-loop reductions stop touching the driver |
| 3.3 | **Eliminate bare `cudaMalloc`s** on op paths: shape/stride metadata (`tensor.cpp:1290-1298`), masking `d_count` (`tensor_masking_ops.cpp:1772`), NaN-check buffers — route through slab pool or immediate params | lfs-tensor-mem §4.3 | Kills untracked classic-heap fragmentation |
| 3.4 | **Slab reclamation**: free fully-empty slabs on `trim_cached_memory` (today slabs only return at shutdown, `gpu_slab_allocator.hpp:283-298`) | lfs-tensor-mem Rank 1 | Steady-state VRAM |
| 3.5 | **Kill the 2× growth spike** in `reserve`/`zeros_direct` replace: since strategies pre-reserve `max_cap`, make any mid-training capacity growth an explicit error/telemetry event rather than a silent double-buffer copy; where growth is legitimate, trim the driver pool immediately after the old free | `tensor.cpp:3365-3432`, `mcmc.cpp:839-877` | Feeds G5 |
| 3.6 | **Lazy machinery out of the hot loop**: audit remaining lazy graphs in the step (scalar loss bookkeeping, mask chains — `trainer.cpp:1768-1882, 5068`); evaluate eagerly or via a single fused kernel. Separately fix D1 double-materialization + snapshot clone-pinning (`preserve_lazy_snapshots_before_write`, `tensor.cpp:386-409`) per the existing D3-campaign design doc | TENSOR_LIB_FINDINGS addendum | Prevents surprise clones of param-sized tensors |
| 3.7 | Empty-tensor 1-byte CUDA sentinels → null-owner path (already exists for `zeros_direct`) | `tensor_unified_ops.cpp:371-379` | Hygiene |

**Correctness dependency (hard):** Phase 2's quantized buffers and Phase 4's index-heavy
paths sit on top of gather/scatter/`index_put_`/`cat`. TENSOR_LIB_FINDINGS still lists ~56
open strided/silent-wrong-result classes; the sites our strategies actually reach
(IGS+ `Tensor::multinomial`, `densification_info.index_select(dim 1)`, sliced-mask
`masked_fill` in `strategy_utils.cpp:199-204`, any slow-path `cat`) must be verified-or-fixed
first or bypassed with the custom kernels of Phase 4. Do not build new packing on unverified
strided ops.

## Phase 4 — MRNF densification: cap peaks, remove syncs

| # | Task | Evidence | Effect |
|---|---|---|---|
| 4.1 | **Kill the ~3× compact pattern**: `compact_splats` does `index_select` → new exact buffer → `reserve(max_cap)` → third buffer, per tensor. Gather directly into a pre-reserved destination (or compact in place via the free-mask) | `mrnf.cpp:932-938` (verified pattern) | G5 |
| 4.2 | **Capacity invariant guard**: slow-path grows set `state.capacity=0`, silently degrading every later grow to `cat` (old+new coexist). Re-`reserve` after any slow path + telemetry when the slow path fires at all | `adam_optimizer.cpp:814-815, 933-934, 1003-1010` | G5, prevents silent churn regression |
| 4.3 | **Reusable densify workspace** for LAS child buffers (means/rot/scale/sh0/shN/opacity, size = worst-case K); grow `densification_info`/score buffers in place via `append_zeros` instead of realloc-zeros each event | `mrnf.cpp:797-806, 441-449`; `mcmc.cpp:750-752` | Event peak + speed |
| 4.4 | **Fused free-slot write**: one kernel writes all attributes + zeros Adam state at target indices, replacing 5-6 `index_put_` launches + duplicate Adam resets (parents zeroed at split *and* again at fill) | `mrnf.cpp:837-842, 1242-1274` | Event speed; avoids Theme-A `index_put_` exposure |
| 4.5 | **One readback per refine**: batch the `sum().item()` / `count_nonzero` / threshold `item_as` host syncs into a single packed D2H (spirulae-style `AsyncReadout` where a 1-event lag is acceptable) | `mrnf.cpp:643, 699, 733, 765` | Removes refine-time pipeline stalls |
| 4.6 | **MRNF: `trim_memory_pool` after refine** (MCMC and IGS+ already do) | `mcmc.cpp:730` vs mrnf | Frees post-event slack |
| 4.7 | **IGS+ `Tensor::multinomial` → fused sample kernel** (MCMC already has one); force-contiguous weights either way | `improved_gs_plus.cpp:417`, findings Theme A | Correctness + speed |
| 4.8 | **Median-by-full-sort → selection/histogram** in edge/error normalization (MRNF + 2× in IGS+) | `mrnf.cpp:310-318`, `improved_gs_plus.cpp:93-115` | Event speed |

## Phase 5 — GUI: pay only for what's on screen

| # | Task | Evidence | Effect |
|---|---|---|---|
| 5.1 | **Grow exportable splat block with live N** (API exists: `growExportableDeviceBlock`) instead of `max(max_cap, min)` at start — a 5M-cap SH3 run reserves ~1.2 GB on step 0 today | `training_manager.cpp:108-124`, `exportable_storage.hpp:58-63` | Largest GUI VRAM item (G7) |
| 5.2 | **Right-size shared scratch**: drop/derive the 384 MiB floor from actual viewport + live-N estimate | `vksplat_viewport_renderer.cpp:931-932, 3192-3194` | Up to 384 MiB on every GUI run |
| 5.3 | **Train-time RT diet**: Main-slot-only + smaller ring while training w/o split/preview; skip depth RT when nothing composites; trim `OutputImagePool` idle on training start (240-tick trim exists) | `vksplat_viewport_renderer.hpp:600-601`, `output_image_pool.hpp:58` | Tens–hundreds of MiB |
| 5.4 | **Wire or delete dead `FramerateController::shouldSkipSceneRender`**; keep 1 Hz training refresh, add idle/minimized suspend | `framerate_controller.cpp:38-76` (never called) | SM contention + dead code |

## Phase 6 — Tensor library deep optimization (second-wave audits, 2026-08-06)

Source: 4 additional Grok audits (`docs/analysis/spirulae-comparison/tl-{kernels,dispatch,lazy,memaxis}.md`),
key claims re-verified. Headline findings: (a) host dispatch for a simple tensor op costs
~5–30 µs — the same band as the CUDA launch — from heap churn and an always-on global IR
lock; (b) the lazy executor does **not** pay for itself on the FastGS hot path; (c) the
trainer retains up to ~650 MiB of mutually-exclusive loss workspaces; (d) fp16 partials are
proven in the fused SSIM path but not deployed to its siblings.

### 6A — Host dispatch overhead (runs on every glue op, every iteration)

| # | Task | Evidence | Effect |
|---|---|---|---|
| 6A.1 | **Share `TensorState` on Tensor copy** (today every copy deep-copies state into a new `make_shared` — "shallow copy" comment notwithstanding); empty tensors without a heap state | `tensor.cpp:636-659, 703`, `tensor_impl.hpp:443` | Biggest structural host win; tens–hundreds of copies/iter |
| 6A.2 | **Gate lazy-IR recording off in production**: every *eager* binary records an IR node under a process-global mutex incl. `shape().str()` string; `lazy_ir_active()` is hard-`true` | `lazy_ir.cpp:189-191, 120-158`, `tensor_impl.hpp:885-886` | Removes an always-on lock+string per op; keep for tests/debug |
| 6A.3 | **Contiguous same-shape same-dtype binary fast path**: validate → empty → launch, skipping `BinaryExpr`/`TensorLeaf` (2 Tensor heap cells + 2 copies per op today) | `tensor_impl.hpp:859-887`, `tensor.cpp:324-338` | ~1–2× launch-cost host time saved per binary |
| 6A.4 | **Inline small-vector shapes/strides** (rank ≤ 8 array + rank), stop `shape.strides()` allocating per call | `tensor_impl.hpp:208-209, 236-246, 445` | Kills 4–8 heap vectors per op |
| 6A.5 | **In-place loss accumulation at call sites** (`loss.add_(tile)` instead of `loss = loss + tile`), `has_lazy_expr()` from local flag not mutex map | `trainer.cpp` loss glue, `tensor_impl.hpp:1382-1384` | Fewer full dispatches |

### 6B — Lazy executor: narrow to where it wins

Verdict from audit: keep the *fusions* (unary→reduce saves a kernel **and** a full-tensor
intermediate), drop the blanket deferral.

| # | Task | Evidence |
|---|---|---|
| 6B.1 | **Defer only fusable patterns** (unary chain ≥2 or producer→reduce); single unaries (sigmoid/exp/sign/pow) at ≥4 KB currently pay defer+3 locks+snapshot then materialize immediately with zero fusion — worst of both worlds (gsplat path pays every forward) | `tensor_expr_impl.hpp:34-46`, `splat_data.cpp:625-640` |
| 6B.2 | **CHW fused absdiff-mean kernel** for the densify L1 error map — current `mean({0})` misses fusion (only last-dim/full reduces fuse) and holds a full-image abs intermediate | `trainer.cpp:4998-5002`, `tensor_unified_ops.cpp:1210-1220` |
| 6B.3 | **Narrow COW snapshots**: skip snapshot when the operand is an exclusively-owned temporary; today a pending fusion snapshot on live param storage can trigger a full param clone on mutation | `tensor.cpp:351-409` |
| 6B.4 | Make `pow` fusable (mask path builds a dead-end deferred node today) | `tensor_impl.hpp:1885` |

### 6C — Kernel-level (weighted by training relevance)

| # | Task | Evidence | Relevance |
|---|---|---|---|
| 6C.1 | **Binary–binary(–reduce) fusion**: `a.mul(b).add(c).sum()` = 3-4 launches + 2 full temporaries today; only unary/scalar chains fuse | `tensor_expr_impl.hpp:560-582` | High (loss/strategy glue) |
| 6C.2 | **Wire the dead Channel3D broadcast kernels** — coalesced + shared-mem variants are implemented but the launcher only calls the slow per-pixel one | `tensor_broadcast_ops.cuh:439-669` vs `:999-1001` | Medium |
| 6C.3 | **SM-capped grid-stride for vectorized elementwise** (today 1 float4/thread, huge grids) + lower the `n>1024` Thrust cutoff if it benches faster | `tensor_vectorized_ops.cuh:33-94` | Medium |
| 6C.4 | **Vectorize generic broadcast + same-shape early-out**; `where` host must stop cloning matched-shape operands (kernel is already shape-aware — host side is the bug) | `tensor_broadcast_ops.cuh:741-799`, `tensor_unified_ops.cpp:1866-1881` | Medium |
| 6C.5 | Device-side mean/prod finalize (drop 1-element Thrust launches and the host-round-trip prod); float4 compares & count_nonzero | `tensor_warp_reduce.cu:663-669`, `tensor_ops.cu:718-724` | Low-medium |
| 6C.6 | **Optimize the hand-rolled SGEMM — no cuBLAS dependency** (explicit decision): grow the register-tile config (BM/BN/BK/TM/TN) per-arch, add vectorized float4 global loads + double-buffered smem in the 64-tile kernel, optional TF32 `mma` path behind the same API, autotune tile pick at init. Matmul is **cold in the 3DGS hot loop** — do only if profiling shows GEMM time | `tensor_matrix_ops.cu:42-105, 371-405` | Low (deprioritized) |
| 6C.7 | `half2`/Packed128 vectorized fp16 elementwise (infrastructure exists, unused) — prerequisite for 6D fp16 wins to also be *fast* | `packed128.cuh:165-167`, `tensor_generic_ops.cuh:67-76` | Tied to 6D |

### 6D — Memory levers inside the lib + trainer workspaces

| # | Task | Saves @1080p | Evidence | Risk |
|---|---|---|---|---|
| 6D.1 | **Union the mutually-exclusive loss workspaces** (fused / decoupled / masked / masked-decoupled / pure-SSIM) into one arena region — they are never live in the same step but are retained forever once touched | **100–650 MiB peak** | `trainer.hpp:539-542`, sizes in tl-memaxis §5 | None (same kernels) |
| 6D.2 | **Delete `zero_terms`**: a full 23.7 MiB tensor of zeros passed as "unused" SSIM partials to the decoupled backward — kernel flag instead | 23.7 MiB (×2 with masked) | `ssim.cuh:206-224`, `ssim.cu:2087-2088` | None |
| 6D.3 | **Port fp16 `dm_*` partials to decoupled/masked/pure-SSIM workspaces** (fused path already ships fp16 partials — proven) | ~47–130 MiB, likely *faster* (½ bandwidth) | `ssim.cuh:220-223, 281-283, 339-342` | Low |
| 6D.4 | **General `out=`/destination API** for binary/reduce/clamp/where (pattern exists: `index_select_into`, NN `_out`) + adopt in losses/regularization/strategy ratios | Steady-state alloc churn → 0 (feeds G2) | `tensor_impl.hpp:2457`, `regularization.cpp:38-39`, `mcmc.cpp:267` | Low |
| 6D.5 | **Fill Float16 API holes** (reductions assert, no half unary/clamp/fill/in-place/broadcast_to) as the gateway to fp16 intermediates; BFloat16 does not exist — defer bf16 until fp16 saturates | enables 6D.3 + error-map fp16 | `tensor_ops.cu:1366-1386`, tl-memaxis §1.3 | Medium |
| 6D.6 | fp16 strategy score buffers (`_error_score_max`, optionally densify info) | ~8–15 MiB @2M | `mcmc.cpp:146-154` | Validate sampling quality |

**Do-not-do (from the audits):** don't re-enable the in-house warp full-reduce for scalars
(CUB is 3–7× faster per in-code benchmark note); don't remove the transpose-then-reduce
path without a strided kernel that actually beats it; don't push image *gradients* to fp16
as a memory play (quality risk) — partials only.

## Phase 7 — Beat spirulae on speed (they left this on the table)

Only after G2/G3 hold (stable pointers + no steady-state allocs are prerequisites):

| # | Task | Why we win |
|---|---|---|
| 7.1 | **CUDA graph capture** of the steady-state step (spirulae: none). Zero-alloc + persistent buffers from Phases 1/3 make the step graph-capturable; densify/eval iterations run un-graphed | Launch overhead ~0 for 200+ kernels/step |
| 7.2 | **Multi-stream overlap**: GT decode/upload + loss readbacks on a copy stream overlapping compute (spirulae is single-stream, sync H2D) | Hides H2D entirely |
| 7.3 | (Experiment) **Ellipse-exact tile intersection** vs our AABB tiles — spirulae's ellipse test yields fewer instances → less sort + blend work | `spirulae IntersectTile.cu:54-89` |

---

## Execution order & dependencies

```
Phase 0 ──► Phase 1 (independent tasks, land continuously)
   │            │
   │            ▼
   ├──► Phase 3.1-3.3 (zero-churn steady state)  ──► Phase 6 (tensor lib deep-dive) ──► Phase 7 (graphs/streams)
   │            │
   ▼            ▼
 strided-op verification (findings gate) ──► Phase 2 (quantization) ──► G1 parity
   │
   ▼
 Phase 4 (densify) — 4.1/4.2 early (pure wins), 4.4+ after fused-kernel infra
 Phase 5 (GUI) — independent, any time
```

Recommended first wave (max win / effort, all quality-neutral): **1.1, 1.2, 1.3, 1.5, 0.1–0.3,
5.1, 4.1, 4.2**. Second wave: 2.2 (Adam codec), 1.4, 1.6–1.9, 3.2–3.4. Third wave: 2.1
(SH quant — the big one), 4.3–4.8, 6A (dispatch), 6D.1–6D.3 (workspace union, zero_terms, fp16 partials), 7.1–7.2.

## What we deliberately do NOT copy from spirulae

- **Exact-size grow** (their pool frees+mallocs on every new high-water) — our 1.5× amortized
  growth is strictly better; keep it.
- **Global static slot registry** for the tensor lib — right for an engine, wrong for a
  general tensor library; we enforce the *invariant* (zero steady-state allocs) instead.
- **Single-stream, no-graph execution** — that's their ceiling, our headroom.
- **Full-capacity VRAM from step 0** as a *forced* policy — we keep `max_cap` pre-reserve
  (it's what makes densify in-place) but GUI-exportable and optional buffers grow with live N.

---

## Phase 8 — SPZ takeaway: per-band SH bit budgets (bonus, 2026-08-06)

Source: `docs/analysis/spirulae-comparison/spz-v4-analysis.md` (deep-dive of
github.com/nianticlabs/spz). Decision: SPZ's disk-format ideas (v4 library upgrade,
checkpoint container, export presets) are explicitly OUT of scope — only the VRAM-relevant
insight is kept.

**8.1 — Per-band SH bit budgets inside Phase-2 block quantization [vram].**
SPZ allocates more bits to SH band 1 than bands 2+ (5 vs 4 in their disk codec — matching
SH energy decay: higher bands carry less energy, so they tolerate fewer bits). Apply the
*insight*, not their bit depths: after Phase 2.1 lands with uniform 16-bit SH blocks, run a
gated A/B giving degree-1 coefficients 16 bits and degrees 2+ e.g. 12 bits (packed), saving
a further ~20 B/splat at SH3 on top of the 90 B quantized budget. Strictly behind gate G6
(no PSNR/SSIM regression on the eval suite) and gate G4 (decode cost must not slow the
SH evaluation). Evidence for the band split in SPZ: `spz load-spz.cc:412-417`.

Non-goals (explicit): upgrading `external/spz`, SPZ-style checkpoint container, .spz export
changes, SPZ codecs for any training state (their 4-5-bit SH / u8 log-scales / u8 opacity
are training-unsafe — see the report's quality-risk table).
