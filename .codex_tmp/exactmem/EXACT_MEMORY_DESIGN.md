# Exact-Memory Contracts: THE DESIGN

Status: normative design proposal. Branch `lfs-elite`, base HEAD `2869f6d27`.
Companion doctrine: `TRAINING_IMAGE_PIPELINE_SPEC.md`. Measurement protocol:
`.codex_tmp/vramgt/round12_report.md` (+ rounds 13/14).
Inputs: 4-lane census (`fastgs.out.md`, `tensorpool.out.md`, `viewer.out.md`,
`training.out.md`) plus direct reading of every cited site.

Owner law: **no memory-size estimates, forecasts, or growth factors anywhere.
Every GPU workspace allocates its exact measured requirement and shrinks to the
current need at safe boundaries.**

---

## 0. Where the bytes actually are (measured, not assumed)

Baseline at HEAD `2869f6d27` (round 14, bicycle 7k mcmc headless, median of 3,
nvImageCodec device-allocator hook OFF):

| Metric | Value |
|---|---:|
| process-net VRAM peak | **1422 MiB** |
| doctrine gate | <= 1386 MiB |
| deficit to close | **36 MiB** |
| wall | 28.9969 s |
| steady dataloader wait | 0.025772 ms/iter |
| steady allocations/iter | 0.231471 |
| pinned active+cached | 0.0096 MiB |

Per-owner ledger rows, from `.codex_tmp/vramgt/round12/gate1/perf_bench.json`
(round 12 tree, net 1600 MiB that run; **re-measure at HEAD before trusting the
absolute values**, the ranking is what matters):

| ledger row | MiB | class |
|---|---:|---|
| `baseline_cuda_context` | 637.9 | not ours |
| `io.external_codec_and_bucketed` | 293.3 | decoder internals (see R7) |
| `unattributed_residual` | 140.3 | **nobody owns this yet** |
| `fastgs_raster_live` | 122.4 | arena, already exact |
| `fastgs_sort_hwm` | **107.0** | 2.25x compounded forecast |
| `training_state` | 378.0 | model + optimizer, live |
| `training_state_capacity_overhead` | **58.7** | 1.5x capacity slack |
| `loss_workspace_arena` | 52.4 | union-of-5 + 25% |
| `io.decoded_frame_ring` | 32.0 | adaptive ring, bounded |
| `pool_bucket_cache` | 10.0 | allocator reuse cache |

Two rows carry the whole deficit: **FastGS sort (107 MiB at 2.25x) and training
state capacity overhead (58.7 MiB at 1.5x)**. Either one alone closes the 36 MiB
gap. That is why the phase order below is what it is.

---

## 1. What the law means operationally

Not every retained byte is the same offence. Collapsing them into one rule is how
this campaign would produce a perf disaster. Three classes:

**Class A: inflates the peak. Kill unconditionally.**
A multiplier, an additive percentage, a forecast from a previous iteration, or a
floor/cap above the measured need. At the instant of peak demand the buffer holds
`factor x required` when only `required` was ever touched. FastGS 2.25x,
gsplat +25%, viewer +12.5%/+50%/+50% and its 128 MiB floor, Adam 1.5x, model
1.5x, loss arena +25%, densify +20%, MRNF x3/2, MCMC 1.05x, the fixed 4M wave cap.

**Class B: does not inflate the peak, inflates steady residency. Bound with a shrink boundary.**
A grow-only high-water of *exact measured* values. At the peak instant the buffer
is exactly as large as that iteration genuinely required, so it costs nothing on
the judged metric; it only holds memory *after* the peak has passed. Legal
between boundaries, illegal across one. Examples: `no_shrink=true` Vulkan
buffers, `ensureCudaInteropBuffer` retention, output-image pool idle entries,
allocator reuse caches, PPISP max-HxW buffers.

**Class C: bounded quantization. Keep, but justify and measure.**
Allocator bucket grid, VMM granularity, storage-buffer offset alignment. These
are not predictions; they are properties of the hardware or of the allocator that
serves the request. They are legal **only** while the waste is measured and
disclosed per workspace (see section 4), because a coarse grid can silently
become a worse multiplier than the factor it replaced (R4).

The doctrine sentence maps onto this exactly: *"allocates its exact measured
requirement"* is Class A; *"shrinks to current need at safe boundaries"* is
Class B; and Class C is what remains after both.

---

## 2. The contract (five rules, testable)

**EXACT-1 (sizing).** The allocation size is a pure function of values already
measured: element counts already resolved on the host, layout math over those
counts, or an allocator/library query (`cub::Device*::…(nullptr, bytes, …)`,
`cuMemGetAllocationGranularity`). No multiplication by a constant > 1, no
additive slack, no value derived from a previous iteration, no floor above the
measured requirement.

**EXACT-2 (reallocation trigger).** A workspace reallocates only when the
measured requirement exceeds its current size. A smaller requirement never
triggers a reallocation inside a phase; that is EXACT-3's job. This is what keeps
`steady_allocs_per_iter` near zero without any slack factor.

**EXACT-3 (shrink boundary).** At each named boundary (section 5) every workspace
either resizes to the current measured requirement or releases entirely, and the
serving pool is trimmed exactly once. Boundaries are points where no GPU work is
in flight for that workspace and the cost is amortized. Trimming is never on the
hot path: `CudaMemoryPool::trim_cached_memory()` performs a full
`cudaDeviceSynchronize`.

**EXACT-4 (retention concentrates in the pool).** A workspace does not keep a
private *reserve* (bytes beyond its measured requirement) so that a future larger
request is cheap. Reuse of surplus is the serving pool's job, because the pool
has a cap, a trim, a pressure hook, and a ledger row, and because one shared
cache is smaller than N private hoards.

**EXACT-4 does not mean "free it every iteration."** A workspace whose exact
requirement recurs every iteration keeps its own exact block under EXACT-2 and
releases it at a boundary under EXACT-3. Cycling such a workspace through the
pool's free list instead converts a bounded private allocation into unbounded
shared cache pressure: the block is not reusable by anyone else (it is requested
again microseconds later), so the cache holds it *plus* the entries for every
neighbouring size the jitter touched. **Measured in phase 1** (section 8):
freeing the FastGS sort buffers every forward removed 44 MiB of forecast slack
and added 240 MiB of `pool_bucket_cache`, saturating its budget ceiling, for a
net **+134 MiB regression** and a doubled allocation rate. Retention moves to the
pool only for allocations that are genuinely transient across consumers.
Corollary: the pool's own cache budget is part of the gate (R4), not free space.

**EXACT-5 (disclosure).** Every workspace publishes the pair
`(required_bytes, allocated_bytes)` for its current state. A steady-state
difference beyond documented Class C quantization is a defect, and the perf-bench
ledger must show it. A workspace that cannot state its required bytes cannot be
audited and is not compliant.

### How per-iteration jitter is handled, with no slack factor

Three jitter regimes, three mechanisms, none of them a multiplier:

1. **Constant in steady state** (loss arena at fixed resolution, per-primitive
   buffers between densifications): allocate exact once under EXACT-1, zero
   churn thereafter under EXACT-2.
2. **Jitters per iteration around a stable envelope** (FastGS `n_instances`
   varies with the sampled view): allocate the exact measured requirement; keep
   it under EXACT-2 while the requirement is smaller; release at the boundary
   under EXACT-3. The retained size is the exact peak requirement, which the run
   genuinely needed at some iteration, so process-net peak is unchanged. This is
   the crucial difference from a forecast: a forecast raises the peak itself,
   an exact high-water cannot.
3. **Jitters with no envelope** (viewer resize, LOD residency): allocate exact
   per event and let the serving pool's reuse cache absorb the churn under
   EXACT-4, with an idle/settle boundary under EXACT-3.

Consolidation rule (follows from EXACT-4 + Class C): where several buffers scale
with the same measured count, allocate **one** block and sub-slice it. Five
buffers of ~10 MiB each pay the allocator's grid five times; one 50 MiB block
pays it once, and the relative waste falls by roughly the number of buffers.
Above 16 MiB the `SizeBucketedPool` grid is 16 MiB wide, so a 17 MiB request
becomes 32 MiB, an effective 1.9x. Consolidation is the difference between the
exactness holding and it inverting at scale. **Consolidate only when serving
from the exact tier** (`ExactAsync` / non-bucketed exact path). Consolidating
into a block that is then rounded by `SizeBucketedPool`'s coarse grid
re-introduces Class A-like waste at scale; consolidation is an exact-tier tool.

---

## 3. Per-subsystem contracts

### S1. FastGS sort workspace (`fastgs/rasterization/src/forward.cu`)

*Today:* `kSortBufferGrowthFactor = 1.5` is applied twice. `predicted =
last_n_instances * 1.5` (forward.cu:635-639 at base HEAD) feeds
`ensure_instance_capacity`, which calls `ensure(bytes)`, which multiplies by 1.5
again. Retained = **2.25x** the largest instance count ever seen, grow-only,
freed only by an explicit release. Measured 107.0 MiB. The buffers bypass
`CudaMemoryPool` entirely (raw `cudaMallocAsync`), and every growth runs a
device-wide `cudaDeviceSynchronize` inside `reset()`.

*Requirement:* `n_instances`, the exact device-resolved count at
`offset[n_primitives-1]`, already copied to a pinned host slot with an event.
Bytes = `n * (2*sizeof(InstanceKey) + 2*sizeof(uint))` (InstanceKey is
`uint32_t`, so 16 B/instance) plus the exact CUB `SortPairs` temp-storage query
for that `n`.

*Contract:*
- Delete `kSortBufferGrowthFactor` and both applications. No multiplier anywhere
  in this file.
- Keep the existing async D2H + event. Two admissible shapes, in preference
  order:
  - **(a) preferred, zero perf risk.** Keep the optimistic `create_instances`
    launch, but clamp it to the workspace's **current exact capacity** (the
    high-water of previously measured `n_instances`), not to a prediction. Then
    `cudaEventSynchronize`, read the exact `n`, and if `n > capacity` take the
    existing overflow re-run (drain, grow to exactly `n`, re-run). Overflow now
    fires only when `n` sets a new record, which after warmup is rare and is
    already counted by `n_instances_fallback_sync_count()`. Exact sizing, and the
    host-latency overlap the current code was built for is preserved.
  - **(b) fallback.** Move `cudaEventSynchronize` before `create_instances`,
    allocate exactly `n`, run one path. Simpler (the overflow path, the
    `force_sync` special case and `last_n_instances` all disappear) but it
    exposes a GPU bubble of roughly one launch round trip per iteration. Only
    admissible if the 2% wall gate passes. **This is the shape the in-flight
    Codex phase 1 has implemented; see section 8.**
- **Consolidate** the four instance buffers plus the CUB workspace into one
  block with internal offsets. Non-negotiable at scale (R4).
- Serve from `CudaMemoryPool` (`allocate_cuda_storage`, Pooled). This also
  deletes the per-growth `cudaDeviceSynchronize`, moves the bytes into the pool
  ledger, and lets memory-pressure recovery see them.
- Preserve the teardown guards: `gpu_process_teardown_started()` on the release
  path, and `safe_cuda_pool_deallocate` in any destructor that can run at static
  or TLS teardown.
- Lifetime: the selected sorted-index slice must outlive backward.
  `release_sorted_primitive_indices` becomes a real stream-ordered free at the
  existing post-backward call site (`rasterization_api.cu:34`), and the
  non-selected half of the CUB double buffer is freed at forward exit. With
  consolidation this is one block released once.

*Reallocation:* only when measured `n` exceeds the current block (EXACT-2).
*Shrink boundary:* B1, B3, B6 via `release_sort_workspace_buffers()` (already
exists and is already wired at `fast_rasterizer.cpp:861`).
*Jitter:* regime 2. Same-size re-requests are bucket-cache hits and cost zero
driver allocations, so the 0.3 allocs/iter gate is unaffected (confirmed:
`alloc_counter` is documented to count only true driver commits, and
`SizeBucketedPool::try_allocate_cached` returns before `record_site`).

*Expected:* 107.0 MiB -> roughly 47 to 50 MiB, i.e. **-55 to -60 MiB**, minus
whatever `pool_bucket_cache` grows by (must be measured, R4).

### S2. gsplat sort / CUB / color-grad caches (`rasterization/gsplat/Common.cpp`, `Intersect.cpp`)

*Today:* `growth_capacity_bytes(required) = required + required/4` and
`growth_capacity(required) = required + required/4`, applied to CUB workspace,
color-gradient workspace, cumulative-tile buffer and the isect/flatten id
buffers. Grow-only, thread-local, served by `DirectDeviceBuffer` /
`StreamOrderedDeviceBuffer` rather than the pool.

*Requirement:* exact CUB query bytes; `n_isects`, `n_elements`, `C*N*channels`
which are all already exact host values.

*Contract:* delete both `+25%` helpers; allocate the measured value; serve from
`CudaMemoryPool`; consolidate the isect/flatten pairs. Same EXACT-2/EXACT-3
treatment as S1.

*Note:* gsplat is not the default rasterizer, so this does **not** move the
bicycle gate. It is cheap, low risk, and removes a second copy of the same
disease before it is cargo-culted. Schedule it after the gate is already passing.

### S3. Loss workspace arena (`training/kernels/ssim.cu`)

*Today:* `LossWorkspaceArena::ensure_capacity` sizes to `max_variant_bytes(shape)`,
the maximum over **all five** layouts (fused, pure SSIM, decoupled, masked x2),
regardless of which loss configuration is actually active, then grows with
`capacity + capacity/4`. Measured 52.4 MiB.

*Requirement:* the layout size of the **active** variant for the current image
shape, from the same layout oracle the `ensure_*` entry points already use.

*Contract:* size to the active variant only; delete the +25%; keep the 256 B
alignment (Class C). A configuration change (mask toggled on, resolution change)
is a legal reallocation trigger under EXACT-2 and is not a hot-path event.

*Reallocation:* on active-variant change or shape change.
*Shrink boundary:* B2, B3.
*Expected:* -10.5 MiB from the multiplier alone; the union removal is worth
another -10 to -20 MiB depending on which variant is active for the gate run.

### S4. Training state: model + optimizer capacity

*Today:* `SplatExportableStorage::growthCapacity` returns `live + live/2`,
clamped to `max_capacity`; `AdamOptimizer::compute_new_capacity` uses
`growth_factor = 1.5f`; strategies additionally pre-size to `max_cap` at init
(`zeros_direct(..., max_cap)`, `allocate_gradients(max_cap)`, free masks, score
buffers, `_ones_int32`). Measured overhead at peak: **58.7 MiB**, and the
pre-size to `max_cap` is a **startup spike**, explicitly a violation under
doctrine section 5.

*Requirement:* live primitive count `N` after each densify/prune, and the exact
per-region SoA layout for that `N`.

*Contract:*
- Capacity tracks live `N` exactly. `growthCapacity` is deleted, not
  re-parameterized; `AdamConfig::growth_factor` is removed from the config
  surface rather than defaulted to 1.0, so it cannot be reintroduced.
- Strategy initialization sizes to the initial `N`, not `max_cap`. `max_cap`
  remains a *policy* limit on population, never an allocation size.
- **VA reservation stays at `max_cap`.** Reserving virtual address space costs no
  physical memory and is not a forecast in the sense the law prohibits; it is what
  makes an in-place physical grow possible at all
  (`growExportableDeviceBlock` hard-fails past `reserved_size`). Keep the
  reservation, commit only the live requirement.

*Reallocation:* at densify/prune completion (B1), which is every 100 iterations
under MCMC, not per iteration.

*The transient hazard, and why this phase is fenced:* `growExportableDeviceBlock`
does not append granules under the reserved VA. It runs `cuMemCreate` for the
**whole** new size, maps it to a temporary VA, `cudaMemcpy` D2D, `cudaDeviceSynchronize`,
then releases the old physical. Peak during a grow is therefore `old + new`.
Making capacity track live `N` exactly multiplies the number of these events by
roughly `1 / log(1.5)` and each one is a full model+optimizer copy. Two possible
outcomes, both must be measured before the phase is accepted:
- the extra copies cost less than 2% wall and the transient `old + new` stays
  below the current peak (because `new` is now only marginally bigger than
  `old`), in which case the phase lands as-is; or
- the transient dominates, in which case the correct fix is **per-region
  physical commit under a per-region VA sub-reservation**: each SoA region gets
  its own VA range sized at `max_cap`, physical commit tracks live `N` per
  region, growth appends granules with no copy and no offset relocation, and
  shrink unmaps granules. That removes the copy, the transient, the region
  relocation, and the `rebindSplatData` churn in one move. It is a large change
  (each region becomes its own shareable handle, so the Vulkan side binds one
  buffer per region instead of one buffer with offsets) and belongs in its own
  campaign with owner sign-off. Do not start it inside this one.

*Shrink boundary:* B1 (after prune), B2, B3.
*Expected:* -58.7 MiB, and removal of the startup spike.

### S5. Rasterizer memory arena (`core/cuda/memory_arena.cu`)

*Today:* the per-frame bump allocation itself is already exact
(`required<PerPrimitiveBuffers>(n)` walks the real layout). The **commit** policy
is not: `initial_commit = 128 MiB` regardless of demand; fallback init
`min(initial_commit, free/2, 256 MiB)` with a 64 MiB floor and a halving retry
loop; VMM growth commits `growth_needed * 2` on the first three retries and
`growth_needed * 3/2` afterwards; the non-VMM `grow_arena` uses
`max(required*2, capacity*1.5)` rounded up to 128 MiB; a 200 MiB
`MIN_FREE_BUFFER` and a `min(1 GiB, total/10)` untouchable reserve gate the
commit; `decommit_unused_memory` deliberately keeps the first chunk forever and
runs only on emergency paths.

*Requirement:* `total_needed - committed`, already computed exactly at the call
site.

*Contract:* commit `align_up(growth_needed, granularity)`, nothing more. Initial
commit is the first frame's measured requirement, not 128 MiB. The free-memory
gates stay (they are OOM protection, not sizing) but must not silently enlarge
the commit. `decommit_unused_memory` becomes reachable from B1/B3/B6, not only
from emergency teardown, and stops exempting chunk 0 when the arena offset is 0.

*Shrink boundary:* B1, B3, B6.
*Note:* granularity rounding is Class C and stays.

### S6. Viewer shared scratch arena + render targets (`visualizer/rendering/`, `rendering/rasterizer/vulkan/`)

*Today, compounding on one path:* `estimateSharedScratchBytes` computes an exact
layout but over 64-px-bucketed pixel/tile extents; `ensureSharedScratchArena`
then adds `required/8` on first allocation or `required/2` when growing, and
floors the result at **128 MiB**; the arena's grow callback `make_grow_fn`
independently asks for `need + need/2`. A grow arriving through the arena
therefore requests up to **2.25x** the measured layout, and because it goes
through `growExportableDeviceBlock` the instantaneous cost is `old + 2.25x new`.
Separately, the virtual reservation is sized from **total device memory**
(`cudaMemGetInfo` total, not free), `growRegionCapacity` adds
`max(target/4, 8 MiB)` or `max(target/2, 16 KiB)`, readback staging grows 1.5x,
several Vulkan helpers double capacity until it fits, `resizeDeviceBuffer`
defaults to `no_shrink=true`, and the sort region is floored at the fixed
`HIGS_DEPTH_WAVE_INSTANCES = 4194304` (4 M x `int32` keys and indices, roughly
100 MiB of scratch that exists whether or not the scene needs it).

*Requirement:* the layout sum over this frame's `num_splats`, `visible_capacity`,
logical tile grid and logical pixel count; and for the sort region, the measured
tile-instance count for the frame.

*Contract:*
- Delete all three slack applications (`required/8`, `required/2`,
  `need + need/2`) and the 128 MiB floor. Commit exactly the layout sum, page
  aligned to the VMM granularity only (Class C).
- Keep the VA reservation but size it from a declared maximum scene bound, not
  from total device memory.
- `growRegionCapacity`, the readback 1.5x, and the `capacity *= 2` helpers all
  become exact.
- `no_shrink=true` stops being the default. Shrinking a Vulkan buffer requires
  retiring it behind the frame timeline; that machinery already exists for
  swapchain and interop resources and must be reused (R6).
- The 64-px extent bucket is Class C **only if** it is measured: it is the
  difference between an interop-image recreate storm during a drag-resize and a
  bounded overshoot. Keep it, publish the waste, and revisit with a settle
  boundary (B5) rather than deleting it blind.
- `HIGS_DEPTH_WAVE_INSTANCES` is Class A and is the single largest viewer item,
  but shrinking it is a correctness change (overflow produces visible artifacts,
  not an error). It gets its own phase with a visual A/B.

*Shrink boundary:* B4 (scene change), B5 (resize settle), B6, B7 (idle).
*Gate:* the viewer is not exercised by the headless bicycle protocol. Every
viewer phase needs a GUI run: load a scene, drag-resize, toggle overlays and
selection, watch `vram.audit.shared_scratch.capacity` and the process VRAM HUD,
and confirm zero validation errors.

### S7. Image loader (decoded-frame ring, pinned staging, decoder internals)

Already redesigned by the WO-VRAMGT campaign and governed by
`TRAINING_IMAGE_PIPELINE_SPEC.md` sections 2 and 6: the ring is adaptive with
hard clamps `[2, 12]`, pinned staging is a small fixed ring, and the host cache
holds compressed bytes only. That controller is **feedback on measured latency**,
not a memory forecast, and is compliant as written.

The open item is `io.external_codec_and_bucketed` at 293.3 MiB, which is
nvImageCodec's own device allocations. The budgeted `cudaMemPool_t` hook exists
and its SIGSEGV was root-caused and fixed in `d786be384` (vendored decoders
dereferenced `pinned_allocator` while checking only `device_allocator`), but
round 14 **disabled it by default** because enabling it measured *worse*:
1491 MiB net and 0.092 ms dl_wait versus 1422 MiB and 0.026 ms with it off. Do
not re-enable it as part of this campaign on the assumption that a budgeted pool
must be smaller. It is an owner decision with a measurement attached (R7).

### S8. The serving pools themselves

EXACT-4 routes more traffic into `CudaMemoryPool`, so the pool's own sizing
behavior becomes part of the contract rather than an implementation detail. It
is all Class C, and all of it must be budgeted and disclosed:

| tier | quantization | worst-case relative waste | contract |
|---|---|---:|---|
| `GPUSlabAllocator` (<= 256 KiB) | power-of-two size classes; slab commit `clamp(block*1024, 256 KiB, 8 MiB)` | just under 2x per block | keep: exact per-block allocation would replace one slab commit with hundreds of driver allocations. Budget it; `reclaim_empty_slabs()` already exists and must run at boundaries. |
| `SizeBucketedPool` (256 KiB to 16 GiB) | fixed grid: 256 KiB / 1 MiB / 16 MiB / 64 MiB / 256 MiB / 1 GiB steps | ~2x just above a step boundary | keep, but consolidate requests so the total lands well inside a step (R3). Any workspace that parks just above a boundary goes to the async tier instead. |
| async tier (`cudaMallocAsync`) | driver granularity only | negligible | this is the **exact** tier. It is currently reachable only when the bucket path fails or the request exceeds 16 GiB. It needs an explicit opt-in so a workspace can declare "my size is measured and I want it verbatim". |
| driver pool release threshold | 64 MiB retained free | 64 MiB absolute | keep (it is what makes exact re-request cheap), but count it: it is retention, and it belongs in the ledger next to `pool_bucket_cache`. |
| `PinnedMemoryAllocator` | 512 B steps below 1 MiB, power-of-two above; 1 GiB cache budget | ~2x above 1 MiB | host memory, outside the VRAM gate, but the 1 GiB ceiling is a Class B retention with no boundary. Measured steady is 0.0096 MiB, so this is a latent risk, not a live one. |

The one structural change this implies: **`CudaStorageMode` needs a third value**
(exact/async) so a caller can bypass bucket quantization for a large measured
allocation. Without it, EXACT-1 at the call site can be silently undone by the
allocator, and R3 has no fix other than padding, which is the thing the law
forbids.

---

## 4. Enforcement: the disclosure gauge

EXACT-5 is the only rule that prevents this campaign from decaying. Concretely:

- Every workspace covered above publishes two ledger values, not one:
  `<name>.required_bytes` and `<name>.allocated_bytes`. `fastgs_sort_hwm` already
  publishes the second; it must publish the first.
- perf-bench computes `slack = allocated - required` per workspace and a total.
  A steady-state total slack above the documented Class C quantization budget is
  a **gate failure**, in the same tier as the process-net peak.
- `pool_bucket_cache` and the driver pool's `ReservedMemCurrent - UsedMemCurrent`
  are part of that budget, not free space. Moving retention into the pool
  (EXACT-4) is only a win if the pool's own retention is on the scoreboard.
- A unit test per workspace asserts `allocated_bytes == required_bytes` modulo
  the declared quantization, at two different measured sizes, so a reintroduced
  multiplier fails a test rather than a 30-minute gate run.
- `unattributed_residual` (140.3 MiB) must reach approximately zero. Until it
  does, any claim that a phase "saved N MiB" is unfalsifiable, because the
  residual is larger than every individual saving in this plan.

---

## 5. Named shrink boundaries

| id | boundary | who runs it | what happens |
|---|---|---|---|
| B1 | densify / prune / refine completion | training thread | S1, S4, S5 resize to current measured need |
| B2 | evaluation or checkpoint save | training thread | S3, S4 resize; pool trim |
| B3 | training pause / stop / completion | training thread | all training workspaces release; pool trim |
| B4 | scene load / clear / project switch | viewer thread | S6 releases; interop buffers released; pool trim |
| B5 | viewport resize settle (debounced) | viewer thread | S6 render targets and scratch resize to logical extent |
| B6 | OOM recovery / memory-pressure signal | pressure handler | every workspace releases what is idle; pool trim |
| B7 | viewer idle (no frame submitted for a settle interval) | viewer thread | output-image pool and staging rings drain |

Rules for boundaries:
- Exactly one `CudaMemoryPool::trim_cached_memory()` per boundary crossing. It
  performs a device-wide synchronize; calling it per iteration or from a worker
  thread mid-training is a defect. `improved_gs_plus.cpp:819` calls it from a
  training path and must be checked against this rule.
- A boundary must not run while any workspace it touches has GPU work in flight.
  For Vulkan resources that means behind the frame timeline, not merely behind a
  CUDA stream sync.
- No cache may require an OOM to shrink (doctrine section 7). B6 is a safety net,
  never the primary mechanism.

---

## 6. Phase plan

Every phase, without exception:
- 3-run perf gate, bicycle 7k mcmc headless with `--perf-bench`, medians reported;
- wall within **2%** of the phase's entry baseline (28.9969 s at HEAD, so <= 29.58 s);
- `steady_allocs_per_iter` <= 0.3; `steady dl_wait` <= 0.05 ms; pinned <= 256 MiB;
- process-net peak strictly **lower** than the entry baseline by the phase's
  expected delta, within +/- 20%, or stop and report;
- `compute-sanitizer --tool memcheck`, headless, >= 200 iterations, zero errors,
  for any CUDA change;
- targeted gtests for the touched subsystem (never full ctest);
- a GUI validation run for any phase touching `src/visualizer` or `src/rendering`;
- report to `.codex_tmp/pools/phaseN_report.md`, one commit per phase, plain
  message, no push.

| phase | scope | expected delta | risk |
|---|---|---:|---|
| **P0** | Disclosure gauges (EXACT-5): `required_bytes` alongside `allocated_bytes` for S1, S3, S4, S5, S6; slack total in perf-bench. No behavior change. | 0 MiB | none; read-only |
| **P0.5** | Attribute `unattributed_residual` (140.3 MiB). Read-only investigation with the VRAM profiler and pool reserved-vs-used. | 0 MiB | none; gates the credibility of every later number |
| **P1** | **S1 FastGS sort.** Delete both 1.5x applications, consolidate to one block, serve from `CudaMemoryPool`, shape (a) preferred. Adds the exact/async `CudaStorageMode` from S8 if the consolidated size parks above a bucket step. | **-55 to -60** | R1, R3, R4 |
| **P2** | **S3 loss workspace arena.** Active-variant layout, delete +25%. | **-20 to -30** | low; variant oracle must match `ensure_*` exactly |
| **P3** | Densify scratch x1.2, MRNF x3/2 deleted-mask growth, MCMC 1.05x population forecast, `_ones_int32` and score/free-mask `max_cap` reserves. | -5 to -15 | low; many small sites, one commit each is fine |
| **P4** | **S5 arena commit policy.** Exact commit deltas, first-frame initial commit, decommit reachable from B1/B3/B6. | -10 to -30 | medium; commit-rate and OOM-gate interaction |
| **P5** | **S4 training state capacity.** Exact live-N capacity, `max_cap` pre-size removed, VA reservation kept. | **-58.7** + startup spike | **highest in-scope**; R2. Fence: if the transient or wall gate fails, stop and escalate to the per-region VMM design, do not tune a factor back in |
| **P6** | **S2 gsplat +25%** removal. | 0 on this gate | low; separate `--method gsplat` gate |
| **P7** | **S6 viewer scratch slack**: three multipliers + 128 MiB floor + total-VRAM VA reserve + `growRegionCapacity` + readback 1.5x + `capacity *= 2` helpers. | large, not on this gate | medium; GUI gate required |
| **P8** | **S6 viewer retention**: `no_shrink` default, interop high-water, output-image idle trim, B4/B5/B7 wiring. | Class B only | medium; R6 timeline retirement |
| **P9** | `HIGS_DEPTH_WAVE_INSTANCES` fixed 4 M cap -> measured tile instances. | ~100 MiB viewer | **high**; correctness/visual, own A/B |

P1 and P2 together are expected to clear the 36 MiB deficit and pass the doctrine
gate. Everything after P2 is margin and scale-readiness, and P5 is where the
architecture actually gets fixed.

---

## 7. Riskiest sites

**R1. FastGS: losing the optimistic `create_instances` overlap.** Shape (b) puts
a host wait between the scan and the first dependent launch every iteration.
Mitigation is shape (a), which keeps the overlap and is still exact because the
clamp is a measured high-water rather than a prediction. Watch
`n_instances_fallback_sync_count()`: a rising overflow rate means the clamp is
too tight for the scene, and the answer is to accept the re-run cost, never to
multiply the capacity.

**R2. Exact model capacity raising the peak it was meant to lower.**
`growExportableDeviceBlock` allocates the full new size before releasing the old,
so a grow costs `old + new` instantaneously. More frequent exact grows can raise
the judged peak even as steady residency falls. Measure the transient, not just
the steady state, before accepting P5.

**R3. Bucket quantization inverting the fix.** `SizeBucketedPool::get_bucket_size`
uses a 16 MiB grid between 16 and 256 MiB. A 17 MiB exact request becomes 32 MiB,
worse than the 1.5x being removed. This is why consolidation is mandatory and why
EXACT-5 measures allocated against required. If a workspace's exact size lands
just above a grid step and stays there, the correct fix is to serve it from the
async tier (exact `cudaMallocAsync`) rather than to pad the request.

**R4. Pool cache budget eating the savings.** `cache_budget_for_total_memory` is
`total/96` clamped to `[64, 256] MiB`, so on a 24 GiB card the bucket cache may
retain up to 256 MiB. Moving five per-iteration workspaces into the pool can grow
`pool_bucket_cache` from its measured 10 MiB toward that ceiling and cancel the
win. The budget must be re-measured after P1 and lowered if it drifts up.

**R5. `trim_cached_memory()` on a hot path.** It runs `cudaDeviceSynchronize`.
`improved_gs_plus.cpp:819` calls it from training. Boundary-only, per section 5.

**R6. Removing `no_shrink=true` in the Vulkan path.** Freeing a device buffer
that an in-flight command buffer still references is a use-after-free that
validation may not catch on every driver. Shrink must go through timeline-based
retirement, reusing the existing swapchain/interop retirement path.

**R7. The nvImageCodec device-allocator hook.** Fixed in `d786be384`, then
disabled in `2869f6d27` because enabling it measured 1491 MiB and 0.092 ms
dl_wait against 1422 MiB and 0.026 ms with it off. The 293.3 MiB decoder row is
the largest single controllable block left, but the evidence currently says the
budgeted pool makes things worse. Owner decision, with measurement, not an
assumption.

**R8. Deleting Class B retention that is load-bearing for latency.** The LOD page
pool, the upload staging ring depth, and the output-image idle trim are retention
policies whose removal shows up as stutter, not as a VRAM number. They are Class
B, they get boundaries, and they do not get deleted.

---

## 8. Status of the in-flight Codex phase 1: measured FAIL

The working tree contains a FastGS rewrite (`forward.cu` -405/+164, plus
`forward.h`, `rasterization_api.cu`, `fast_rasterizer.hpp`). Its 3-run gate
completed while this design was being written. **It regresses the metric it was
meant to improve.**

| metric | baseline (HEAD 2869f6d27) | phase 1 median | verdict |
|---|---:|---:|---|
| process-net peak | 1422 MiB | **1556 MiB** | **FAIL, +134 MiB** (gate 1386) |
| steady allocations/iter | 0.2315 | **0.4787** | **FAIL** (gate 0.30) |
| wall | 28.9969 s | 29.1223 s | PASS (+0.43%) |
| `pool_bucket_cache` | 10.0 MiB | **250.5 MiB** | pinned at the budget ceiling in all 3 runs |
| `fastgs_sort_hwm` | 107.0 MiB allocated | 63.0 MiB required | the forecast slack, ~44 MiB, is genuinely gone |
| `training_state_capacity_overhead` | 58.7 MiB | 58.7 MiB | untouched, still the next target |

Read together: the phase **succeeded** at deleting the 2.25x forecast (the sort
workspace's true requirement is 63 MiB, and 44 MiB of pure prediction is gone)
and **failed** by moving that workspace's retention into the shared bucket cache,
which promptly saturated its 256 MiB budget. Freeing five buffers per forward
also added roughly 0.25 driver allocations per iteration from bucket misses as
`n_instances` jitter walks across grid steps. R3 and R4 were not hypothetical.

This is the whole argument for EXACT-2 and the EXACT-4 caveat: exactness is a
property of the resident set, not of the allocation call rate. What it did:

- Deleted `kSortBufferGrowthFactor` and both applications. **Correct.**
- Replaced the TLS grow-only cache with `ExactPoolDeviceBuffer`, allocating
  through `allocate_cuda_storage(..., Pooled, ...)` and freeing with
  `safe_cuda_pool_deallocate`. **Correct**, and it removes the device-wide
  synchronize that the old `reset()` performed on every growth.
- Chose shape **(b)**: `cudaEventSynchronize` before `create_instances`, then
  exact allocation, one path, no overflow re-run. **Admissible only if the 2%
  wall gate passes** (R1). If it does not, the fix is shape (a), not a factor.
- Allocates **five separate buffers** per forward. This is the gap: it does not
  consolidate, so it pays the allocator grid five times, and above roughly 4 M
  instances each key/index buffer crosses into the 16 MiB grid and the effective
  waste can exceed the 1.5x that was removed (R3). At bicycle scale (roughly
  10 MiB per buffer, 1 MiB grid) it is fine, which means the gate will not catch
  it. **Consolidation must be added before this phase is accepted.**
- Frees the workspace every forward, so retention moves wholesale into
  `pool_bucket_cache`, whose budget is up to 256 MiB (R4). The phase report must
  include the `pool_bucket_cache` ledger row before and after, not only the
  process-net peak.
- Reduces `sort_workspace_high_water_bytes()` to "last requirement", which
  quietly changes what an existing telemetry API means. Under EXACT-5 that value
  should be published as `required_bytes` with `allocated_bytes` alongside it.

### Required rework before phase 1 can be accepted

Keep the deletion of `kSortBufferGrowthFactor`; keep the pool as the allocator;
keep the teardown guards. Change three things:

1. **Stop freeing per forward.** Hold the workspace across iterations under
   EXACT-2 (reallocate only when the measured `n_instances` exceeds it) and
   release it at B1/B3/B6 through the existing
   `release_sort_workspace_buffers()`. Resident becomes ~63 MiB exact instead of
   63 MiB live plus 250 MiB cached, and `steady_allocs_per_iter` returns to
   roughly the 0.2315 baseline because the steady case stops allocating at all.
2. **Consolidate** the four instance buffers and the CUB workspace into one
   block with internal offsets, so the allocator grid is paid once (R3).
3. **Publish the pair.** `fastgs_sort_hwm` now silently means "last requirement";
   under EXACT-5 it must report `required_bytes` and `allocated_bytes`
   separately, and the phase report must carry the `pool_bucket_cache` delta
   alongside the process-net peak. Had that row been a gate condition, this
   regression would have been caught before three full runs.

With those three, the expected result is the -55 to -60 MiB in the P1 row, and
the shape question (a) versus (b) can be settled on wall time alone, since (b)
has now measured at +0.43%, comfortably inside the 2% budget. That is a useful
result on its own: **the mid-pipeline sync is affordable**, so the optimistic
`create_instances` overlap is not worth preserving for its own sake and shape (b)
can stand.

Do not start P5 until P0.5 has attributed the residual.

## OWNER AMENDMENT (2026-08-10, final authority)
Splats are PREALLOCATED in splat_data: the max_cap preallocation of splat/training state is sanctioned design, not an estimate. Section S4 / Phase P5 is CANCELLED (no exact live-N capacity work, no grow-path changes). Everything else must be exact from iteration to iteration: exact-grow the moment the measured requirement exceeds holdings; shrink at EXACT-3 boundaries. All other rules and audit amendments stand.
