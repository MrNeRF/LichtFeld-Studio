I'll explore the tensor library's memory path end-to-end—pools, slabs, lazy materialization, and any known findings—then produce a report with concrete file:line references.Reading the multi-tier pool, slab, and related allocator sources in depth.# Tensor Library GPU Memory Behavior Report

**Scope:** `src/core/tensor/` (+ `src/core/memory_pressure.cpp` entrypoint, densification call sites)  
**Mode:** static analysis only — no runtime measurements.  
**Date of tree:** current workspace (post-`#1413` hardening notes in `TENSOR_LIB_FINDINGS.md`).

---

## 1. Full allocation stack

### 1.1 End-to-end path (Tensor creation → device heap)

```
Tensor::empty / zeros / ones / rand / …  (tensor_factory.cpp)
        │
        ▼
Tensor::load(LoadOp::*, …)               (tensor_unified_ops.cpp:343+)
        │
        │  CUDA non-empty:
        │  allocate_cuda_storage(bytes, stream)  (memory_pool.hpp:41–46;
        │                                         impl memory_pressure.cpp:278–340)
        │
        ▼
MemoryPressureCoordinator
  try_allocate → CudaMemoryPool::try_allocate  (or Direct cudaMalloc)
  on OOM → trim caches + retry once
        │
        ▼
CudaMemoryPool::try_allocate             (memory_pool.hpp:104–207)
  tier 1: GPUSlabAllocator   if bytes ≤ 256 KiB
  tier 2: SizeBucketedPool cache hit, else cudaMallocAsync(bucket_size)
  tier 3: cudaMallocAsync(exact bytes)   if > 16 GiB or bucket path failed
  tier 4: cudaMalloc(exact) “direct tier” fallback
        │
        ▼
Tensor owns storage via shared_ptr deleter →
  CudaMemoryPool::deallocate / cudaFree (Direct) / PinnedMemoryAllocator (CPU pinned)
```

Concrete factory path:

| Step | Location | Behavior |
|------|----------|----------|
| `Tensor::empty` | `tensor_factory.cpp:94–101` | Builds `LoadArgs`, calls `load(LoadOp::Empty)` |
| CUDA alloc | `tensor_unified_ops.cpp:404–418` | `allocate_cuda_storage(bytes, s)` then `adopt_storage` with pool deleter |
| Empty-numel sentinel | `tensor_unified_ops.cpp:371–379` | Still allocates **1 byte** CUDA sentinel via pool |
| Stream home | `tensor_unified_ops.cpp:362` | `state_->stream = getCurrentCUDAStream()` |
| CPU path | `tensor_unified_ops.cpp:419–449` | Pinned cache allocator or `malloc` |

`allocate_cuda_storage` (`memory_pressure.cpp:278–340`):

1. Optionally fail under OOM probe.
2. `CudaStorageMode::Pooled` (default) → `CudaMemoryPool::try_allocate`.
3. `CudaStorageMode::Direct` → bare `cudaMalloc` (`memory_pressure.cpp:96–119`).
4. On `cudaErrorMemoryAllocation`: pressure episode (trim tensor pool, pinned cache, etc.) then **one** retry; else throw `MemoryAllocationError`.

`CudaMemoryPool::allocate` is a thin wrapper that always uses pooled mode (`memory_pool.hpp:100–102`).

### 1.2 When does `cudaMalloc` / `cudaMallocAsync` actually run?

| API | When | File:line |
|-----|------|-----------|
| **Slab growth** `cudaMalloc(slab_size)` | First miss in a size class; slabs 256 KiB–8 MiB | `gpu_slab_allocator.hpp:237–250` |
| **Bucket miss** `cudaMallocAsync(bucket_size, stream)` | Request ≤16 GiB, cache empty | `memory_pool.hpp:163–177` |
| **Async exact** `cudaMallocAsync(bytes, stream)` | >16 GiB **or** bucket alloc failed | `memory_pool.hpp:186–203` |
| **Direct tier** `cudaMalloc(bytes)` | Async path unavailable/failed | `memory_pool.hpp:536–562` |
| **Direct storage mode** `cudaMalloc` | `zeros_direct`, `reserve` | `tensor.cpp:3368–3370`, `3493–3495`; `memory_pressure.cpp:96–119` |
| **Bypass pool (ops)** `cudaMallocAsync` | Warp-reduce partials, fused pointwise partials, optimized dot partials | `tensor_warp_reduce.cu:1078–1080`, `tensor_fused_pointwise.cu:319`, `tensor_dot_optimized.cu:130` |
| **Bypass pool** `cudaMalloc` | `CudaDeviceMemory` shape metadata; NaN-check buffers; strided upload metadata | `cuda_memory_guard.hpp:25`, `tensor_ops.cu:3088–3089`, `tensor.cpp:1290–1298` |

**Note:** `SizeBucketedPool::allocate` (`size_bucketed_pool.hpp:203–224`) also does `cudaMallocAsync`, but production routing only uses `try_allocate_cached` + free-side `cache_free` via `CudaMemoryPool` (`memory_pool.hpp:152`, `604`). Fresh bucket growth is done **inside** `CudaMemoryPool`, not via `SizeBucketedPool::allocate`.

### 1.3 Cache retention / eviction by layer

#### Layer A — GPU slab (≤256 KiB)

| Property | Value | Ref |
|----------|-------|-----|
| Size classes | 11 power-of-2: 256 B … 256 KiB | `gpu_slab_allocator.hpp:32–35`, `146–160` |
| Slab commit | `cudaMalloc` whole slab; carve fixed blocks into **virgin** free list | `237–271` |
| Free-list model | Per-stream free lists + virgin (stream-free) | `25–29`, `200–205` |
| Same-stream reuse | Pop from that stream’s free list (no event) | `308–314` |
| Cross-stream reuse | “Steal” from richest other stream + `bridgeStreams` | `323–345` |
| Eviction / return to driver | **None until process shutdown** (`cleanup` → `cudaFree` each slab) | `283–298`, `51–57` |
| Cap | `MAX_BLOCKS_PER_CLASS = 512K` bookkeeping only | `38` |

#### Layer B — Size-bucketed reuse cache (256 KiB–16 GiB)

| Property | Value | Ref |
|----------|-------|-----|
| Rounding | Multi-tier: 256 KiB → 1 MiB → 16 MiB → 64 MiB → 256 MiB → 1 GiB steps | `size_bucketed_pool.hpp:57–71` |
| Per-bucket cache depth | 4 (≤16 MiB), 3 (≤64 MiB), 2 (≤256 MiB), **1** (larger) | `88–96` |
| Global cache budget | `clamp(total_vram/96, 64 MiB, 256 MiB)` | `98–102`, `338–361` |
| Free → cache | Tag with home stream; FIFO entry eviction at depth | `145–200` |
| Large first-use skip | If `bucket_size > budget/2` and never hit → **immediate** `cudaFreeAsync` (not cached) | `156–169` |
| Cross-stream reuse | Prefer same-stream entry; else `bridgeStreams` | `115–128` |
| Budget eviction | Oldest `last_hit_epoch`, prefer larger buckets on tie | `363–420` |
| Explicit trim | `trim_cache()` frees all; also used by pressure / `trim_cached_memory` | `267–293`; `memory_pool.hpp:464–496` |

#### Layer C — CUDA driver default mempool (`cudaMallocAsync`)

| Property | Value | Ref |
|----------|-------|-----|
| Release threshold | **64 MiB** (`cudaMemPoolAttrReleaseThreshold`) | `memory_pool.hpp:389–396` |
| Comment intent | Keep per-iter scratch resident; reclaim densification spikes (not `UINT64_MAX`) | `389–392` |
| Trim | `cudaMemPoolTrimTo(pool, 0)` on `trim` / `trim_cached_memory` | `717–729` |

#### Layer D — Async tier frees (no app cache)

`AllocMethod::Async` blocks are freed with `cudaFreeAsync` on home stream only — reuse is **driver-side**, not app-side (`memory_pool.hpp:615–630`).

#### Layer E — Direct / long-lived param storage

`zeros_direct` / `reserve` use `CudaStorageMode::Direct` → `cudaMalloc` + `cudaFree` deleter, labeled `cuda.direct`, **not** in slab/bucket maps (`tensor.cpp:3368–3396`, `3493–3524`). Training explicitly prefers this for densification params (`mcmc.cpp:839–849`).

#### Layer F — Pinned host allocator

| Property | Value | Ref |
|----------|-------|-----|
| Backend | `cudaHostAlloc` with `malloc` fallback | `pinned_memory_allocator.cpp:239–258` |
| Rounding | Exact &lt;4 KiB; 512 B buckets &lt;1 MiB; power-of-2 ≥1 MiB | `154–169` |
| Cache budget | Default **1 GiB** (`LFS_PINNED_CACHE_LIMIT_MB`) | `21`, `27–42` |
| Reuse gate | All recorded CUDA events complete | `90–109`, `201–230` |
| LRU eviction | While `cached_bytes > limit` | `338–361` |
| Stream free | Events per stream on deallocate | header `62–68`; impl `439+` |

#### Layer G — Lazy executor materialization cache

| Property | Value | Ref |
|----------|-------|-----|
| Lifetime | **Per root materialization only** (`LazyExecutorContext` on stack) | `lazy_executor.cpp:28–30`, `794–798` |
| Key | Lazy IR `node_id` → `Tensor` | `685–707` |
| Early release | Memory planner drops dead intermediates after last consumer step | `256–295`, `485–498` |
| Defaults | Fusion ON, memory planner ON | `152–168` |
| Size defer threshold | **4096 bytes** | `111`, `915–934` |

Registry of deferred materializers is process-wide; pruned when owner weak_ptr expires and when size ≥16 (`112`, `170–181`, `566–569`).

#### Layer H — CUDA event pool

| Property | Value | Ref |
|----------|-------|-----|
| Cap | 512 events | `cuda_event_pool.hpp:20` |
| Flags | `cudaEventDisableTiming` | `cuda_event_pool.cpp:28–29` |
| Overflow | Destroy excess | `45–56` |

Used by `bridgeStreams` / stream waits (`cuda_event_pool.cpp:79–117`, `cuda_stream_context.cpp:23–64`). Events are handles, not large VRAM.

### 1.4 Free / rehome / multi-stream

On free (`memory_pool.hpp:311–348`, `594–630`):

1. Take `AllocationInfo` (size, method, `home_stream`, `extra_streams`).
2. `bridgeStreams(extra → home)` for every recorded use.
3. Route: slab free-list / bucket cache / `cudaFree` / `cudaFreeAsync`.

`record_stream` / `rehome_stream` update the map (`211–309`). Non-owning tensors skip pool tracking (`tensor.cpp:855–868`) — D4 in findings.

---

## 2. Fragmentation analysis

### 2.1 Size bucketing waste (internal fragmentation)

Rounding rules (`size_bucketed_pool.hpp:57–71`):

| Request range | Granularity | Max waste (almost next step) |
|---------------|-------------|------------------------------|
| ≤256 KiB | floor 256 KiB | (slab handles ≤256 KiB separately) |
| ≤1 MiB | 256 KiB | ~255 KiB |
| ≤16 MiB | 1 MiB | ~1 MiB − 1 |
| ≤256 MiB | 16 MiB | ~16 MiB − 1 |
| ≤1 GiB | 64 MiB | ~64 MiB − 1 |
| ≤8 GiB | 256 MiB | ~256 MiB − 1 |
| &gt;8 GiB | 1 GiB | ~1 GiB − 1 |

`get_waste_percentage` is available (`319–323`). Stats accumulate lifetime `bytes_wasted` (sum of round-ups), not current free waste (`134`, `223`, `316`).

**Slab internal waste:** request of 257 B takes 512 B class; request of 128 KiB+1 takes 256 KiB (`146–160`). Worst case nearly 50% within a class.

### 2.2 Slab reuse conditions (stream matching)

Order in `pop_block` (`gpu_slab_allocator.hpp:301–345`):

1. Same `stream` free list.
2. Virgin (never used / merged after device sync).
3. Steal richest foreign stream + **GPU event edge** (no host sync on success).

Free always pushes onto the **home stream** free list (`348–353`), after `free_routed` bridges extras to home (`memory_pool.hpp:594–601`).

**Fragmentation implication:** free blocks are siloed by stream. Many short-lived streams or concurrent streams can leave free capacity stranded on idle streams until steal (event cost) or `merge_stream_into_virgin` / `trim_cached_memory` (device sync) (`gpu_slab_allocator.hpp:90–111`, `memory_pool.hpp:471–491`).

### 2.3 No true defragmentation

| Subsystem | Compaction? | Evidence |
|-----------|-------------|----------|
| Slab | **No** — blocks never coalesce; slabs never return until shutdown | `cleanup` only in `shutdown` |
| Size bucket | **No** — only free whole bucket-sized blocks; no coalescing of live tensors | cache_free / cudaFreeAsync |
| Direct params | **No** — grow = alloc new + copy + free old (`tensor.cpp:3365–3432`) |
| Driver pool | Soft reclaim via release threshold / trim only | `memory_pool.hpp:389–396`, `717–729` |

### 2.4 Worst-case densification / growth scenarios

Training long-lived Gaussian params intentionally **exit the pool**:

```839:852:src/training/strategies/mcmc.cpp
// ELIMINATE ALL POOL ALLOCATIONS: Replace pool-allocated parameters with direct cudaMalloc versions
...
auto new_param = Tensor::zeros_direct(param.shape(), capacity);
cudaMemcpy(...);
param = std::move(new_param);
```

Effects:

1. **Peak 2× during growth:** old buffer + new `zeros_direct` capacity held until move/destructor (`mcmc.cpp:842–844` comment; same pattern `tensor.cpp:reserve` 3365–3432).
2. **Direct heap vs async pool split:** large params live in the classic `cudaMalloc` heap; per-step scratch still uses `cudaMallocAsync` pool → two heaps, less cross-reuse.
3. **Pool scratch size churn:** intermediate tensors change with image resolution, Gaussian count (masks, gathers, error maps). As sizes climb, new buckets fill; old buckets retain up to depth/budget until LRU/probationary free (`size_bucketed_pool.hpp:156–169`, `387–420`).
4. **Large one-shot buffers:** first free of a &gt;budget/2 block with no prior hits is not cached — good for densification spikes — but **repeat** same large size (e.g. densify every N iters) will re-`cudaMallocAsync` after free if entry was not retained (max 1 entry for large buckets).
5. **`reserve` on Direct tensors** (`command_api.cpp:447–448`, `tensor.cpp:3368–3370`): always Direct realloc; cannot grow into pooled slack.

---

## 3. Allocation churn in a typical training step

### 3.1 Paths that allocate **fresh** storage most steps

| Path | Mechanism | Ref |
|------|-----------|-----|
| Op results (`empty` for outputs) | Full pool path every new tensor | `tensor_unified_ops.cpp:404–418`; expr evals `tensor_expr_impl.hpp:94`, `341`, … |
| Deferred expr materialization | `Tensor::empty` inside unary/binary eval | `tensor_expr_impl.hpp:34–45`, `86–94` |
| Lazy pointwise fusion output | `Tensor::empty` then fused kernel | `lazy_executor.cpp:436–440` |
| `.contiguous()` if non-contiguous | Always `empty` + strided copy | `tensor.cpp:994–1019` |
| Reductions (general) | Output `empty`; may `contiguous()` first | `tensor_unified_ops.cpp:1389–1456` |
| Transpose-reduce opt | `permute.contiguous()` **full copy** + reduce when `inner_size ≥ 256` | `tensor_unified_ops.cpp:1321–1375` |
| Randint Float32/UInt8 | Temp `int` buffer via pool | `tensor_unified_ops.cpp:802–826` |
| Odd-length normal | Scratch `n+1` empty + D2D copy | `tensor_unified_ops.cpp:752–763` |
| Warp-reduce / dot / fused PW | Per-call `cudaMallocAsync` partials (~grid floats) unless buffer passed | `tensor_warp_reduce.cu:1078–1080`, `tensor_dot_optimized.cu:130–133` |
| CUB workspace | Pool-backed scoped buffers | `cub_workspace.hpp:19–53` |
| Rank&gt;3 / generic strided paths | Bare `cudaMalloc` shape/stride metadata | `tensor.cpp:1286–1298`, `1021–1037` |
| Densification error maps | Re-`empty` when H/W change | `trainer.cpp:4972–4975` |

### 3.2 Paths that **reuse** across steps

| Path | Mechanism | Ref |
|------|-----------|-----|
| Same-size pooled temps | Slab free-list / bucket cache / driver pool | §1.3 |
| Long-lived params / Adam moments | `zeros_direct` capacity; in-place rows | `adam_optimizer.cpp:316–327`, `404` |
| Pre-reserved max_cap | Avoids reallocation until growth | `mcmc.cpp:830–877` |
| `bg_mix_buffer_`, `densification_error_map_` (stable shape) | Member tensors reused | `trainer.cpp:3556–3557`, `4972–4979` |
| Lazy fusion chain | One launch vs N intermediate tensors when fusion hits | `lazy_executor.cpp:404–441`, `515–525` |
| Lazy early-release | Drops intermediate tensors mid-plan | `lazy_executor.cpp:485–498` |
| Column-reduce fast path | No transpose copy for 2D dim0 | `tensor_unified_ops.cpp:1299–1317` |
| Contiguous identity | Shallow copy, no alloc | `tensor.cpp:1000–1002` |
| Views (permute/slice/reshape when already dense) | Share `data_owner_` | `tensor_shape_ops.cpp:114–118`, `tensor_impl.hpp:990–997` |
| NaN-check buffers | `thread_local` once per thread | `tensor_ops.cu:3079–3125` |
| Event pool | Reuse up to 512 | `cuda_event_pool.cpp:16–38` |

### 3.3 Expression / lazy intermediate lifetime

1. Operator conversion defers if `bytes ≥ 4096` (`tensor_expr_impl.hpp:34–45`; threshold `lazy_executor.cpp:111`).
2. Materialize: planner topo-executes registered nodes, caches tensors in **local** context, optional early free (`lazy_executor.cpp:462–542`, `737–814`).
3. Root may still call materializer fallback if root not in cache (`787–813`).
4. After materialize completes, context destructor drops all cached intermediate `Tensor`s → pool free/cache.

**Peak within one materialize:** sum of live intermediates before early-release + outputs. Memory planner reduces this when enabled (default on).

---

## 4. Known waste

### 4.1 D1 deferred identity / double materialization

Documented in `TENSOR_LIB_FINDINGS.md:253–260`:

- Copy ctor deep-copies `TensorState` (`tensor.cpp:636–639`).
- `LazyExprState` is **`shared_ptr`** inside `TensorState` (`tensor_impl.hpp:438`, `2654–2670`), so **plain copies share** the same lazy cell (gate + `result` + materializer) — pure shallow copies of a deferred tensor should materialize once.
- Remaining amplifier (still in tree): deferred `permute` / `create_view` / etc. build **new** deferred tensors whose materializers **copy** the source and re-enter materialization (`tensor_impl.hpp:971–978`, `tensor_shape_ops.cpp:102–109`). Nested graphs can re-evaluate / allocate multiple result buffers if fusion/planner miss.
- Registry still maps **one** materializer per `node_id` (`lazy_executor.cpp:547–570`); unregister on first successful materialize (`tensor.cpp:480–485`). Desync risk if distinct `LazyExprState` instances ever share a `node_id` without sharing the cell (documented D1 scenario).

**VRAM impact (when triggered):** double compute + **two** pool allocations for the “same” logical result; aliases can diverge.

### 4.2 Lazy cache growth

- **Per-execution** cache only — does **not** retain across training steps (`lazy_executor.cpp:794–798`).
- Process-wide registries retain materializer/`std::function` + fusion `shared_ptr<Tensor>` operand snapshots until prune (`573–617`, `lazy_executor_snapshot_operand`). Snapshots can pin **live storage** or force clone-on-write via `preserve_lazy_snapshots_before_write` (`tensor.cpp:386–409`).
- Operand mutation while deferred graphs live can force **clone of full tensors** (`406–408`) — large sudden VRAM if params are mutated under pending lazy ops.

### 4.3 Per-op temp buffers (pool bypass or churn)

| Buffer | Alloc API | Pooled? | Ref |
|--------|-----------|---------|-----|
| Warp-reduce partials | `cudaMallocAsync` | Driver only | `tensor_warp_reduce.cu:1078–1080` |
| Dot partials | `cudaMallocAsync` + free | Driver only | `tensor_dot_optimized.cu:130–133` |
| Fused pointwise partials | `cudaMallocAsync` | Driver only | `tensor_fused_pointwise.cu:319` |
| Randint conversion | `CudaMemoryPool::allocate` | Yes | `tensor_unified_ops.cpp:802–826` |
| `CudaDeviceMemory` | bare `cudaMalloc` | **No** | `cuda_memory_guard.hpp:25–46` |
| Shape metadata | bare `cudaMalloc` | **No** | `tensor.cpp:1290+`, `tensor_ops.cu:2975+` |
| CUB | pool | Yes | `cub_workspace.hpp` |
| Masking `d_count` | bare `cudaMalloc` | **No** | `tensor_masking_ops.cpp:1772` |

Untracked `cudaMalloc` contributes to fragmentation and is **invisible** to `allocation_map_` / VramProfiler pool tiers.

### 4.4 `thread_local` staging

| Site | Behavior | Ref |
|------|----------|-----|
| `NaNCheckBuffers` | 4 B device + 4 B pinned host, permanent per thread | `tensor_ops.cu:3079–3125` |
| Pool label string | Host only | `tensor.cpp:810` |
| Current CUDA stream | Handle only | `cuda_stream_context.cpp:13` |
| Lazy context pointer | Pointer only | `lazy_executor.cpp:124` |
| `TensorRowProxy` staging slots | Host float slots + D2H/H2D per element access | `tensor_row_proxy.cpp:49–104`; findings SWR-13 |

Findings also cite **`thread_local` shape buffers** in broadcast/`where` kernels (`TENSOR_LIB_FINDINGS.md:143–144`, `tensor_masking_ops.cu`) as race/correctness hazards; VRAM is small (host), but reuse across streams is unsafe.

### 4.5 Other structural waste

- **Empty CUDA tensors allocate 1-byte sentinels** (`tensor_unified_ops.cpp:371–379`).
- **Bucket waste tracked but retained:** free returns full `bucket_size` to cache, not exact request (`size_bucketed_pool.hpp:192–195`).
- **Slab never shrinks** → steady-state reserved VRAM grows with historical peak of small-alloc diversity (`gpu_slab_allocator.hpp:283–298`).
- **Direct param over-capacity:** `zeros_direct` zeros **full capacity**, not logical size (`tensor.cpp:3497–3498`) — intentional headroom, real VRAM.
- **`contiguous_read` is not a deferred barrier** (`TENSOR_LIB_FINDINGS.md:251–252`; `tensor_impl.hpp:567–574` checks `is_contiguous()` only; deferred tensors set `is_contiguous_=true` with null data at `tensor.cpp:433–438`).

---

## 5. Peak VRAM contributors beyond raw model tensors

| Contributor | Steady-state? | Bound / behavior | Ref |
|-------------|---------------|------------------|-----|
| Slab reserved memory | Yes (monotonic until trim/shutdown; trim **does not free slabs**) | Sum of committed slabs; published to VramProfiler | `gpu_slab_allocator.hpp:267–268`, `362–367`; trim merges free lists only (`memory_pool.hpp:489–490`) |
| Bucket reuse cache | Yes | 64–256 MiB budget | `size_bucketed_pool.hpp:34–35`, `338–361` |
| Driver async pool reserved − used | Yes soft | Release threshold 64 MiB; can still hold more until trim | `memory_pool.hpp:389–396` |
| Lazy intermediate peak | Per graph only | Tracked as `peak_cache_bytes` | `lazy_executor.cpp:64–66`, `476–483` |
| CUDA event pool | Negligible VRAM | ≤512 events | `cuda_event_pool.hpp:20` |
| Pinned host cache | Host RAM, not VRAM | ≤1 GiB default | `pinned_memory_allocator.cpp:21` |
| Direct param capacity headroom | Yes | `max_cap − live Gaussians` × features | `mcmc.cpp:830–877`, `zeros_direct` |
| Growth spikes | Transient | 2× during `zeros_direct`/`reserve` replace | `mcmc.cpp:842–844` |
| Rasterizer arena | Separate subsystem | `GlobalArenaManager` trimmed on trainer shutdown (`trainer.cpp:3317`) | outside tensor lib but same process |
| Untracked `cudaMalloc` metadata / partials | Transient or sticky (NaN buffers) | Small but fragmenting | §4.3 |
| Pressure reclaim | On OOM | Trims bucket cache + driver pool + pinned | `memory_pressure.cpp:242–266` |

**Defragmentation ability: effectively none** for tensor live data. Only:

- free-list reuse,
- cache eviction,
- OOM trim (`trim_cached_memory` needs **device sync** — `memory_pool.hpp:475–491`),
- driver mempool trim.

No live-tensor relocation/compaction API exists.

---

## 6. Opportunities to reduce steady-state and peak VRAM **without slowing down**

Ranked by expected impact (static estimate). “No slowdown” assumes same kernel work, fewer alloc/free round-trips or less retained free memory.

### Rank 1 — Cap / reclaim slab reserved memory after peak (high steady-state)

**Problem:** Slabs only `cudaFree` at shutdown (`gpu_slab_allocator.hpp:283–298`). Historical small-alloc diversity permanently reserves VRAM.  
**Opportunity:** After training warmup or on `trim_cached_memory`, free **completely empty** slabs (all blocks free) or demote unused size classes.  
**Risk:** Need careful stream/event safety; do only when free-count == blocks_per_slab.  
**Speed:** Steady-state free VRAM; alloc path unchanged when class still warm.

### Rank 2 — Avoid 2× peak on densification growth (high peak)

**Problem:** `zeros_direct` + copy + destroy holds both buffers (`mcmc.cpp:847–852`, `tensor.cpp:3365–3432`).  
**Opportunity:** Grow with **capacity already reserved** at `max_cap` (already partial strategy) — ensure all code paths use pre-reserve so growth never reallocates mid-training; or allocate new, copy, free old under a “swap” that trims driver pool immediately after free.  
**Existing good practice:** Skip re-alloc if capacity already sufficient (`mcmc.cpp:847–848`).  
**Speed:** Same kernels; fewer peak OOMs allows larger scenes.

### Rank 3 — Pool-route reduction partials / op scratch (medium steady + less driver churn)

**Problem:** Frequent `cudaMallocAsync`/`cudaFreeAsync` for partials bypasses SizeBucketed/slab and only hits the driver pool (`tensor_warp_reduce.cu:1078–1080`, `tensor_dot_optimized.cu:130–133`).  
**Opportunity:** Thread-local or stream-local **reusable partial buffers** via `CudaMemoryPool` (or keep a `grid_size`-sized scratch tensor on the trainer). Warp-reduce already accepts `partial_buffer` (`tensor_warp_reduce.cu:1051–1059`) — **pass it from callers**.  
**Speed:** Removes alloc/free from hot reductions → typically faster or equal.

### Rank 4 — Fix remaining deferred double-materialization (medium peak + compute)

**Problem:** Nested deferred views/expressions can materialize intermediate full buffers more than once (D1 / Scenario E; `TENSOR_LIB_FINDINGS.md:253–260`).  
**Opportunity:** Shared `LazyExprState` for identity; ensure view materializers return **views of cached root storage** without re-running full eval graphs; keep planner cache hits for shared subgraphs.  
**Speed:** Fewer kernels + fewer allocs.

### Rank 5 — Route shape/stride metadata through the pool (medium fragment cleanup)

**Problem:** Bare `cudaMalloc` for rank metadata (`tensor.cpp:1290–1298`, `cuda_memory_guard.hpp:25`) fragments classic heap and bypasses tracking.  
**Opportunity:** Use slab pool (sizes are tiny, ≤ few hundred bytes → 256 B class) or host-immediate parameters (already done for rank≤4 contiguous / rank-3 upload — `tensor.cpp:1017–1019`, `1262–1274`).  
**Speed:** Immediate-parameter paths already preferred; extending them removes alloc latency.

### Rank 6 — Prefer column-reduce / fused reduce over transpose+copy (medium peak bandwidth)

**Problem:** Single-axis non-last dim reduce does full transpose contiguous when `inner_size ≥ 256` (`tensor_unified_ops.cpp:1342–1362`) — temporary full-size tensor.  
**Opportunity:** Extend strided/column reduce coverage so large training tensors never allocate the transpose buffer.  
**Speed:** Comment claims strided is slower today; only “no slowdown” if strided kernel is improved to match — mark as **conditional**. Safer no-slowdown win: **reuse a persistent transpose workspace** instead of allocate-per-call.

### Rank 7 — Tighten driver release threshold under low free VRAM (medium steady)

**Problem:** 64 MiB threshold keeps async free memory resident (`memory_pool.hpp:393–394`).  
**Opportunity:** Dynamic threshold based on free VRAM / pressure coordinator (already trims to 0 on OOM — make proactive trim when free &lt; reserve, `memory_pressure.cpp:342–360`).  
**Speed:** Only runs when under pressure; hot path unchanged.

### Rank 8 — Bucket cache: prefer exact-size reuse for densify-sized scratch (low–medium)

**Problem:** Growing image/Gaussian-related buffers change buckets; old entries sit until LRU.  
**Opportunity:** On densify/resolution change, call existing `Tensor::trim_memory_pool` only for **probationary large buckets**, or lower large-bucket max entries (already 1 for &gt;256 MiB).  
**Already good:** Large probationary free (`size_bucketed_pool.hpp:156–169`).  
**Speed:** Unchanged for stable sizes.

### Rank 9 — Lazy size heuristic / fusion (low–medium peak for graph ops)

**Problem:** Defer at ≥4 KiB creates graphs + snapshots (`lazy_executor.cpp:111`, `573–606`). Snapshots can pin or clone storage on write (`tensor.cpp:386–409`).  
**Opportunity:** For training-hot ops that always materialize immediately, eager eval avoids snapshot pinning; keep fusion for multi-op chains that **do** fuse (net fewer temps).  
**Speed:** Fusion reduces launches; careful not to disable fusion.

### Rank 10 — Empty-tensor 1-byte sentinels (low)

**Problem:** Zero-numel CUDA tensors still allocate 1 B (`tensor_unified_ops.cpp:371–379`).  
**Opportunity:** Null data + dummy owner (already used in `zeros_direct` zero-byte path `tensor.cpp:3475–3479`).  
**Speed:** Negligible.

### Rank 11 — Event pool / pinned / thread_local (low VRAM)

Already well bounded. Not primary VRAM wins. Pinned is host RAM (`DEFAULT_CACHE_LIMIT` 1 GiB).

### Explicit non-goals for “no slowdown”

- Forcing `trim_cached_memory` every iteration (device sync — `memory_pool.hpp:475–477`) would **hurt** throughput.
- Turning off size-bucket cache would increase `cudaMallocAsync` traffic.
- Moving Direct params back into the async pool (opposite of `mcmc.cpp:839`) may improve packing but risks multi-stream reuse bugs and free-list churn — not free speed.

---

## Appendix A — Threshold constants (quick reference)

| Constant | Value | Location |
|----------|-------|----------|
| Slab max / threshold | 256 KiB | `memory_pool.hpp:28`; `gpu_slab_allocator.hpp:33` |
| Bucket max tracked | 16 GiB | `memory_pool.hpp:29`; `size_bucketed_pool.hpp:32` |
| Bucket cache budget | 64–256 MiB | `size_bucketed_pool.hpp:34–35` |
| Driver release threshold | 64 MiB | `memory_pool.hpp:393` |
| Lazy defer threshold | 4 KiB | `lazy_executor.cpp:111` |
| Event pool max | 512 | `cuda_event_pool.hpp:20` |
| Pinned cache default | 1 GiB | `pinned_memory_allocator.cpp:21` |
| VRAM reserve (pressure) | 512 MiB default, clamp [128 MiB, total/4] | `memory_pressure.cpp:344–358` |

## Appendix B — Ownership summary diagram

```
Tensor (data_owner_ shared_ptr)
  ├─ Pooled CUDA ──► CudaMemoryPool::allocation_map_
  │                    ├─ Slab blocks (cudaMalloc slabs, free-lists)
  │                    ├─ Bucket blocks (cudaMallocAsync, free cache)
  │                    ├─ Async exact (cudaMallocAsync, no app cache)
  │                    └─ Direct tier fallback (cudaMalloc)
  ├─ cuda.direct ──► bare cudaMalloc / cudaFree  (zeros_direct, reserve)
  ├─ external.* ───► foreign owner (Vulkan etc.)
  └─ CPU ──────────► PinnedMemoryAllocator or malloc
```

---

**Confidence notes (speculation marked):**

- Relative ranking of opportunities is **judgment** based on structure, not measured VRAM traces.
- Whether D1 still double-allocates on the **hot training path** after shared_ptr `LazyExprState` needs runtime verification; the nested deferred-view materializer pattern remains concrete in code.
- Training peak is likely dominated by **Direct param capacity + rasterizer arena + render buffers**, not slab free-lists; tensor-lib waste still matters for OOM headroom and multi-stream fragmentation.
