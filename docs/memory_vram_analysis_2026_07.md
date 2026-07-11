# Memory / VRAM Waste Hunt — Analysis Round

## Review contract and baseline

- Marker: `MEMORY-VRAM-HUNT-2026-07-11`
- Branch observed: `vulkan-hardening`
- Pinned review commit: `c08091557853014b99b68834980ed42661baee8f`
- Concurrent-edit drift: the live branch had advanced to `54dae2936f42cd9e68a6723e7c20a76c513010fa` at the final audit; no evidence anchor was rebased onto that moving checkout.
- Method: static analysis only. No source was changed, no build was run, and no repository binary was launched.
- The checkout was being edited concurrently. Every source anchor below is therefore `file:line@c08091557853014b99b68834980ed42661baee8f`; use `git show c08091557853014b99b68834980ed42661baee8f:<path>` if live line numbers drift.
- Prior art exclusion: findings already covered, fixed, refuted, or deferred in `docs/vulkan_rendering_analysis_2026_07.md` section 3 and its execution log through “Bonus round 2” are not re-reported. In particular, this report does not count dormant split-view output rings, the shared-scratch 384 MiB floor itself, the old per-readback staging allocations, the GT preparation cache, cached-frame synchronization, the environment cache, or the LOD staging ring as new findings.

Severity is ordered as requested: **Leak** (ownership is lost or a cache survives its owning scene/phase), then **Stability** (credible OOM, corruption, or use-after-recycle path), then **Waste** (recoverable high-water capacity or copies). Byte estimates are binary MiB/GiB unless explicitly marked decimal. Examples are not additive unless a section says they are.

## 1. Memory map

### 1.1 Allocator and owner map

| Owner / allocator | What it owns and when | Lifetime / release | Existing accounting | Important blind spot |
|---|---|---|---|---|
| CUDA primary context, modules, device runtime | Driver stacks, modules, library state, default CUDA pool | Process lifetime | Startup phase probes and process-level sample in `VramProfiler`; CUDA pool used/reserved | Not attributable to individual tensors; enabling the profiler later cannot reconstruct already-live allocations |
| `SplatExportableStorage` | One CUDA VMM physical block for means, SH, scale, rotation, opacity; Vulkan imports the same physical block | Shared by trainer/model/renderer owners | `exportable_splat_bytes` scalar gauge and process sample | Gauge represents one value, not a set of blocks; an old and new block can overlap; growth does not refresh the gauge (`src/core/cuda/exportable_storage.cpp:249-274,277-380@c08091557853014b99b68834980ed42661baee8f`) |
| `CudaMemoryPool` live allocations | Slab (<=256 KiB), size-bucketed async (up to 16 GiB), async/direct fallback | Tensor `shared_ptr` deleter, stream-routed | Requested live bytes in allocation map and `VramProfiler`; CUDA default-pool used/reserved | Requested bytes are not physical bucket bytes; direct-capacity tensor accounting is in a separate text counter; slab bytes are raw `cudaMalloc`, not default-pool bytes |
| `GPUSlabAllocator` | Raw `cudaMalloc` slabs, 11 size classes | Process shutdown only | `cuda_slab_reserved_bytes`, cumulative stats | Explicit pool trim merges free lists but never releases a fully idle slab |
| `SizeBucketedPool` | Freed `cudaMallocAsync` blocks retained for reuse | Explicit trim/shutdown, nominal 64-256 MiB budget | `cuda_pool_bucket_cache_bytes`, CUDA pool metrics | A sole oversized block is exempt from the budget and can be multiple GiB |
| Rasterizer arena | Internal VMM chunks in headless mode, fallback monolithic allocation without VMM, or borrowed exportable VkSplat shared scratch in GUI training | Frame reset; `full_reset`; process teardown | committed/peak/realloc statistics; individual chunks sent to profiler | Arena chunks are labeled `Direct`, not `Arena`; the internal reset currently fails to decommit after a high-water frame. Borrowed GUI scratch is prior art and must not be double-counted |
| Persistent training model and optimizer | SH3 model is 62 floats/Gaussian = 248 B/G; quantized Adam is 124 B/G moments + 48 B/G scales = 172 B/G; non-fused full gradients add 248 B/G | Trainer/model lifetime | Trainer’s manual `train.persistent` breakdown when live profiling is enabled; tensor storage text summary for direct allocations | Full `max_cap` is physically committed even when live `N` is much smaller; completed training retains optimizer and workspaces |
| Training strategy and loss workspaces | Densification state, free/visibility/error masks, SSIM variants, depth/edge workspaces, temporary sort/scan buffers | Mostly trainer lifetime; some raw temporaries are per refine/backward | Some manual trainer gauges and scoped VRAM deltas | Mutually exclusive loss modes retain all previously touched workspaces; several raw CUB allocations bypass the tensor pool and do not check OOM |
| Pipelined image loader | Ready RGB/mask/depth tensors, pending pairs, decode input staging, nvImageCodec decoder state, compressed CPU JPEG cache | Loader lifetime; ready/pending bounded by count; compressed cache uses byte/LRU pressure | Final pending/output tensors have explicit byte counters; nvImageCodec has a driver-delta scope | Prefetch tuning prices RGB as 3 B/pixel regardless of 16-bit mode and ignores depth plus concurrent decode staging |
| Pinned host allocator | Default backing for every CPU `Tensor`, not just transient transfer staging | Active block -> unbounded per-size cache -> shutdown `empty_cache` | `allocated_bytes` is published as “pinned host” | Cached bytes are omitted from the HUD; fallback `malloc` provenance is lost; archival clipboard/undo/mesh data is unnecessarily page-locked |
| Legacy `CacheLoader` / nvImageCodec | CPU image/JPEG caches and a deliberately leaked legacy decoder pool | CPU caches have pressure eviction and explicit clear; legacy decoder is process-lifetime | CPU cache logs; `io.nvimagecodec` driver-delta probe | Legacy pool can coexist with pipeline-owned pools; exact driver VRAM is dynamic |
| Vulkan/VkSplat renderer | VMA images/buffers, renderer-owned sort/output/overlay resources, imported model views | Renderer/resource retirement | Vulkan VMA used/block bytes and `vksplat.memory` | Use these as the renderer term; do not add imported model views or borrowed shared scratch again |
| Undo / clipboard / scene caches | Deep snapshots, selection masks, clipboard models, CPU mesh ray-pick mirrors | History/clipboard/global-cache lifetime | Undo has logical CPU/GPU breakdown; most tensors appear only if profiler was enabled before allocation | One oversized undo entry bypasses 512 MiB trimming; hot entries are restored to GPU by count, not GPU bytes; CPU offload uses pinned tensors |
| Lazy tensor IR | Metadata node and tensor-id maps for tensor expressions | Process lifetime; only a test-only clear exists | Node telemetry count only | Production registry never removes dead tensors and is always active |

### 1.2 Peak composition: typical training with GUI

Reference configuration for a static estimate:

- MRNF defaults, SH degree 3, `max_cap=5,000,000`, 2,000,000 live Gaussians;
- FastGS fused backward, so no full 248 B/G gradient set;
- one 1920x1080 training image, unmasked 8-bit loader, prefetch 8;
- fused L1+SSIM; no previously touched alternate loss mode;
- GUI viewer active with zero-copy exportable model.

Known static residents are:

| Component | Formula | Estimate |
|---|---:|---:|
| Exportable SH3 model | `248 * max_cap` | 1.155 GiB |
| Quantized Adam | `172 * max_cap` | 0.801 GiB |
| MRNF capacity trackers + live densification info | `9 * max_cap + 8 * N` | 56.8 MiB |
| Active fused loss workspace at 1080p | `34 * H * W` | 67.2 MiB |
| Eight ready 8-bit RGB images | `8 * 3 * H * W` | 47.5 MiB |
| All slab classes + scalar/CUB minimums, if touched | `55.75 + 32 + 4 MiB` | 91.75 MiB |
| **Known subtotal** |  | **about 2.22 GiB** |

Masks and depths add 63.3 MiB each at this queue depth. The true process peak is:

`2.22 GiB + raster/shared-scratch committed + VkSplat/Vulkan-owned + CUDA/Vulkan context/modules + allocator high-water`.

The shared scratch alone has the prior-art 384 MiB floor. A practical working envelope is therefore roughly **3.2-5.5 GiB** for this reference configuration, but the renderer, arena, and driver terms must be replaced by the actual `vksplat.memory`, arena-stat, VMA, and process samples; static code cannot make those scene/camera-dependent terms exact.

The dangerous loader configuration is much larger. At 16 in-flight 4K images with 16-bit color, masks, and depths, final RGB is 1.48 GiB, source u16 staging can be 0.74 GiB, masks are 0.49 GiB, and depths are 0.49 GiB: **about 3.21 GiB** for the loader envelope while the tuner prices about 0.49 GiB. With the 4K fused-loss workspace, the known subtotal becomes about **5.58 GiB before arena, renderer, and context memory**.

### 1.3 A defensible recoverable total

For one stated, non-double-counted phase: a completed 10M-Gaussian SH3 run at 4K after writing a checkpoint, with the model kept for viewing and without undo/clipboard/mesh-cache payloads:

- completed-training optimizer + active fused workspace: about **1.86 GiB VRAM** recoverable while preserving the model;
- pinned checkpoint cache: about **4.94 GiB host memory** recoverable;
- Lazy IR at an illustrative 100 recorded nodes/iteration for 30k iterations: about **0.56-0.73 GiB host memory** recoverable.

That is **about 7.4-7.6 GiB total recoverable memory** in the stated phase. MRNF tracking buffers would add a small additional amount. This total intentionally excludes max-cap slack, undo snapshots, loader peaks, background variants, mesh cache, the model required for viewing, renderer resources, and the arena, so the larger per-finding peaks below are not falsely summed.

## 2. Findings

### 2.1 Leaks

| ID | Severity | Area | File:line@hash | Description | Estimated bytes | Evidence |
|---|---|---|---|---|---:|---|
| L-01 | **Leak / Critical** | Tensor lazy IR | `src/core/tensor/lazy_ir.cpp:29-60,88-98@c08091557853014b99b68834980ed42661baee8f`; `src/core/tensor/tensor.cpp:650-657@c08091557853014b99b68834980ed42661baee8f` | The production Lazy IR is always active. Every expression inserts a node and tensor-id mapping; tensor destruction does not unregister it. The only clear function is explicitly test-only. CPU memory therefore grows with total tensor operations for the entire process. | Roughly `200-260 B * nodes`. At 100 nodes/iter * 30k = 3M nodes: **0.56-0.73 GiB**; growth is unbounded. | Global `nodes` and `tensor_to_node`; unconditional `return true`; insert-only registration; no destructor hook or production GC. |
| L-02 | **Leak / Critical** | Pinned allocator fallback | `src/core/tensor/pinned_memory_allocator.cpp:155-175,238-240,268-299@c08091557853014b99b68834980ed42661baee8f`; `src/core/include/core/pinned_memory_allocator.hpp:149-182@c08091557853014b99b68834980ed42661baee8f` | On `cudaHostAlloc` failure the allocator falls back to `malloc`, but neither live nor cached block metadata records the backend. `empty_cache` later calls `cudaFreeHost` for every block, logs the error, clears the pointer from the map, and never calls `free`. | Up to the failed bucket size per block. A checkpoint SH allocation is **2 GiB**; two simultaneous SH buffers can strand **4 GiB**. | Fallback and pinned pointers share identical `AllocationInfo`/`Block`; the only cache release is `cudaFreeHost`. |
| L-03 | **Leak / High** | GUI training exportable storage | `src/visualizer/training/training_manager.cpp:85-125,259-293@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/scene/scene_manager.cpp:1128-1144@c08091557853014b99b68834980ed42661baee8f`; `src/core/splat_exportable_storage.cpp:49-88@c08091557853014b99b68834980ed42661baee8f` | `clearTrainer()` resets the trainer but not `splat_storage_`. Scene clear then destroys the model, leaving the manager optional as the final owner of the full physical VMM block until another training allocator is created or the manager dies. | SH3 `248 * max_cap`: **1.155 GiB at 5M**, **2.310 GiB at 10M**. | The optional is reset only when creating the next allocator; scene teardown calls trainer clear before `scene_.clear()`. |
| L-04 | **Leak / High** | Viewer mesh CPU ray-pick cache | `src/visualizer/scene/scene_manager.cpp:1549-1560,1610-1645@c08091557853014b99b68834980ed42661baee8f` | Global cache entries are keyed by raw `MeshData*`, survive scene clear, and have no byte budget or teardown hook. It clears only on the next miss after reaching 64 entries. Besides retaining data, pointer reuse plus a matching generation can return stale geometry. | Example 10M vertices + 20M triangles: logical 343 MiB, allocator-rounded **384 MiB pinned per mesh**. 64 such entries would be **24 GiB**. | Global map; only `size >= 64 -> clear`; CPU tensor copies are stored in entries. |
| L-05 | **Leak / Low** | Trainer construction error path | `src/training/trainer.cpp:1797-1833@c08091557853014b99b68834980ed42661baee8f` | The scene constructor creates three streams, events, and pinned loss slots before checking `scene.hasTrainingData()`. Throwing at that validation point bypasses `Trainer::~Trainer`, so all raw handles/slots leak. CUDA creation return codes are also ignored. | Three streams, event arrays, and `LOSS_RING * sizeof(float)` pinned; driver bytes are implementation-defined. | Validation throw occurs after `createSyncPrimitives`; cleanup is only in `shutdown()`/destructor. |

### 2.2 Stability / OOM and corruption paths

| ID | Severity | Area | File:line@hash | Description | Estimated bytes | Evidence |
|---|---|---|---|---|---:|---|
| S-01 | **Stability / Critical** | Async CPU->GPU copies | `src/core/tensor/tensor.cpp:593-608,961-1018,1108-1124@c08091557853014b99b68834980ed42661baee8f`; `src/core/tensor/tensor_unified_ops.cpp:162-173@c08091557853014b99b68834980ed42661baee8f` | Explicit-stream H2D paths call `set_stream` on the CPU source. `set_stream` informs only the CUDA pool; it does not call the pinned allocator’s `record_stream`. The CPU deleter captured the allocation-time stream, so a temporary source can be returned to the pinned cache while another stream is still reading it. This is a use-after-recycle/data-corruption path, not just waste. | Any source tensor; checkpoint/image buckets range from MiB to **2 GiB**. | The correct CPU branch exists in `record_stream`, but transfer paths call `set_stream` instead. |
| S-02 | **Stability / Critical** | Tensor capacity growth | `src/core/tensor/tensor.cpp:2651-2760@c08091557853014b99b68834980ed42661baee8f`; `src/core/include/core/cuda_debug.hpp:14-22,60-61@c08091557853014b99b68834980ed42661baee8f` | `reserve()` uses non-throwing `CHECK_CUDA(cudaMalloc)`. On OOM it continues with null `new_data`, copies into it, installs it, and releases the old owner. Size multiplication also lacks overflow checks. | SH3 shN growth to 10M requests **1.79 GiB**; growing from 6.67M holds roughly **2.98 GiB** old+new at the failure point. | `CHECK_CUDA` only logs; `reserve` treats the call as successful and commits state unconditionally. |
| S-03 | **Stability / High** | Image-loader VRAM budgeting | `src/training/trainer.cpp:907-1002,4572-4604@c08091557853014b99b68834980ed42661baee8f`; `src/io/pipelined_image_loader.cpp:699-765@c08091557853014b99b68834980ed42661baee8f`; `src/training/dataset.hpp:632-669@c08091557853014b99b68834980ed42661baee8f` | The tuner always estimates RGB as 3 B/pixel, adds at most 1 B/pixel for a mask, and ignores 16-bit-to-float output, depth, and concurrent decode staging. Non-JPEG input raises prefetch to 16 and cold workers to half the CPU. Count-bounded is not byte-bounded. | 16x4K, 16-bit RGB+mask+depth+u16 staging: **3.21 GiB actual envelope vs about 0.49 GiB priced**, an undercount of **2.72 GiB**. | Final 16-bit training tensor is float32 RGB (12 B/pixel); source staging is 6 B/pixel; mask/depth are each 4 B/pixel. |
| S-04 | **Stability / High** | Checkpoint load peak | `src/core/tensor/internal/tensor_serialization.hpp:55-86@c08091557853014b99b68834980ed42661baee8f`; `src/core/splat_data.cpp:1077-1124@c08091557853014b99b68834980ed42661baee8f` | Deserialize first materializes five CPU tensors, uploads each through a temporary CUDA tensor, allocates the final external destination, then uploads the entire canonical shN before swizzling into the final block. The temporary exists beside the full destination and viewer/context memory. | 10M SH3: 2.31 GiB final model + allocator-rounded **1.75 GiB canonical shN temp = 4.06 GiB VRAM**, before optimizer/renderer. The avoidable part is about **1.75 GiB**. | Whole-tensor stream operators and `reorder_canonical_into_swizzled(shN_canon.cuda(), final_ptr, ...)`. |
| S-05 | **Stability / High** | Raw training scratch allocation | `src/training/kernels/mrnf_kernels.cu:191-255,315-428@c08091557853014b99b68834980ed42661baee8f`; `src/training/rasterization/gsplat/Rasterization.cpp:380-432@c08091557853014b99b68834980ed42661baee8f`; `src/training/kernels/mcmc_kernels.cu:918-952,1025-1049@c08091557853014b99b68834980ed42661baee8f` | MRNF sort, gsplat backward, and MCMC scan allocate raw async scratch in hot/refine paths without checking return status. A failed allocation is immediately passed to kernels/CUB/memset. Repeated alloc/free also raises pool high-water and fragmentation. | MRNF Gumbel arrays are `24*N`: **229 MiB at 10M**, plus CUB temp. Percentile uses 76 MiB + CUB; gsplat RGB gradient is `12*N` for one RGB camera = 114 MiB. | Bare `cudaMallocAsync` followed by unconditional use. |
| S-06 | **Stability / High** | Undo snapshots | `src/visualizer/operation/undo_history.cpp:370-388,425-437,446-505@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/operation/undo_entry.cpp:679-698,832-886@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/gui_capabilities.cpp:748-783@c08091557853014b99b68834980ed42661baee8f` | Trim refuses to remove the last entry even when it exceeds 512 MiB, while residency restores the newest five entries to their preferred device. A bake-transform entry deep-clones the model both before and after. One normal edit can therefore allocate several times the model while GUI training/viewing is already near its peak. | 10M SH3 bake: logical before+after **4.62 GiB**; pool-rounded physical blocks about **5.16 GiB retained VRAM**. With the live model, operation-time model storage is about 7.47 GiB. | Deep `clone()` of every splat tensor; two FULL captures; `size > 1` trim guard; hot restore. |
| S-07 | **Stability / High** | Pinned host cache | `src/core/tensor/pinned_memory_allocator.cpp:91-105,126-175,184-240,268-299@c08091557853014b99b68834980ed42661baee8f`; `src/core/tensor/tensor.cpp:634-639@c08091557853014b99b68834980ed42661baee8f` | Every free block enters an unbounded per-size cache. GPU pool trim does not trim pinned memory. Multi-GiB page-locking survives checkpoint completion and can exhaust OS pin limits or force later `cudaHostAlloc` failures into L-02. The HUD reports active bytes only, so the cache appears to vanish. | One 10M SH3 model+Adam checkpoint leaves an estimated **4.94 GiB pinned cache** (two 2 GiB SH blocks, one 512 MiB Adam SH bucket, plus 256/128/64 MiB classes). | No cache budget, pressure eviction, TTL, or production trim call; cached bytes are tracked internally but not published. |
| S-08 | **Stability / Medium** | Checkpoint save peak | `src/core/splat_data.cpp:678-713,1026-1053@c08091557853014b99b68834980ed42661baee8f`; `src/core/tensor/internal/tensor_serialization.hpp:24-52@c08091557853014b99b68834980ed42661baee8f`; `src/training/optimizer/adam_optimizer.cpp:1174-1218@c08091557853014b99b68834980ed42661baee8f` | Saving shN allocates a canonical CPU tensor while a full swizzled CPU download is still live. Serialization is whole-tensor, and Adam adds another large size class. This creates a page-locked host spike at exactly the phase where training and GUI VRAM are also high. | 10M SH3: two simultaneous **2 GiB pinned blocks** for shN; about **4.44 GiB** pinned resident by the model save and **4.94 GiB** cached after Adam. | `out = Tensor::empty(CPU)` plus `_shN.cpu()` in the same scope; all tensor stream operators materialize complete CPU tensors. |

### 2.3 Waste and high-water retention

| ID | Severity | Area | File:line@hash | Description | Estimated bytes | Evidence |
|---|---|---|---|---|---:|---|
| W-01 | **Waste / High** | Max-capacity policy | `src/visualizer/training/training_manager.cpp:85-120@c08091557853014b99b68834980ed42661baee8f`; `src/core/splat_exportable_storage.cpp:49-88@c08091557853014b99b68834980ed42661baee8f`; `src/training/optimizer/adam_optimizer.cpp:138-183,262-290@c08091557853014b99b68834980ed42661baee8f` | GUI training physically commits the full exportable model and quantized Adam capacity at startup. Virtual address reservation is cheap, but this code commits physical memory, so low live utilization directly burns VRAM. | SH3 model+Adam slack is `420 * (max_cap-N)`. At 2M live / 5M cap: **1.17 GiB**; at 2M / 10M: **3.13 GiB**, before strategy-specific buffers or gradients. | Exportable capacity is `max(max_cap,min)`; `create` commits the whole block; Adam state uses initial max capacity. |
| W-02 | **Waste / High** | Finished training lifecycle | `src/training/trainer.cpp:4756-4810@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/training/training_manager.cpp:929-973@c08091557853014b99b68834980ed42661baee8f`; `src/training/trainer.hpp:411-469@c08091557853014b99b68834980ed42661baee8f` | Successful completion clears the active loader/CPU image cache but keeps the trainer, optimizer, strategy state, datasets, backgrounds, and all loss workspaces while transitioning to Finished. Viewing needs the model and perhaps appearance model, not Adam/densification state. | Optimizer alone: **0.80 GiB at 5M**, **1.60 GiB at 10M**. Full gradients, if materialized, add 1.15/2.31 GiB. Workspaces add tens of MiB to GiB. | No finalize-for-viewing/release phase before `transitionToFinished`; manager retains `trainer_`. |
| W-03 | **Waste / High** | Loss workspace modes | `src/training/include/lfs/kernels/ssim.cuh:14-49,128-164,200-232,263-291,318-352@c08091557853014b99b68834980ed42661baee8f`; `src/training/trainer.hpp:447-466@c08091557853014b99b68834980ed42661baee8f` | Trainer owns standard, fused, masked, decoupled, masked-decoupled, and densification SSIM workspaces simultaneously. Changing modes/resolutions replaces or adds buffers but never releases inactive variants. | At 4K: standard 570 MiB, fused 269 MiB, masked 411 MiB, each decoupled variant 696 MiB, densification map+error about 127 MiB. If all are touched: **about 2.70 GiB retained**. | Separate member objects with shape-high-water tensors; no active-mode union or release. |
| W-04 | **Waste / Medium** | Trainer teardown ordering | `src/training/trainer.cpp:2855-2919@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/training/training_manager.cpp:259-293@c08091557853014b99b68834980ed42661baee8f` | `shutdown()` trims the CUDA pool before Tensor-valued trainer members are destroyed. Their destructors run after `shutdown` returns and put blocks back into the bucket cache; `clearTrainer` does no post-reset trim. | Normally up to the **256 MiB cache budget**; potentially one oversized multi-GiB block because of W-05. | Strategy is reset before trim, but member workspaces/backgrounds are not; manager resets trainer after its only trim. |
| W-05 | **Waste / High** | Size-bucketed pool | `src/core/tensor/internal/size_bucketed_pool.hpp:29-33,58-71,89-103,339-367@c08091557853014b99b68834980ed42661baee8f` | Budget enforcement explicitly stops when one cached entry remains. The nominal 64-256 MiB budget is therefore not a hard budget. | A freed 10M shN/copy bucket is **2 GiB**, leaving **1.75 GiB above** the 256 MiB maximum. Larger requests can retain up to the 16 GiB tracked ceiling. | `if (cached_entry_count() <= 1) break`. |
| W-06 | **Waste / High** | Internal VMM arena reset | `src/core/cuda/memory_arena.cu:601-633,1077-1141@c08091557853014b99b68834980ed42661baee8f`; `src/core/cuda/memory_arena.hpp:25-31@c08091557853014b99b68834980ed42661baee8f` | `full_reset()` calls `decommit_unused_memory` before setting offset to zero. After a high-water frame, almost no chunk starts beyond the old offset, so the reset preserves committed high-water. This affects internal VMM/headless mode; borrowed GUI shared scratch is a different prior-art path. | Example 2 GiB committed with the default 128 MiB first chunk: a correct reset recovers **1.875 GiB**. | Decommit reads current offset and preserves chunk zero; offset is zeroed only afterward. |
| W-07 | **Waste / Medium** | Background image cache | `src/training/trainer.hpp:421-425@c08091557853014b99b68834980ed42661baee8f`; `src/training/trainer.cpp:3110-3153@c08091557853014b99b68834980ed42661baee8f` | One full float32 RGB background is cached for every unique camera resolution, with no entry/byte budget. Variable-resolution datasets grow it monotonically until mode/path change or trainer destruction. | `12*H*W` each. Ten 1080p sizes: **237 MiB**; ten 4K sizes: **949 MiB**. | Dimension-keyed map inserts every miss; only mode/path changes clear it. |
| W-08 | **Waste / High** | Archival CPU tensors | `src/visualizer/scene/scene_manager.cpp:87-100,4241-4275@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/operation/undo_entry.cpp:314-343@c08091557853014b99b68834980ed42661baee8f`; `src/core/tensor/tensor_unified_ops.cpp:158-185@c08091557853014b99b68834980ed42661baee8f` | `.cpu()` always allocates pinned memory. Long-lived clipboard models, offloaded undo history, and mesh ray-pick mirrors therefore consume scarce page-locked memory even though they are archival rather than in-flight transfer buffers. | A full 10M SH3 clipboard model is allocator-rounded **2.69 GiB pinned**. Offloading a before+after full undo entry is about **5.38 GiB pinned**. | CPU tensor factory defaults to pinned; all three archival call sites use `.cpu()`/`to(CPU)` with no unpinned option. |
| W-09 | **Waste / Medium** | GPU slab allocator | `src/core/tensor/internal/gpu_slab_allocator.hpp:29-37,160-165,227-275@c08091557853014b99b68834980ed42661baee8f`; `src/core/tensor/internal/memory_pool.hpp:321-342@c08091557853014b99b68834980ed42661baee8f` | Slabs expand on demand and are released only at shutdown. Explicit trim only merges free lists. Fully idle slabs cannot be returned after a phase spike. | Touching all 11 classes commits **55.75 MiB** minimum; every additional slab adds up to 8 MiB. | No per-slab live count or idle-slab trim path. |
| W-10 | **Waste / Medium** | Global tensor scratch | `src/core/tensor/tensor_ops.cu:42-80,82-142@c08091557853014b99b68834980ed42661baee8f` | Process-global CUB scratch grows and never shrinks. The first scalar reduction also reserves a fixed 32 MiB plus scalar buffers. These use raw CUDA allocation and bypass pool labels. | **36 MiB minimum** once both paths are touched, plus arbitrary CUB high-water. | Static singletons; only destructors free; no trim/accounting hook. |
| W-11 | **Waste / Medium** | Tensor conversion semantics | `src/core/tensor/tensor.cpp:703-713,913-925,1151-1165,1217-1241@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/scene/scene_manager.cpp:4294-4314,4668-4692@c08091557853014b99b68834980ed42661baee8f` | `to(same_device)` and `to(same_dtype)` deep-clone, unlike `contiguous()` which is shallow when already contiguous. Float32->Bool on CUDA takes a full D2H float vector, CPU bool tensor, then H2D instead of one kernel. Call sites must defensively check device/dtype to avoid full copies. | Same-device/dtype cost is the full tensor. Float->Bool at 100M elements allocates **476.8 MiB host logical + 95.4 MiB GPU output** and transfers 500 MB decimal. | Explicit `return clone()` and CPU fallback conversion path. |
| W-12 | **Waste / Medium** | Preprocess CPU pipeline | `src/preprocessing/preprocess.cpp:95-99,502-535,948-954,1030-1063@c08091557853014b99b68834980ed42661baee8f` | `resize_for_inference(const Image&)` returns by value even when no resize is needed, deep-copying float RGB. Three prefetched loads plus the current item can hold four duplicate inference copies. | One 4K float RGB image is 94.9 MiB; four avoidable copies are **379.7 MiB CPU**. | Return-by-value from const lvalue at both no-resize exits; prefetch depth 3 plus current. |
| W-13 | **Waste / Low-Medium** | Legacy nvImageCodec pool | `src/io/cache_image_loader.cpp:571-587@c08091557853014b99b68834980ed42661baee8f`; `src/io/pipelined_image_loader.cpp:101-159@c08091557853014b99b68834980ed42661baee8f`; `src/io/nvcodec_image_loader.cpp:621-690,698-766@c08091557853014b99b68834980ed42661baee8f` | Legacy JPEG decode intentionally leaks an 8-decoder loader for process lifetime while the pipeline owns separately keyed pools. Both can coexist. Normal pipeline entries are owner-counted and released; the legacy one is not. | Driver-defined; use the existing **`io.nvimagecodec`** scope for the exact baseline/lazy delta. | Deliberate `new` singleton comment; separate pipeline cache by decoder-pool size. |
| W-14 | **Waste / Low** | Loader initialization bandwidth | `src/io/pipelined_image_loader.cpp:1417-1435,1533-1541,1659-1672@c08091557853014b99b68834980ed42661baee8f` | Several output tensors use `zeros()` immediately before a kernel overwrites every element. This does not reduce peak bytes, but adds full-buffer memset traffic and allocator work in loader hot paths. | One 4K float RGB zero is **94.9 MiB of redundant writes**; alpha/depth adds 31.6 MiB each per occurrence. | Conversion/split kernels populate every output element. |
| W-15 | **Waste / Low** | Non-VMM arena growth | `src/core/cuda/memory_arena.cu:1258-1296,1319-1389@c08091557853014b99b68834980ed42661baee8f` | Fallback caller adds 2x the deficit to current committed size, then `grow_arena` treats that as a required size and doubles it again. Large growth requests can overshoot far beyond the requested working set before 128 MiB rounding/max cap. | Example 512 MiB full arena + 512 MiB request: caller asks 1.5 GiB; grow targets **3 GiB**, versus 2 GiB from doubling the actual 1 GiB requirement: about **1 GiB extra**. | Two independent growth multipliers on the non-VMM path. |

### 2.4 Measurement defects (do not count as recovered bytes)

| ID | Severity | Area | File:line@hash | Description | Evidence |
|---|---|---|---|---|---|
| M-01 | **Measurement / High** | Cross-allocator accounting | `src/diagnostics/vram_profiler.cpp:621-652,687-710,1191-1226@c08091557853014b99b68834980ed42661baee8f`; `src/visualizer/gui/vram_hud_overlay.cpp:398-416,1036-1055@c08091557853014b99b68834980ed42661baee8f`; `src/core/cuda/memory_arena.cu:1062-1066,1365-1368@c08091557853014b99b68834980ed42661baee8f` | Profiler misses allocations made while disabled; pinned gauge omits cached bytes; arena chunks are classified Direct despite an Arena category; slab requested bytes are included in `accounted_cuda_pool_live_bytes` even though slabs are raw `cudaMalloc`, biasing the HUD’s default-pool “untracked” subtraction; direct tensor capacity, global CUB/scalar scratch, and transient loader staging are outside normal allocation rows; exportable splat accounting is one scalar rather than per-block. | Recording returns early while disabled; method switch combines Slab with default-pool methods; arena passes Direct; HUD displays active pinned only. |

## 3. Ranked fix plan for the follow-up round

| Rank | Exact change | Files | Risk | Expected recovery / stability gain | Verification |
|---:|---|---|---|---|---|
| 1 | Make Lazy IR opt-in to actual deferred execution, or add production weak-owner pruning/unregister and a bounded registry. Publish current node/map counts. | `src/core/tensor/lazy_ir.cpp`, tensor lifecycle/internal lazy headers | **High — tensor library** | Stops unbounded host growth; hundreds of MiB to GiB over long training. | Node counts plateau over 7k/30k; RSS slope flattens; heaptrack/LSan confirms no retained `LazyExprNode` chain. |
| 2 | Add allocation-backend provenance to pinned `Block`/`AllocationInfo`; `free` fallback blocks; do not cache fallback memory. Add a byte budget, pressure/TTL eviction, `trim_pinned_cache()`, and publish active + cached + peak separately. | `pinned_memory_allocator.hpp/.cpp`, tensor public trim surface, profiler/HUD | **Medium-high — core tensor/IO** | Fixes real fallback leaks and recovers up to 4.94 GiB after the reference checkpoint. | Force a small pin limit/fault injection; active and cached gauges return to baseline; ASan/LSan sees fallback freed; `/proc/$PID/status` `VmLck` drops after trim. |
| 3 | In CPU->GPU async transfer paths call source `record_stream(transfer_stream)`; do not use `set_stream` as a substitute. Audit all explicit-stream H2D paths. | `src/core/tensor/tensor.cpp` | **High — tensor library synchronization** | Removes use-after-recycle/corruption for buffers of any size. | Stress explicit-stream transfers while immediately destroying/reusing sources; compute-sanitizer memcheck/racecheck where supported; checksum output across repeated runs. |
| 4 | Make `reserve` failure-atomic: checked size multiplications, explicit CUDA status, trim/retry, throw while the old owner/state remains intact, and commit new state only after successful copy. | `src/core/tensor/tensor.cpp`, CUDA error helpers | **High — tensor library** | Converts near-OOM corruption into a recoverable error; avoids losing a multi-GiB model. | Constrain free VRAM and request growth; old pointer/shape/capacity/content remain valid after exception; memcheck clean. |
| 5 | Reset `splat_storage_` during trainer/scene teardown after GPU drain and renderer/model ownership transition. Keep shared ownership so edit-mode model remains valid; remove only the manager’s stale owner. Make exportable accounting per-block/additive. | `training_manager.cpp/.hpp`, possibly `scene_manager.cpp`, `exportable_storage.cpp`, profiler | **Medium — GUI training/Vulkan ownership** | 1.15 GiB at default MRNF 5M, 2.31 GiB at 10M after scene clear. | Clear to empty and load a non-training scene; process VRAM and exportable-block count drop exactly once; Vulkan validation remains clean; switch-to-edit still renders. |
| 6 | Price loader requests in bytes, not count: output dtype, RGB/mask/depth, undistort intermediates, decoder/source staging, and worker concurrency. Enforce a byte semaphore and adapt prefetch/workers on current free VRAM. | `trainer.cpp`, `pipelined_image_loader.hpp/.cpp`, `dataset.hpp` | **Medium — training/IO behavior** | Prevents up to 2.72 GiB under-budgeted 16x4K envelope; graceful depth/mask/16-bit degradation instead of OOM. | Run 8/16-bit, JPEG/non-JPEG, mask/depth matrix; loader byte counters + process sample stay under target; output parity/checksums. |
| 7 | Stream checkpoint save/load in fixed 32-64 MiB chunks. Deswizzle/swizzle shN by chunk directly between final storage and reusable pinned staging; load directly into final parameter/optimizer destinations. Trim pinned/cache at phase boundary. | `tensor_serialization.hpp`, `splat_data.cpp`, `checkpoint.cpp`, `adam_optimizer.cpp` | **High — core tensor + checkpoint compatibility + training** | Save: removes 4 GiB live SH staging and 4.94 GiB retained cache. Load: removes about 1.75 GiB VRAM temp and most whole-tensor pinned peak. | Byte-for-byte or semantically equivalent checkpoint round-trip at SH0-3; resume at 7k; nvidia-smi/RSS/VmLck phase traces; corrupted/truncated input still fails safely. |
| 8 | Add `Trainer::finalize_for_viewing()`: after the final save, release optimizer, densification-only strategy state, gradients, loader refs, eval/loss/edge/depth/background transients while retaining model and required PPISP viewing state. Gate resumable behavior explicitly. | `trainer.cpp/.hpp`, strategy/optimizer interfaces, `training_manager.cpp` | **High — training lifecycle** | At 10M, at least 1.60 GiB optimizer plus active/inactive workspaces; about 1.86 GiB in the reference completion state. | Compare final model render/checkpoint before/after; Finished-state VRAM drop; stop/resume semantics explicitly tested by smoke workflow, not assumed. |
| 9 | Replace all concurrently resident loss workspaces with an active-mode variant/shared scratch layout. Clear inactive variants on mode/resolution changes and before pool trim. | `trainer.hpp/.cpp`, `photometric_loss.*`, `ssim.cuh` | **Medium-high — training kernels/workspace lifetime** | Up to 2.70 GiB at 4K after all modes have been touched. | Cycle every mask/appearance mode and 1080p/4K; profiler current bytes follow only active mode; loss/gradient parity. |
| 10 | Make undo GPU residency byte-budgeted. Reject/drop or immediately offload entries over the total budget; store archival snapshots in unpinned/compressed host or disk storage; for bake, keep one prior snapshot plus deterministic redo/delta instead of two full models. | `undo_history.*`, `undo_entry.*`, `gui_capabilities.cpp`, CPU tensor API | **Medium-high — visualizer/undo semantics** | 5.16 GiB retained VRAM for the 10M bake example; avoids 5.38 GiB pinned when offloaded. | Bake/undo/redo exact model parity; history GPU bytes never exceed configured hard cap; OOM-pressure shrink is automatic; no pinned-cache jump. |
| 11 | Make size-bucket budget hard under pressure: oversized singleton TTL/phase eviction, explicit pressure trim, and optional “keep one” only below a small threshold. Release fully idle slabs on explicit trim using per-slab live ownership. | `size_bucketed_pool.hpp`, `gpu_slab_allocator.hpp`, `memory_pool.hpp` | **Medium — allocator performance** | 1.75 GiB above budget for a retained 2 GiB block; 55.75 MiB+ idle slabs. | Cache/slab gauges fall after phase trim; allocation throughput A/B; stream-order stress and memcheck. |
| 12 | Zero arena offset before decommit inside `full_reset` after frame/release drain; fix fallback growth to take actual `total_needed` exactly once. Classify arena allocations as Arena. | `memory_arena.cu/.hpp` | **Medium — core rasterizer arena** | 1.875 GiB in the 2 GiB internal-VMM example; up to about 1 GiB fallback overgrowth example. | Required headless 7k smoke ends near the 128 MiB first chunk; committed/peak/realloc log; VMM and forced non-VMM paths; memcheck leak mode. |
| 13 | After `trainer_.reset()`, perform the pool/pinned trim, not before member destruction. Add a single teardown routine that clears Tensor members before trim. | `trainer.cpp/.hpp`, `training_manager.cpp` | **Medium — training teardown ordering** | Normally up to 256 MiB pool cache, or an oversized singleton until rank 11 lands. | Finished -> clear -> empty scene trace returns pool used/reserved and bucket cache to baseline. |
| 14 | Add byte-bounded LRU for background images and mesh CPU mirrors; erase mesh entries on scene/node destruction and key by stable ownership/generation rather than raw pointer. Use unpinned CPU storage. | `trainer.cpp/.hpp`, `scene_manager.cpp`, CPU tensor factory/API | **Low-medium** | Background: up to 949 MiB for ten 4K sizes. Mesh example: 384 MiB each, up to 24 GiB envelope. | Vary resolutions/remove scenes repeatedly; byte gauges plateau; ray-pick correctness after address reuse. |
| 15 | Introduce reusable checked workspaces for MRNF/MCMC/gsplat scratch and use tensor/pool RAII. Add a direct CUDA Float32->Bool kernel; make no-op `to` shallow or add an explicit `clone_if_same` compatibility transition. | MRNF/MCMC kernels, gsplat rasterizer, tensor conversion code/call sites | **High — training + tensor semantic change** | 229 MiB+CUB MRNF churn; 476.8 MiB host Float->Bool staging at 100M; removes null-pointer OOM launches. | OOM injection returns an error/fallback; repeated refinement has flat alloc-event count; dtype/device call-site audit; output parity. |
| 16 | Remove no-resize preprocess copies via move/shared immutable image storage; replace overwrite-complete loader `zeros` with `empty`; consolidate or safely retire the legacy nvImageCodec singleton after measuring it. | `preprocess.cpp`, `pipelined_image_loader.cpp`, `cache_image_loader.cpp`, nvcodec lifecycle | **Low-medium** | 379.7 MiB CPU at four 4K loads; redundant 95-127 MiB writes/image; dynamic `io.nvimagecodec` baseline. | Preprocess output hashes; initcheck for every zeros->empty site; nvcodec scope returns to zero when no owner and shutdown remains clean. |
| 17 | Make profiler snapshots additive and allocator-correct: cached pinned bytes, per-exportable-block records, arena method, separate raw-slab versus default-pool totals, global scratch and loader transient gauges. Add a serializable snapshot/diff endpoint. | diagnostics profiler/HUD, allocators, loader, exportable storage | **Low runtime risk; medium diagnostics surface** | No direct recovery; makes every fix above measurable and prevents false “untracked” conclusions. | Sum reconciliation: process ~= tracked CUDA + Vulkan + driver/context + explicit gap; enabling before load gives stable alloc/free balance. |

Tensor-library and training edits are intentionally called out above. Ranks 1, 3, 4, 7, and 15 touch core tensor semantics or serialization; ranks 8-9 touch training lifecycle/numerics. They should not be bundled with low-risk cache cleanup in one fix.

## 4. Before/after measurement protocol

### 4.1 Common controls

1. Use the same binary, driver, GPU clocks/power state, dataset, seed, SH degree, strategy, `max_cap`, image resize, viewport size, and UI visibility for each A/B pair.
2. Start `VramProfiler` **before dataset/model load** (`lichtfeld.set_vram_profiler_enabled(True)` or the existing UI control). Enabling it after allocation misses live records by design.
3. Set `LFS_MEM_BREAKDOWN=1` for the training runs. Preserve the complete log, including tensor storage summaries, loader bytes, pool stats, `vksplat.memory`, and final `Arena stats`.
4. Record host `RSS`, `VmLck`, and `VmPin` (where available), not only GPU memory. Pinned-cache and Lazy-IR fixes primarily move those values.
5. Sample the process externally at 200 ms during each run, for example:

   ```sh
   nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits -lms 200 > nvidia-smi-memory.csv
   ```

6. Mark timestamps/phases: pre-context, empty GUI, dataset loaded, model initialized, first batch, first raster, steady iteration, immediately before/after refine, eval 7k, checkpoint 7k, post-save idle, training Finished, switch-to-edit, scene clear, and application exit.

### 4.2 Required headless 7k smoke

- Run the existing headless 7k training smoke command with `LFS_MEM_BREAKDOWN=1` and identical parameters before/after.
- Capture:
  - peak process VRAM from nvidia-smi;
  - `cuda_direct`/`vulkan_external` tensor storage summaries;
  - CUDA pool used/reserved and bucket/slab gauges;
  - final `Arena stats: committed, peak, reallocs` emitted at arena destruction;
  - process RSS/VmLck at initialization, pre-save, post-save, and exit.
- Acceptance after the arena reset fix: internal VMM committed memory returns to approximately the first 128 MiB chunk after full reset; peak remains historical; realloc count does not regress materially.
- Acceptance after checkpoint/pinned fixes: post-save VmLck/cache returns to the configured staging budget, rather than remaining near 4.94 GiB.

### 4.3 GUI training/viewing protocol

Use one fixed camera and viewport size, with profiler enabled before load. Capture the full HUD tree (or a serialized snapshot once rank 17 exists) and the corresponding `vksplat.memory` line at each milestone:

1. empty GUI;
2. dataset loaded but training not started;
3. after training model/optimizer initialization;
4. steady training at fixed `N`;
5. just before and after a densification/refine event;
6. before/after 7k eval and checkpoint;
7. Finished while still viewing;
8. switch to edit mode;
9. clear scene.

Reconcile these independent views:

- process VRAM: nvidia-smi/NVML;
- CUDA context and default pool: profiler process rows;
- model: exportable block and manual splat/optimizer breakdown;
- renderer: VMA plus `vksplat.memory` renderer-owned total;
- raster scratch: arena committed/peak, counting borrowed shared scratch only once;
- loader: pending/output byte counters plus a new transient-staging gauge;
- host pinned: active + cached + peak, plus OS VmLck;
- residual: process minus all of the above. A residual is a measurement task, not automatically a leak.

### 4.4 Targeted stress matrix

| Finding group | Stress | Pass condition |
|---|---|---|
| Lazy IR | 7k then 30k, same scene | Registry count and RSS reach a plateau relative to live deferred expressions |
| Pinned allocator/checkpoint | Save 10M SH3 twice; trim between; force pin-allocation failure once | Second save reuses bounded staging; fallback frees with the correct backend; VmLck returns to budget |
| Loader | 8/16-bit x JPEG/PNG x mask/depth x 1080p/4K | Byte semaphore stays under budget and gracefully reduces concurrency |
| Undo | 10M bake transform, undo/redo, pressure shrink | GPU history never exceeds hard byte cap; exact model parity |
| Training finalize | Complete, stop, resume-if-supported, view, switch-to-edit | Model/appearance output unchanged; optimizer/workspaces released only in non-resumable state |
| Scene teardown | Train -> clear -> load plain PLY -> clear, repeated | Exportable block count and process VRAM return to baseline every cycle |
| Mesh cache | Load/pick/remove varying large meshes, including pointer reuse | Cache byte cap holds; no stale ray-pick geometry |

### 4.5 Sanitizers and allocation-failure proof

- CUDA allocation/handle leaks: `compute-sanitizer --tool memcheck --leak-check full <smoke command>` for bounded headless variants. Repeat around checkpoint load/save, refine, scene teardown, and forced non-VMM arena operation.
- Uninitialized output after `zeros -> empty`: use compute-sanitizer initcheck on the exact loader matrix before accepting each replacement.
- Host ownership: ASan/LSan for constructor and fallback paths; heaptrack (or equivalent) for Lazy IR and global cache growth.
- OOM behavior: run under constrained free VRAM or allocator fault injection. `reserve`, MRNF/MCMC/gsplat scratch, and loader creation must return a controlled error/degrade while existing tensors remain valid.
- Do not interpret process-exit driver reclamation as proof of correct module/scene teardown; the critical deltas are Finished, edit-mode switch, and scene clear while the process stays alive.

There is no public serialized `VramProfiler` dump API at this commit; the current practical “dump” is the HUD/store snapshot plus logs. Adding a JSON/CSV snapshot/diff endpoint is part of fix-plan rank 17.

## 5. Considered and rejected / not re-reported

| Candidate | Disposition |
|---|---|
| Vulkan output-image rings, shared-scratch floor/headroom, readback staging, GT preparation cache, cached-frame synchronization, environment cache, RmlUi retirement, LOD ring | Prior art in `docs/vulkan_rendering_analysis_2026_07.md`; deliberately excluded. Shared scratch remains in the memory map only so it is counted once. |
| 32 GiB arena virtual reservation | Rejected as physical waste: VMM address space is virtual and free until chunks are committed. The reset-order finding concerns committed physical chunks. |
| `CudaEventPool` | Rejected as a leak: pool is bounded at 512 events, destroys overflow immediately, and destroys pooled events at shutdown (`src/core/tensor/cuda_event_pool.cpp:14-59@c08091557853014b99b68834980ed42661baee8f`; header `:15-55`). |
| `DeferredFreeQueue` | Rejected as an active growth source: it is bounded by work submitted, flushes on trim/shutdown, and no current tensor-pool free path calls `defer_free`. Retain as dead-code/complexity cleanup, not a memory finding. |
| Lazy executor materialization cache/registries | Rejected as the same leak as L-01: execution cache is local to one materialization context, and deferred/fusion registries prune expired weak owners. The separate Lazy IR node graph is the unbounded object. |
| `contiguous()` on already contiguous tensors | Rejected: it returns a shallow copy (`src/core/tensor/tensor.cpp:703-713@c08091557853014b99b68834980ed42661baee8f`). Same-device/dtype `to`, not `contiguous`, is the hidden deep-copy issue. |
| Pipeline JPEG CPU cache | Rejected as unbounded: it has a configured 4 GiB default byte cap, LRU eviction, and physical-memory pressure reduction (`src/io/pipelined_image_loader.cpp:795-839@c08091557853014b99b68834980ed42661baee8f`). Four GiB may still be an aggressive default, but it is not a leak. |
| Legacy `CacheLoader` CPU/JPEG maps | Rejected as leaks: they have memory-pressure LRU eviction and explicit clears; training completion clears the CPU cache. The process-lifetime nvImageCodec object is reported separately. |
| Pipeline nvImageCodec instance cache | Rejected as a leak: normal entries are keyed by pool size, owner-counted, erased, and released outside the mutex at owner count zero (`src/io/pipelined_image_loader.cpp:101-159@c08091557853014b99b68834980ed42661baee8f`). |
| Mirror-operation multiplier cache | Rejected as material: it is a fixed set of tiny tensors (three position, three quaternion, nine small SH multipliers), not data-size proportional. |
| Mesh importer `TextureLoader` cache | Rejected as a persistent cache: the loader is local to one mesh load, so its map dies after import. |
| Vulkan mesh GPU cache | Rejected as unbounded: it has last-used retirement and full destruction. L-04 is a different global CPU ray-pick cache. |
| CudaMemoryPool deallocation after allocator shutdown | Rejected as a live-session leak: late deallocation returns without freeing, but this is static-destruction/process teardown where the CUDA context/OS reclaims memory. It remains teardown-order technical debt, not a training/viewing recovery. |
| Arena move assignment overwriting existing storage | Not reported: static search found no demonstrated call site that move-assigns into a live arena. Keep for a focused ownership audit if that API becomes used. |
| Undo history “bounded at 512 MiB” | Refuted as a rejection: the `stack.size() > 1` condition makes the bound soft for exactly the dangerous single oversized entry, and GPU residency is count-based. Hence S-06 remains. |
| Renderer imported model views | Rejected as duplicate VRAM: the GUI training path imports the same exportable physical block into Vulkan. Only extra owners/lifetime, private renderer buffers, or fallback copies count. |

## 6. Recommended order of attack

The safest high-value first fix batch is: L-03 manager owner release, L-04 mesh-cache lifetime/byte cap, W-07 background LRU, W-04 post-destruction trim, and accounting corrections. These are localized and do not change tensor semantics or training numerics.

The core-hardening batch should then take L-01, L-02/S-07, S-01, and S-02 together with focused failure/stream validation. Checkpoint chunking and adaptive capacity are separate high-risk projects: they recover the most memory, but cross tensor serialization, optimizer state, Vulkan interop, and training resume behavior.

## Memory fix round

Execution date: 2026-07-11. Baseline commit: `f2734dfd3` (the analysis was written at `c08091557`; all fixes below were reconciled against the advanced branch). The implementation commits are listed at the end of this section. Build and runtime artifacts are under `output/memory_fix_round/` and are intentionally not versioned.

### Finding disposition

| ID | Disposition | Round result and evidence |
|---|---|---|
| L-01 | **Fixed** | Tensor destruction now unregisters retired lazy-IR ids, expired owners are pruned, the registry is capacity-bounded, and live/peak node and map counts are published. The 1,024-create/drop regression ends with zero live nodes/maps and bounded peaks. Commit `3774e759a`. |
| L-02 | **Fixed** | Pinned blocks retain `CudaHost` versus `MallocFallback` provenance through live and cached states; fallback blocks bypass the pinned cache and route to `free`. Fault injection verifies one fallback allocation and one matching free with no `cudaFreeHost` call. Commit `b80ffd61c`. |
| L-03 | **Fixed** | `TrainerManager::clearTrainer()` drops `splat_storage_` after the trainer is drained/reset, then trims the pool after tensor members have actually died. Scene teardown and edit-mode ownership regressions pass. Commit `ef257b156`. |
| L-04 | **Fixed** | Mesh ray-pick mirrors use stable `NodeId`, a weak owner, and geometry generation validation instead of a raw-pointer identity. A 256 MiB physical-upper-bound LRU and scene/destructor clears bound residency. Address-reuse and clear regressions pass. Commit `68351be87`. |
| L-05 | **Fixed** | The trainer validates scene data before allocating raw CUDA resources. Stream/event creation is checked and constructor failure unwinds transactionally. All constructor-failure regressions pass. Commit `7ad58cd1f`. |
| S-01 | **Fixed** | Every audited explicit-stream H2D source now uses pinned-aware `record_stream`; `set_stream` is no longer used as a lifetime substitute. The regression drops a pinned source immediately and proves guarded reuse remains unavailable until the non-default-stream event completes. Commit `1bf3c0df3`. |
| S-02 | **Fixed** | `reserve()` checks byte/element multiplication, leaves the old owner/state installed until allocation and copy succeed, and propagates allocation failure through the tensor assertion convention. Injected allocation failure and overflow both preserve pointer, shape, capacity, and contents. Commit `1bf3c0df3`. |
| S-03 | **Fixed for the conservative round scope** | The tuner now prices 8-bit color at 6 B/pixel per in-flight slot, 16-bit-to-float color at 18 B/pixel, and mask/depth at 4 B/pixel each, with checked arithmetic and a clamp that can reduce prefetch to one. Auxiliary configuration is applied before tuning. A byte semaphore/undistort-wide pipeline redesign remains architectural and was not approximated. Commit `5cd8c2b7c`. |
| S-04 | **Deferred** | A bounded checkpoint upload requires chunk-aware deserialization and direct canonical-to-swizzled placement into final storage. The current whole-tensor operator and final-publication semantics cannot be safely changed in a minimal patch; design below. |
| S-05 | **Fixed** | MRNF, MCMC, and gsplat hot/refine scratch use checked stream-ordered RAII allocations. Allocation failure throws through the existing fast-rasterizer iteration boundary before any kernel/CUB call sees null. Fault-injection regressions cover gsplat and MCMC. Commit `dd412a9b0`. |
| S-06 | **Fixed for the mandated policy** | Any single undo entry above the 512 MiB GPU budget is retained in history but immediately demoted to CPU, including the last entry. The broader aggregate hot-set and pageable archival redesign is deferred with W-08. Commit `65be5ec8e`. |
| S-07 | **Fixed** | The pinned cache has an environment-tunable hard byte budget (1 GiB default), global LRU eviction across size classes, phase-trim integration, and active/cached/peak telemetry. The 7k shutdown cache falls from 1,081.870014 MiB to 0.01 MiB. Commit `b80ffd61c`. |
| S-08 | **Deferred** | A bounded save requires chunked swizzle/download and optimizer serialization while preserving the existing byte format and corruption behavior. The whole-tensor serialization contract makes this inseparable from S-04; design below. |
| W-01 | **Deferred** | Adaptive physical commitment crosses exportable Vulkan storage and Adam capacity ownership. A correct solution needs growable shared backing and capacity migration tests; reducing the cap or overcommitting elsewhere would be a workaround. |
| W-02 | **Deferred** | Releasing optimizer/strategy/workspaces on `Finished` changes checkpoint-resume semantics. It needs an explicit resumable versus view-finalized state and render/checkpoint equivalence coverage. |
| W-03 | **Deferred** | Converting six loss workspaces to an active-mode union changes workspace identity and backward lifetime assumptions. It requires a mode/resolution matrix with loss and gradient parity. |
| W-04 | **Fixed** | Pool trimming now occurs after trainer reset/member destruction, rather than before late tensor deleters repopulate the cache. Commit `ef257b156`. |
| W-05 | **Fixed** | The bucket cache hard budget no longer exempts the last oversized block. A 2 MiB singleton under a 1 MiB injected budget is fully evicted. Commit `7116bf19d`. |
| W-06 | **Fixed** | `full_reset()` publishes zero high-water before VMM decommit. The forced VMM regression grows beyond 64 MiB and returns to exactly the initial 64 MiB; the 7k process returns from 144 MiB to the 128 MiB first chunk. Commit `831fc18b0`. |
| W-07 | **Fixed** | Resolution-keyed background tensors use a 256 MiB physical-bucket-aware LRU instead of an unbounded map. Commit `c540921cd`. |
| W-08 | **Deferred** | CPU `Tensor` currently means pinned storage throughout the API. Adding pageable archival storage requires allocator provenance, copy semantics, and explicit staging for later async H2D; silently substituting `malloc` at selected call sites would violate the transfer contract. |
| W-09 | **Deferred** | Releasing idle slabs needs per-slab live ownership and safe removal of every free-list entry before `cudaFree`. A phase-only raw free would risk dangling suballocations. |
| W-10 | **Fixed** | Process-global CUB/scalar scratch was replaced by scoped, labeled pool buffers; every operation performs a checked size query and allocation before execution. Commit `f5484b5d6`. |
| W-11 | **Deferred** | Making no-op `to()` shallow and replacing Float32-to-Bool host staging change shared tensor semantics. This needs a call-site compatibility audit and dedicated conversion kernels, not a memory-round shortcut. |
| W-12 | **Deferred** | Preprocessing is outside this round's disjoint source scope and is active in the parallel checkout. The no-resize ownership change should be handled as an isolated preprocess patch with output-hash coverage. |
| W-13 | **Deferred** | The legacy nvImageCodec singleton has deliberate process-lifetime ownership. Removal requires migrating all legacy `CacheLoader` callers to owner-counted pipeline instances and measuring the driver delta first. |
| W-14 | **Fixed** | Five loader outputs proven fully overwritten by their following kernels now use `empty()` instead of `zeros()`. Focused loader tensor-format and round-trip checks pass. Commit `6db9ddbc0`. |
| W-15 | **Fixed** | Non-VMM growth applies one multiplier to the actual required size. The forced fallback regression grows a 64 MiB arena to 256 MiB for a 128 MiB request, rather than 384 MiB. Commit `831fc18b0`. |
| M-01 | **Partially fixed; remainder deferred** | Pinned active/cached/peak bytes are separate, bucket/slab gauges are published, and VMM chunks are classified `Arena`. Profiler enable-after-allocation gaps and per-block exportable accounting remain; those need a wider diagnostics snapshot redesign. Commits `b80ffd61c`, `7116bf19d`, and `831fc18b0`. |

Supervisor-added critical audits also landed:

- Tensor and gsplat CUB wrappers reject null/nonzero workspaces after a checked size query; scalar reduction and every repository CUB site were audited (`f5484b5d6`, `dd412a9b0`).
- Gsplat intersection buffer growth is transactional: failed growth leaves null members and zero capacity, and checked output allocation propagates through the rasterizer boundary (`dd412a9b0`).
- Two loader stats/publication lock inversions found while establishing the baseline were fixed independently (`19ac4c280`, `009e81130`).
- Compute-sanitizer exposed a queued GPU tensor outliving the image-loader decode stream. Loader shutdown now retires queued tensors before stream destruction, with normal and memcheck regression coverage (`1e88bf8a8`).
- The live GUI gate exposed `_VulkanBuffer::allocSize` being overloaded as both backing-buffer size and region capacity. Backing size, view capacity, and active bytes are now separate, range validation is single-sourced, and shared scratch retains per-region growth bounds (`d44ca2c5a`).

### Checkpoint chunking design note (S-04/S-08)

The serialization format can remain unchanged, but the implementation must gain a bounded transfer primitive rather than wrapping the current whole-tensor operators:

1. Use one or two reusable 32-64 MiB pinned pages and checked byte-range reads/writes. Preserve every existing tensor header, dtype, shape, ordering, and failure message.
2. For ordinary tensors, upload/download directly between each page and its final destination range on one explicit stream, recording the page on that stream before reuse.
3. For shN, define a row-block canonical/swizzled mapping and transform each bounded Gaussian range directly between the page and final model storage. Never materialize full canonical CUDA or simultaneous canonical/swizzled CPU tensors.
4. Deserialize into unpublished model/optimizer owners; validate all byte counts, shapes, overflow, and truncation before committing the new scene/trainer state. On failure, destroy only the new transaction and preserve the old model.
5. Serialize Adam states through the same pages, then trim pinned/default-pool caches at the completed phase boundary.
6. Gate with SH0-SH3 old-reader/new-writer and old-writer/new-reader fixtures, truncated/corrupt inputs at every chunk boundary, semantic model equality, optimizer equality, and 7k resume parity. Peak RSS/VRAM must stay within final storage plus the fixed staging budget.

This crosses tensor serialization, swizzle layout, optimizer state, scene publication, and compatibility fixtures. Implementing only shN or adding an extra whole-buffer copy would hide the peak rather than remove its mechanism, so it is intentionally deferred.

### Before/after measurements

Both 7k runs used bicycle `images_4`, MCMC, `max_cap=1,500,000`, the same output/save points, and the worktree executable. A separate 11 GiB GPU process was present during the baseline, so timing is reported but not claimed as a code-speed recovery.

| Metric | Baseline | Final | Delta |
|---|---:|---:|---:|
| Result | 7,000 iterations; 1,293,710 splats | 7,000 iterations; 1,293,710 splats | Exact output count parity |
| Per-process peak GPU | 1,422 MiB | 1,422 MiB | 0 MiB |
| `/usr/bin/time` peak RSS | 1,741,512 KiB | 1,673,000 KiB | **-68,512 KiB (-70,156,288 bytes)** |
| 200 ms sampled peak RSS | 1,740,160 KiB | 1,616,748 KiB | **-123,412 KiB (-126,373,888 bytes)** |
| Pinned cache at shutdown | 1,081.870014 MiB / 960 blocks | 0.01 MiB | **about -1,081.860014 MiB (-1,134,412,446 bytes)** |
| Arena after trainer teardown | 144 MiB committed | 128 MiB committed | **-16 MiB (-16,777,216 bytes)** |
| Arena historical peak/reallocs | 141 MiB / 4 | 141 MiB / 4 | No regression |
| Step-7k pool/slab | 106 MiB used, 192 MiB reserved, 55.75 MiB slab | Same | No steady-state regression |
| Wall time | about 95.3 s | 32.92 s | Not comparable because baseline GPU contention differed |

Baseline artifacts: `output/memory_fix_round/baseline/`. Tier-A checkpoint: `output/memory_fix_round/tier_a/`. Final artifacts: `output/memory_fix_round/final_smoke/`.

### Verification record

- Release build completed for explicit `sm_89`; every build in the round used at most `-j6` after the machine-wide cap was issued.
- Required tensor/core filter: **271/271 passed**, including `TensorBasic*`, `TensorMasking*`, `TensorMove*`, `TensorReduction*`, `TensorIndexing*`, all `*AssertHardening*`, multistream, reserve failure/overflow, lazy registry, allocator policy, arena, and pinned telemetry regressions.
- New cross-area regression bundle: **7/7 passed** (bucket singleton budget; VMM reset; fallback arena growth; oversized undo CPU demotion; trainer clear/storage/pool; mesh clear; mesh address/NodeId reuse).
- Training failure/constructor/undo regressions: **7/7 passed**. Gsplat and MCMC injected allocation failures fail cleanly before kernel launch.
- Focused Python IO filter: **36 passed, 2 asset-dependent skips**. Real pipeline benchmark and tensor-format checks pass. The broader IO run still has the pre-round missing worktree assets, stale PLY message, RAD full-chunk contract tests, and UInt8-mask-versus-Float32-test expectation; none overlap the changed behavior.
- Vulkan focused baseline: **53/56 passed** after adding the buffer-view contract regression, with the same three pre-round debounce expectation failures (`dirty=16` versus `9`) before and after.
- Full Lazy IR suite: 52 pass; five pre-round tests still encode obsolete eager/deferred and invalid-index no-op expectations. The required long-create/drop and capacity fallback tests pass.
- Final 7k log contains no error-level, assertion, validation, CUDA-error, OOM, or crash line and exits 0.
- Standard 300-iteration compute-sanitizer memcheck: **exit 0, `ERROR SUMMARY: 0 errors`**, training completed successfully, 33.55 s wall time, 775,284 KiB peak RSS. Leak mode is not the gate because NVIDIA documents that CUDA VMM allocations are unsupported by leak checking; the required access/API memcheck used its default `--leak-check no`. Artifacts: `output/memory_fix_round/sanitizer_final/`.
- GUI PLY gate: the worktree binary loaded `splat_64400.ply` (3,000,000 SH3 Gaussians) and `3k.ply` (4,999,807 SH0 Gaussians), accepted scripted camera changes, returned nonempty viewport PNG captures, and shut down through the app API with exit 0. A separate normal-profile run autoloaded the user's densification plugin and also rendered/captured/exited 0. No reported run contains an error-level, assertion, Vulkan-validation, CUDA-error, or X11-error line. An earlier forced window-manager close was invalidated after it produced `BadWindow` and external `pycolmap` teardown noise; it is not counted as a gate result. Artifacts: `output/memory_fix_round/gui_final/{ply,default_profile}/`.
- Live-viewport training gate: with external plugin autoload disabled, the real GUI imported bicycle `images_4`, completed MCMC iteration 300 with 54,275 Gaussians, saved its checkpoint, rendered a nonempty capture from dataset camera `_DSC8679.JPG`, released the 384 MiB shared scratch arena, and exited 0 via the app API. There are zero error-level, assertion, validation, or CUDA-error lines. Artifacts: `output/memory_fix_round/gui_final/live_training/`.

### Commits

```text
19ac4c280 fix(io): avoid loader stats lock inversion
009e81130 fix(io): publish completed images outside pairing lock
1bf3c0df3 fix(tensor): make async copies and reserve failure safe
b80ffd61c fix(tensor): bound pinned cache and preserve allocation provenance
f5484b5d6 fix(tensor): make CUB workspace failures explicit
dd412a9b0 fix(training): make CUDA scratch allocation failure safe
7ad58cd1f fix(training): make trainer construction transactional
3774e759a fix(tensor): bound lazy IR registry lifetime
ef257b156 fix(training): release cleared training storage
68351be87 fix(viewer): bound mesh ray-pick cache
5cd8c2b7c fix(training): price image pipeline memory honestly
c540921cd perf(training): bound background image cache
65be5ec8e fix(viewer): demote oversized undo snapshots
7116bf19d perf(core): enforce bucket cache byte budget
831fc18b0 perf(core): correct arena reset and fallback growth
6db9ddbc0 perf(io): skip overwritten loader initialization
1e88bf8a8 fix(io): retire pipeline tensors before stream teardown
d44ca2c5a fix(vulkan): separate buffer backing and view capacity
```
