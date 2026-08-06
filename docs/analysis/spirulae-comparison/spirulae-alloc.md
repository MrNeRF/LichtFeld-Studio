I'll reverse-engineer the GPU memory allocation model from the focus files and follow every allocation/pool/scratch thread. Starting with the named entry points and allocation APIs.Core pool design is clear. Next I'll trace engine state, densify growth, data-manager staging, and train-step scratch sizes.# GPU Memory Allocation Model — Reverse Engineering Report

**Scope:** training engine path (CUDA default backend; Vulkan notes called out).  
**Method:** static analysis of allocation APIs and call sites only. No files were modified.

---

## Executive summary

The training path does **not** use CUDA’s stream-ordered allocator (`cudaMallocAsync`), does **not** use PyTorch’s caching allocator, and does **not** use a binning freelist. Almost all device memory goes through two process-global high-water-mark structures:

1. **`DevicePool`** — one grow-only allocation per named logical buffer (`PoolSlot` / sub-index).
2. **`DeviceScratch`** — a single grow-only blob for all CUB temporary storage.

That design makes **pool-level fragmentation impossible** (there is no free-list to fragment) and keeps **“cached” bytes equal to the historical high-water mark of each named buffer**, not a pool of recycled anonymous blocks. The maintainer’s claim is accurate for the *application-level* model; the CUDA *driver heap* can still fragment when a slot grows (free old + malloc larger) or when non-pool paths allocate.

Densification **never reallocates** splat tensors: capacity is fixed at `max_num_splats` up front; only `cur_num_splats` advances.

---

## 1. Allocation primitives and where they are used

| Primitive | Implementation | Where | Role |
|---|---|---|---|
| `backend::device_malloc` / `device_free` | CUDA: `cudaMalloc` / `cudaFree` | `src/backend/cuda/BackendRuntimeCuda.h:126–147` | Sole GPU alloc for engine pool/scratch |
| | Vulkan: dedicated `VkBuffer` + device memory per call | `src/backend/vulkan/VulkanRuntime.cpp:428–458` | Same API surface |
| `backend::device_malloc_checked` | throws friendly OOM | `src/backend/api/BackendRuntime.h:132–136` | Used by pool/scratch on grow |
| `backend::host_malloc_pinned` / `host_free_pinned` | CUDA: `cudaMallocHost` / `cudaFreeHost` | `BackendRuntimeCuda.h:149–153` | Pinned host |
| **`DevicePool`** | dense array of slots; grow-if-needed | `src/core/Tensor.h:139–329` | All named engine buffers |
| **`DeviceScratch`** | single pointer + cap | `src/core/Tensor.h:367–413` | CUB temp storage only |
| **`CUB_WRAPPER`** | probe size → `DeviceScratch::acquire` | `src/core/Common.cuh:182–190` | Explicitly avoids PyTorch caching allocator |
| **`AsyncReadout<T>`** | 2-slot pinned ring + events | `src/core/Tensor.h:69–136` | Non-blocking scalar D→H |
| **Direct `cudaMalloc`/`cudaFree`** | outside pool | `src/mesh/Meshing.cu`, `MeshingRaster.cu` | Meshing offline path only |
| **`cudaMallocHost` (local, freed after call)** | | `src/kernels/optim/AdamOptim.cu:173–197` | Fully host-side Adam chunks |
| **`RenderWorker::DevBuf`** | grow-only, not pool | `src/app/webviewer/RenderWorker.cpp:32–54` | Interactive viewer scratch |
| **Vulkan staging ring** | one host-visible buffer (default 64 MiB) | `src/backend/vulkan/VulkanRuntime.cpp:208–349` | Pageable H2D/D2H via ring |
| **`cudaMallocAsync` / `cudaFreeAsync`** | — | **not used** in `src/` | — |

Process VRAM accounting for `device_malloc` is a side table of sizes + atomic total (`BackendRuntimeCuda.h:17–36, 126–147`). Comment at `Tensor.h:22–23` notes this map is touched rarely because allocation is pooled upstream.

**Not present on the training path:** arenas with suballocation freelists, slab allocators, PyTorch `CachingAllocator`, or CUDA memory pools (`cudaMemPool`).

---

## 2. Slot / pool design

### 2.1 Metadata: one row per logical buffer

`src/core/PoolSlots.h` defines every pool buffer via `POOL_SLOT_TABLE` with:

- enumerator (`PoolSlot`)
- display name
- `VramCategory` (splat / splat×img / image / appearance / viewer / other)
- `SaveClass` (Never / Resume / Always)

Sub-indices allow two physical allocations under one logical slot (packed + bounds for quantized codecs):

```449:475:src/core/PoolSlots.h
// PoolKey = (PoolSlot << kSubBits) | sub
static constexpr uint32_t kSubBits = 4;
static constexpr uint32_t kSubMain   = 0;
static constexpr uint32_t kSubPacked = 1;
static constexpr uint32_t kSubBounds = 2;
// ...
static constexpr uint32_t kNumPoolKeys = (uint32_t)PoolSlot::Count << kSubBits;
```

### 2.2 Grow policy (exact capacity, not 2×)

```172:188:src/core/Tensor.h
// Grow-if-needed on a Slot already selected by the caller (mutex held).
template<typename T>
static T* _acquire_into(Slot& slot, size_t n) {
    size_t bytes = n * sizeof(T);
    if (bytes > slot.cap_bytes) {
        if (slot.ptr) backend::device_free(slot.ptr);
        slot.ptr = nullptr;
        slot.cap_bytes = 0;
        slot.ptr = backend::device_malloc_checked(bytes);
        slot.cap_bytes = bytes;
    }
    slot.used_bytes = bytes;
    ...
}
```

**Policy facts:**

| Property | Behavior | Evidence |
|---|---|---|
| Growth factor | **None** — new `cap_bytes = exact requested bytes` | `Tensor.h:176–183` |
| Shrink on resize | **Never** (capacity monotonic per slot) | `Tensor.h:139–141`, comment; `IntersectTile.cu:437` |
| Release path | `free(slot)` or `freeAll()` only | `Tensor.h:220–235` |
| Storage index | Fixed dense array of size `kNumPoolKeys` | `Tensor.h:145–168` |
| Dynamic keys | Side `unordered_map` for runtime names | `Tensor.h:163, 213–218` |

There is **no** “allocate 2× and amortize.” Growth cost is OOM-or-succeed exact size; slack after a shrink of *logical* size is left as unused capacity (`used_bytes ≤ cap_bytes`).

### 2.3 Views are non-owning

`DeviceVector` / `DeviceTensor2D/3D/5D` only store a pointer + shape; `resize()` calls `DevicePool::acquire` (`Tensor.h:460–466, 549–554, …`). Engine state comments: members are non-owning views (`EngineState.cpp:18–22`).

### 2.4 Densify / prune: in-place into fixed capacity

**Capacity** is set once at world upload:

```20:31:src/engine/EngineSetup.cpp
int64_t max_num_splats = std::get<2>(means)[0];
...
engine().cur_num_splats = num_splats;
engine().max_num_splats = max_num_splats;
```

Seeding (CLI path) preallocates to `cap_max` when MCMC + flag:

```151:152:src/app/TrainerCore.cpp
const int64_t cap = cfg.preallocate_splat_tensors && cfg.use_mcmc
    ? std::max<int64_t>(num, cfg.cap_max) : num;
```

Optimizer state, radii, accum, grads are all sized with **`N = max_num_splats`**, not current count (`EngineOptim.cpp:65–153`, `EngineLoss.cpp:20–21`).

Densify step:

```274:276:src/engine/EngineDensify.cpp
int64_t n_target = std::min(max_num_splats, (int64_t)(cfg.growth_factor * cur_num_splats));
num_added = (int)std::max((int64_t)0, n_target - cur_num_splats);
```

```347:348:src/engine/EngineDensify.cpp
engine().cur_num_splats = cur_num_splats + num_added;
return num_added;
```

- New splat rows are **written into existing device buffers** (kernels receive the same `DeviceVector`s sized for `max_N`).
- **No** `resize`/`device_malloc` of world or Adam state during densify.
- Growth is gated by `growth_factor` (default `1.0f` in `EngineConfig.h:181`) and hard cap `max_num_splats`.
- “Prune/relocate” mutates in place (opacity/means/etc.); does not free GPU memory.

**Headroom:** `max_num_splats - cur_num_splats` unused rows in every per-splat buffer. That is intentional preallocation, not a free-list.

When quant layouts flip, code **frees** unused fp32 slots (`EngineOptim.cpp:78–109`) and may allocate the alternate quant packing — those are mode switches, not densify.

### 2.5 DeviceScratch

```390:399:src/core/Tensor.h
void* acquire(size_t bytes) {
    if (bytes > _cap) {
        if (_ptr) backend::device_free(_ptr);
        _ptr = backend::device_malloc_checked(bytes, "scratch buffer");
        _cap = bytes;
    }
    return _ptr;
}
```

Single global buffer; all CUB scans/sorts share it serially via `CUB_WRAPPER`. Comment: “monotonically growing, never fragmented” (`Common.cuh:182–183`).

---

## 3. Lifetime model

### 3.1 Engine as process singleton

`engine()` is a function-local static (`EngineState.cpp:12–15`). `engine_reset()` replaces state and calls `freeAllDeviceMemory()` (`EngineState.cpp:17–24`).

### 3.2 Persistent (cross-step) — `SaveClass::Resume` / `Always`

| Group | Examples | Sized by | Notes |
|---|---|---|---|
| World params | `world.means/quats/scales/opacities/features_*` | `max_N` (and `K` SH bands) | Init once (`EngineSetup.cpp:36–46`) |
| Adam moments | `eng.g1_*`, `eng.g2_*` or quant `*_qfpbo` / `sh_quant*` | `max_N` | `_ensure_optim_state` |
| Densify aux | `eng.radii`, `eng.accum_buffer`, bias steps | `max_N` | Zeroed on densify steps as needed |
| Appearance | bilagrid grids, PPISP params, bg SH, color matrices | cameras × grid shape | Init APIs |
| Quant value SH | `eng.world.sh_vq{8,16}{,_fpbo}` | `max_N * K * 3` cells | Optional |

### 3.3 Per-step / per-forward scratch — `SaveClass::Never`

Re-acquired every forward/backward with the **same** `PoolSlot` keys → reuse capacity after first peak:

- Projection: `proj.*` (`ProjectionFwd.cu:64–69`, `ProjectionPackedFwd.cu:89–130`)
- Tile intersect: `isect.*` (`IntersectTile.cu:308–357`) — sizes track `n_isects` high-water
- Raster outputs: `render.*`, `renders.*`, `distortions.*`
- Gradients: `eng.v_*` (and quant variants) — zeroed each bwd unless sub-batch accumulate
- GT staging: `gt.*` re-uploaded each step
- Camera table: re-copied each frame (`EngineState.h:112`)
- Bilagrid post images / image grads: batch-sized

**ForwardCache** holds non-owning views into these pool buffers for the duration of the step (`EngineState.h:124–150`).

### 3.4 How scratch is recycled

1. Caller `resize(slot, n)` → if `n * sizeof(T) ≤ cap_bytes`, return same `ptr` (no free/malloc).
2. Logical shrink only updates `used_bytes` (post-filter isects: `IntersectTile.cu:437–441`).
3. CUB temps: always the one `DeviceScratch` blob.
4. Dynamic keys (`ppl.s{n}.*`, `renders.rgb`, `proj.screen.0`, …): same high-water map under string keys (`Primitive.cuh:436–456`, `PoolSlots.h:39–43`).

There is **no** “release to freelist for other buffers to steal.” Unused capacity in slot A never becomes capacity for slot B.

### 3.5 Sub-batch lifetime nuance

`skip_grad_zero` keeps gradient and radii accumulation across sub-batches (`EngineState.h:250–262`, `EngineForward.cpp:47–60`, `EngineLoss.cpp:116–127`). Still the same pool buffers; no extra allocations.

---

## 4. Why fragmentation “cannot occur” — and where it still can

### 4.1 What the design actually guarantees

| Claim | Mechanism |
|---|---|
| No freelist fragmentation | Pool never returns memory to a binning allocator; each key owns one block |
| No thrashing of many small anonymous temps | Named slots + one CUB scratch |
| No densify realloc churn | Fixed `max_N` capacity; densify is in-place |
| Stable addresses across steps (after warm-up) | Grow only when size exceeds prior cap |
| Avoid PyTorch caching allocator traps | Direct `cudaMalloc` + own pool; CUB via `DeviceScratch` |

These are **application-level** guarantees: the *pool* cannot fragment itself.

### 4.2 Where fragmentation / churn can still occur (driver / edges)

| Location | Mechanism | Severity for training |
|---|---|---|
| Slot growth | `device_free` old + `device_malloc` larger exact size | Once per high-water increase per slot |
| Mode switch frees | `DevicePool::free(WorldFeaturesSh)` etc. (`EngineOptim.cpp:78–109`) | Rare config flips |
| `engine_reset` / exit | `freeAll` then re-acquire | Between runs |
| Many simultaneous live slots | External fragmentation of CUDA heap if grow events interleave | Possible under many growing image/isect buffers |
| Meshing | Unpooled `cudaMalloc`/`cudaFree` | Offline; not train step |
| Viewer `DevBuf` | Independent grow-only allocs | Viewer only |
| Vulkan | Each `device_malloc` = dedicated device memory object | Grow = destroy/create Vk memory |
| `DeviceScratch` grow | Same free+malloc as pool slots | Once per new CUB size peak |
| CUDA driver internal | Even sequential free+malloc can fragment device VA | Speculative / driver-dependent |

**Honest restatement of maintainer claim:**  
“Never fragments” = *the caching layer itself does not fragment* (no freelist bins, no partial reuse of differently sized blocks). It is **not** a proof that `cudaMalloc` external fragmentation is impossible forever.

### 4.3 “Shrink without realloc” is explicit

```437:441:src/kernels/tile/IntersectTile.cu
/* Read back actual count and shrink logical sizes (no realloc — pool only grows) */
...
isect_ids_out.resize(PoolSlot::IsectPostIds, n_isects);
```

---

## 5. What “cache at minimum” means concretely

### 5.1 There *is* a caching allocator — but minimal

| Layer | Caches what? | Waste policy |
|---|---|---|
| `DevicePool` per key | Last high-water device block | Exact-size grow; never shrinks |
| `DeviceScratch` | One high-water CUB buffer | Exact-size grow |
| PyTorch CachingAllocator | **Not used** on compute path | N/A |
| CUDA caching allocator (async pool) | **Not used** | N/A |
| Vulkan staging | Fixed ring (default 64 MiB) | Constant, not grow-per-call |

“Minimum” means:

1. **No cross-key free list** holding anonymous freed blocks.
2. **No geometric over-allocation** (no `max(2*old, new)`).
3. **Cached bytes ≈ Σ high-water marks of live keys + scratch cap**, reported as `cap_bytes` vs `used_bytes` (`Tensor.h:237–244`, `getBreakdown`).
4. After warm-up (largest image, largest `n_isects`, full `max_N`), **steady-state steps issue ~0 new `cudaMalloc`s**.

### 5.2 Peak vs steady VRAM

| Phase | Behavior |
|---|---|
| Init | World + optim at `max_N`; appearance tables |
| Early steps | Projection/isect/raster slots grow to first large batch |
| Densify | **No extra VRAM** from realloc; only more *logical* active rows within capacity |
| Peak | Max over steps of (Σ slot caps + scratch + non-pool) |
| Steady | Caps plateau; `used_bytes` may be below `cap_bytes` for variable `nnz` / `n_isects` |

Reporting hooks: `engine_get_pool_breakdown[_categorized]`, `engine_get_scratch_bytes` (`EngineQuery.cpp:180–201`). Process total: `backend::memory_usage().process_bytes`.

### 5.3 Intentional “waste” that is not freelist cache

- Preallocated inactive splat rows: `(max_N - cur_N) × per-splat footprint`.
- Double buffers for radix sort: `isect.ids_a/b`, `isect.flat_a/b` both held at `n_isects` high-water.
- Packed path keeps `C*N` mask/scan even when `nnz ≪ C*N` (mask still full size).
- Quant storage typed pessimistically at 8-bit footprint for SH optim quant holders (`EngineState.h:200–202`).

---

## 6. CPU-side memory tricks

| Mechanism | Detail | Location |
|---|---|---|
| **Pinned ring D→H** | `AsyncReadout`: 2× pinned slots, async memcpy + event; next-iter sync is cheap | `Tensor.h:69–125`; used in `PerPixelLoss.cu`, `EngineTrainStep.cpp` |
| **GT staging on GPU** | Host uint8/uint16 → pool `gt.staging_u8/u16` → convert kernel → `gt.rgb` float | `EngineCommon.h:192–228` |
| **Zero-copy if already device** | `_hv_to_dv` skips H2D when `is_device_pointer` | `EngineCommon.h:91–101` |
| **DataManager CPU cache** | Full decoded dataset in pageable host RAM | `DataManager.h:12–17`, `_rgb_cache` etc. |
| **DataManager DISK prefetch** | Bounded ready queue `prefetch_batches` (default 4) | `DataManager.h:82–84` |
| **Batch host buffers** | `std::vector` per modality; views live until next `next_train_batch` | `EngineDataManager.cpp:10–13` |
| **Semi-offloaded Adam** | Moments on host; 64 MiB device chunk buffer from pool | `AdamOptim.cu:201–221` |
| **Fully offloaded Adam** | Temp double-buffered pinned chunks, free at end of call | `AdamOptim.cu:169–197` |
| **Vulkan staging ring** | Host-visible 64 MiB (env `SSPLAT_VK_STAGING_MB`) for pageable transfers | `VulkanRuntime.cpp:220–227` |
| **Bilagrid axes** | Device tables for warp; managed by DataManager | `DataManager.h:185–187` |

Training GT path is typically: **pageable host decode → H2D into pool staging/GT → train**. Not pinned end-to-end for full images.

---

## 7. Buffer inventory table

Notation:  
`N = max_num_splats`, `n = cur_num_splats` (kernel work often uses `n`, storage often `N`),  
`K = num_sh` (SH residual bands in buffer),  
`C = batch cameras` (post-split),  
`H,W = image size`,  
`nnz = visible (cam,splat) pairs` (packed),  
`n_isects = tile×splat intersections`,  
`tile_w = ⌈W / TILE_IX⌉`, `tile_h = ⌈H / TILE_IY⌉`,  
`B = block 256`,  
`cells_sh = N·K·3`.

Element sizes: `f=4`, `f2=8`, `f3=12`, `f4=16`, `i32=4`, `i64=8`, `u8=1`, `bool=1`.

### 7.1 Persistent splat / optim (DevicePool, Resume)

| Buffer name | Size formula (bytes, typical) | Lifetime | Mechanism |
|---|---|---|---|
| `world.means` | `N · 12` | Persistent | Pool `WorldMeans` |
| `world.quats` | `N · 16` | Persistent | Pool |
| `world.scales` | `N · 12` | Persistent | Pool |
| `world.opacities` | `N · 4` | Persistent | Pool |
| `world.features_dc` | `N · 12` | Persistent | Pool |
| `world.features_sh` | `N · K · 12` if SH value bits=32; else shape-only, 0 pool bytes | Persistent | Pool or freed (`EngineOptim.cpp:77–81`) |
| `eng.world.sh_vq8{,_fpbo}.q/.qb` | packed: `cells_sh · 1` (8-bit) or ·2 (16-bit); bounds: `ceil(cells/B)·8` or `ceil(N/B)·8` | Persistent | Pool quant sub-slots |
| `eng.g1_*` / `eng.g2_*` (non-SH attrs) | same as world attrs | Persistent | Pool; freed if non-SH optim quant |
| `eng.g1_features_sh` / `g2_features_sh` | `N·K·12` each if SH optim bits=32 | Persistent | Pool |
| `eng.sh_quant{,_fpbo}.q/.qb` | packed ≈ `cells_sh · 2` (8-bit AoS); bounds layout cell- or splat-block | Persistent | Pool |
| `eng.*_qfpbo` (means/quats/…) | packed 16-bit joint Adam; `N·prims·4` + `ceil(N/B)·16` bounds | Persistent | Pool |
| `eng.radii` | `N · 4` | Persistent (zeroed/accum per step) | Pool |
| `eng.accum_buffer` | `N · 8` (`float2`) | Persistent densify score | Pool |
| `eng.bias_correction_steps` | `N · 4` if enabled | Persistent | Pool |
| `eng.densify.world_grad_score` | `N · 4` if score blend > 0 | Per-step write, Never save | Pool |

Densify **does not** change these allocations; it only fills rows `[0, cur_N)`.

### 7.2 Per-step gradients (Never)

| Buffer | Formula | Lifetime | Mechanism |
|---|---|---|---|
| `eng.v_means` etc. | `N · {12,16,12,4,12}` + `N·K·12` SH | Per bwd; zeroed (or accum) | Pool @ `max_N` |
| `eng.v_*_q` | cells `N·prims` + bounds `ceil(N/256)·float2` | Per bwd if grad quant | Pool |
| FPBO path | many `v_*` **not allocated** for 3dgs/mip | Saves VRAM | `EngineLoss.cpp:32–49` |

### 7.3 Projection / screen (Splat×Img, Never)

| Buffer | Formula | Lifetime | Mechanism |
|---|---|---|---|
| `proj.aabb` | non-packed: `C·N·16`; packed: `nnz·16` | Forward → bwd | Pool |
| `proj.depths` | `C·N·4` or `nnz·4` | same | Pool |
| `proj.mask` / `proj.scan` | `C·N · 1` / `C·N · 8` (packed only) | Forward | Pool |
| `proj.camera_ids` / `gaussian_ids` | `nnz · 4` | packed | Pool |
| `proj.screen.{i}` | dynamic; size `C·N` or `nnz` × channel width (xy/depth/conic/opac/rgb…) | Forward cache | `acquire_dynamic` |
| `fused_proj_bwd.*` | `nnz` or `N+1` | Bwd / FPBO | Pool |

### 7.4 Tile intersect (Splat×Img, Never)

| Buffer | Formula | Lifetime | Mechanism |
|---|---|---|---|
| `isect.tiles_per_splat` | `total_count · 8` (`total_count = C·N` or `nnz`) | Forward | Pool |
| `isect.cum_tiles` | same | Forward + CUB scratch | Pool + DeviceScratch |
| `isect.offsets` | `C · tile_h · tile_w · 4` | Forward | Pool |
| `isect.ids_a/b` | `n_isects · 8` each | Sort double buffer | Pool |
| `isect.flat_a/b` | `n_isects · 4` each | same | Pool |
| `isect_post.*` | ≤ prior `n_isects` | Optional filter | Pool HWM |

### 7.5 Raster / image (Image, Never)

| Buffer | Formula | Lifetime | Mechanism |
|---|---|---|---|
| `renders.rgb` | `C·H·W·12` | Forward → loss → bwd | dynamic under `Renders` |
| `renders.depth` | `C·H·W·4` if RGB_D | same | dynamic |
| `distortions.*` | subset of channels · `C·H·W` | if dist weight ≠ 0 | dynamic |
| `render.Ts` | `C·H·W·4` | bwd needs transmittance | Pool |
| `render.last_ids` | `C·H·W·4` | bwd | Pool |
| `render.median` | `C·H·W·4` if requested | optional | Pool |
| `raster_bwd.accum_weight` | `N · 4` | densify score path | Pool |
| `raster_bwd.v_viewmats` / `v_screen` / `v_world` | bwd intermediates | step | Pool |
| `eng.v_rgb`, `eng.v_depth`, … | image grads `C·H·W·…` | loss bwd | Pool |
| `gt.rgb` (float) | `C·H·W·12` | step | Pool |
| `gt.staging_u8/u16` | raw GT numel | step upload | Pool |
| `gt.depth/normal/alpha` | modality shapes | step | Pool |
| `ppl.*` | loss maps; multiscale dynamic keys | step | Pool + dynamic |

### 7.6 Appearance (persistent + step scratch)

| Buffer | Formula | Lifetime | Mechanism |
|---|---|---|---|
| `eng.bg.{rgb,depth,normal}.grids` | `n_grids · L · H · W · C` floats (or quant) | Persistent Always | Pool |
| Adam/AdaGrad for grids | same volume or quant packed | Persistent Resume | Pool |
| `eng.bg.*.post` / `image_grad` | batch image / grid grads | Never | Pool |
| `eng.ppisp.params` | `n_grids · P` | Always | Pool |
| `eng.bg_sky.sh_coeffs` | `(deg+1)² · 12` | Always | Pool |
| color space matrices | `3×3` | Always | Pool |

### 7.7 Global scratch & non-pool

| Buffer | Formula | Lifetime | Mechanism |
|---|---|---|---|
| **DeviceScratch** | max CUB temp over all ops (sort/scan/select) | Process HWM | `DeviceScratch::global` |
| `semi_offloaded_adam_buf` | `2 · (64 MiB / 8)` floats ≈ 16 M floats | While that optim path runs | Pool |
| `AsyncReadout` host | `2 · n_elems · 4` pinned | Process static | `cudaMallocHost` |
| Viewer BVH / thumbs | `N_post · S² · 4` etc. | Viewer session | Pool Viewer category |
| Meshing temps | various | Offline | raw `cudaMalloc` |
| Vulkan staging | default 64 MiB | Backend lifetime | dedicated host-visible |

### 7.8 CUB path

```182:190:src/core/Common.cuh
// Routes CUB temporary storage through DeviceScratch (monotonically growing,
// never fragmented) instead of the PyTorch caching allocator.
#define CUB_WRAPPER(func, ...) \
    func(nullptr, _cub_temp_bytes, __VA_ARGS__); \
    func(DeviceScratch::global().acquire(_cub_temp_bytes), _cub_temp_bytes, __VA_ARGS__);
```

Used heavily in tile sort (`IntersectTile.cu:332–385`), packed projection scan, densify sorts, etc.

---

## 8. Train-step flow (memory perspective)

Simplified path through focus files:

1. **DataManager** materializes host batch (`EngineDataManager.cpp`).
2. **Camera + GT** uploaded into pool slots (`EngineSetup.cpp` / warped setup).
3. **Forward** (`EngineForward.cpp`): radii @ `max_N`; projection acquires `proj.*`; intersect acquires `isect.*` (grows with `n_isects`); raster acquires `renders`/`render.Ts`/….
4. **Loss/bwd** (`EngineLoss.cpp`): grads @ `max_N` (or quant / FPBO skip); image grads; bilagrid/PPISP hooks.
5. **Optim** (`EngineOptim.cpp`): updates in place in world/Adam buffers; optional world_grad_score.
6. **Densify** (`EngineDensify.cpp`): in-place relocate/add; `cur_num_splats += num_added`; no realloc.
7. Pool capacities remain; next step reuses.

---

## 9. Design intent vs. optimization implications

**What the design optimizes for**

- Predictable VRAM after warm-up.
- Zero densify-time allocation failures (if init fit, densify fits).
- Simple ownership (one owner: `DevicePool` + `DeviceScratch`).
- Checkpoint metadata completeness (`PoolSlots` X-macro).

**Tradeoffs relevant to an optimization plan**

| Tradeoff | Effect |
|---|---|
| Exact-size grow | No internal over-provision waste, but **more frequent reallocs** if sizes climb gradually (each new max frees+allocs) |
| Never shrink | High-water of worst camera / worst isect count holds forever until `engine_reset` |
| Fixed `max_N` | **Peak ≈ full-cap** even early training when `cur_N ≪ max_N` |
| Double-buffered isect sort | ~2× keys+values at peak isects |
| Packed mask/scan at `C·N` | Large even when nnz sparse |
| Dynamic string map | Extra map for screen/render channel keys; same HWM semantics |
| Meshing / viewer bypass pool | Separate fragmentation story |

**Suggested optimization angles (analysis only, not implemented)**

1. Growth factor or power-of-two caps for highly variable `n_isects` / image sizes to cut free+malloc churn.
2. Optional compact / re-pack after densify when not preallocating to `cap_max`.
3. Shared arena for Splat×Img Never buffers that are never live simultaneously (hard: forward holds many concurrently).
4. Cap isect double-buffers if sort can be single-buffer + temp (trade scratch).
5. Quant / FPBO already major VRAM reducers — inventory already branches layouts carefully.

---

## 10. Key source anchors (quick index)

| Topic | File:lines |
|---|---|
| Pool grow semantics | `src/core/Tensor.h:139–218` |
| Scratch + freeAll | `src/core/Tensor.h:367–421` |
| Slot catalog | `src/core/PoolSlots.h:87–369` |
| CUB routing | `src/core/Common.cuh:182–190` |
| CUDA malloc | `src/backend/cuda/BackendRuntimeCuda.h:126–153` |
| Runtime contract | `src/backend/api/BackendRuntime.h:88–96` |
| World max_N | `src/engine/EngineSetup.cpp:20–46` |
| Optim @ max_N | `src/engine/EngineOptim.cpp:65–153` |
| Grad @ max_N | `src/engine/EngineLoss.cpp:20–128` |
| Densify no realloc | `src/engine/EngineDensify.cpp:274–348` |
| Preallocate cap | `src/app/TrainerCore.cpp:151–152` |
| Reset frees all | `src/engine/EngineState.cpp:17–24` |
| Isect sizes / HWM | `src/kernels/tile/IntersectTile.cu:297–441` |
| Projection sizes | `src/kernels/projection/ProjectionFwd.cu:63–69`, `ProjectionPackedFwd.cu:89–130` |
| Render sizes | `src/primitives/Primitive.cuh:61–91`, `Rasterization*.cu` |
| Screen dynamic keys | `src/primitives/Primitive.cuh:436–456` |
| GT staging | `src/engine/EngineCommon.h:172–228` |
| DataManager host cache | `src/data/DataManager.h:10–28` |
| Vulkan staging | `src/backend/vulkan/VulkanRuntime.cpp:208–349, 428–458` |

---

**Bottom line:** Training GPU memory is a **named, high-water-mark, exact-size, never-shrink pool** plus a **single CUB scratch buffer**, with densify operating **in-place inside a fixed `max_num_splats` capacity**. That is how allocation “never fragments” *inside the app’s caching model* and how the cache stays “at minimum” *relative to freelist/caching allocators* — while peak VRAM still tracks the historical maximum of every live named buffer, and the driver heap remains free to fragment on rare grow/free-all events.
