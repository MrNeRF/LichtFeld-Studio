# Epic #1496 — Sweep C (latent bug patterns)

**Branch:** `epic-1496-computed-barriers`  
**Date:** 2026-08-06  
**Scope:** `src/` + `tests/`, pattern-driven structural search (ast-grep + targeted ripgrep/scripts)  
**Method:** Read-only. Inspired by campaign findings (`Buffer`/`_VulkanBuffer` handle aliasing; cumsum `require_backing`; push-descriptor size contracts).  
**Quality rule:** Verified findings only — no speculative laundry lists.

---

## 1. Handle-identity traps

### C1.1 — `_VulkanBuffer` / `Buffer<T>` shallow copy + external destroy (epic baseline, still live)

| Field | Value |
|---|---|
| **Location** | `src/rendering/rasterizer/vulkan/src/buffer.h:23–119` (copy/assign); destroy at `buffer.cpp:302–344` (`VulkanGSPipeline::destroyBuffer`) |
| **Pattern** | Handle bag with **manual shallow copy** of `VkBuffer` + `VmaAllocation` + size fields; ownership released only via free function / pipeline API, not RAII |
| **Evidence** | `_VulkanBuffer` defines copy ctor + `operator=` that assign `buffer`/`allocation` without nulling the source. `Buffer<T>` copies `deviceBuffer` the same way. `destroyBuffer` calls `vmaDestroyBuffer` when `allocation != VK_NULL_HANDLE`. Two live `Buffer`/`_VulkanBuffer` values that share a non-null `allocation` → double-destroy or use-after-free on the survivor. Epic inv notes (`debug/epic1496/inv_cumsum_report.md` §4) already flag this; structural re-scan confirms **no** deleted copy, **no** move-only ownership. |
| **Severity** | **High** (latent design footgun; production mostly passes by reference, but any `Buffer` value copy or `deviceBuffer = other.deviceBuffer` of an **owned** buffer is undefined) |
| **Minimal action** | Delete copy ctor/`operator=` on `_VulkanBuffer` and `Buffer<T>`; add explicit `view()` factory for non-owning copies (`allocation == null`); optionally `assert` in `destroyBuffer` that no second live owner shares `allocation`. |

### C1.2 — `ScopedStagingBuffer` RAII destroy + default copy (other than Buffer)

| Field | Value |
|---|---|
| **Location** | `src/visualizer/rendering/point_cloud_vulkan_renderer.cpp:162–181` (dtor); use at `:1987–2005` |
| **Pattern** | `{VkBuffer, VmaAllocation}` + destructor that `vmaDestroyBuffer`s; **no** deleted copy/move |
| **Evidence** | Structural scan of `src/` for structs with GPU handle fields and a destructor that destroys them: **only** `ScopedStagingBuffer` lacks deleted copy. Default-generated copy/move would duplicate handles; both dtors free the same allocation. Current call site is stack-only (`ScopedStagingBuffer staging{}`) — not currently copied — but the type is still the classic double-destroy trap. |
| **Severity** | **Medium** (latent; unsafe under any future copy/return/emplace) |
| **Minimal action** | `= delete` copy; implement move that nulls source handles (or use `unique_ptr`/existing move-only CUDA-style guards). |

### C1.3 — `ManagedBuffer` POD ownership bag (sibling class, free-function destroy)

| Field | Value |
|---|---|
| **Location** | `src/visualizer/rendering/point_cloud_vulkan_renderer.cpp:97–114` (`ManagedBuffer` + `destroyBuffer`); stored in `std::vector<ManagedBuffer> pending_stagings` (`:445`, push at `:630`) |
| **Pattern** | `{VkBuffer, VmaAllocation}` + **default copy** + free-function destroy; no destructor |
| **Evidence** | `destroyBuffer` always `vmaDestroyBuffer`s a non-null `buffer`. Type is trivial: `push_back(std::move(staging))` is a **handle copy** (source not zeroed). Safe today only because the local `staging` is not destroyed after the push and vector reallocation does not call a freeing dtor. Same ownership model as `_VulkanBuffer` without even an explicit copy warning. |
| **Severity** | **Medium** (latent; footgun under any second `destroyBuffer` on a copied value) |
| **Minimal action** | Delete copy; move-only with null-out; or store `unique_ptr`/index into a pool. |

**Not reported as findings (checked, safe or not ownership):**  
`CudaDeviceMemory` (deleted copy + move), mesh2splat `ShaderModule`/`ConversionPipelineResources` (deleted copy), `_Stager` (mutex → non-copyable). Many other POD handle bags exist (`buffer_data_t`, `ExternalBuffer`, `GpuMesh`, …) with free-function destroy; they share the class but are not listed unless a concrete copy+double-destroy path is visible (quality filter).

---

## 2. Size / capacity contracts (mixed sources)

### C2.1 — Visible-chain cumsum: resize/output from `visible_capacity`, input unbound by local contract

| Field | Value |
|---|---|
| **Location** | `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp:2970–3020` (`executeCalculateIndexBufferOffsetVisible`); contrast classic trap at `:1798–1817` (`executeCumsum` `require_backing`) |
| **Pattern** | Dispatch/descriptor size derived from one source; bound range comes from another without a host-side backing check |
| **Evidence** | Classic `executeCumsum` computes `num_elements = input_buffer.deviceSize()`, resizes output, then **`require_backing` on input/output/block_sums** against that element count. Visible path: `output`/`_cumsum_blockSums*` resized from **`visible_capacity`**, but `input = buffers.tiles_touched_depth_ordered.deviceBuffer` is taken as-is with **no** `require_backing` that `input.size/allocSize >= visible_capacity * sizeof(int32_t)`. Element count for the scan is GPU-side (`g_counts[level]` from `cumsum_counts`). Shader still indexes `g_input[gid]` for `gid < numElements`. Call graph today usually resizes input in `executeMacroCoverage` (`:2685`), but that is **caller convention**, not a local contract — the same class of silent OOB the epic hardened for classic cumsum. |
| **Severity** | **Medium** (missing contract on a public multi-phase scan path that already bit one epic investigation) |
| **Minimal action** | Reuse/adapt `require_backing` for `input`, `output`, `block_sums`, `block_sums2`, and `counts` (4 words) against the host capacities implied by `visible_capacity` before any phase dispatch. |

### C2.2 — `prepare_visible_sort`: index from push-constant `num_splats`, range from cumsum-sized buffer

| Field | Value |
|---|---|
| **Location** | Host: `gs_renderer.cpp:2266–2284`; shader: `shader/src/slang/prepare_visible_sort.slang:19–20` |
| **Pattern** | Storage access `visible_prefix[uniforms.num_splats - 1]` while descriptor range is `visible_prefix.deviceBuffer.size` from a **prior** `executeCumsum` that sized by `input.deviceSize()`, not by re-checking `uniforms.num_splats` |
| **Evidence** | Host passes `prepare_uniforms.num_splats = uniforms.num_splats` and binds `buffers.visible_prefix.deviceBuffer` without asserting `visible_prefix.deviceSize() >= num_splats`. Coupling today: `resizeDeviceBuffer(visible_flags, num_splats)` then cumsum → same element count. Any future soft-view rebind, soft-clear, or path that leaves `visible_prefix` smaller than `uniforms.num_splats` yields OOB vs the bound range (06936-class). |
| **Severity** | **Medium** (latent; same dual-source contract as cumsum require_backing) |
| **Minimal action** | Before dispatch: `throw_renderer_contract` if `visible_prefix.deviceSize() < num_splats` (and dispatch-args buffer large enough). |

### C2.3 — `prepare_tile_sort` (classic): same dual-source index

| Field | Value |
|---|---|
| **Location** | Host: `gs_renderer.cpp:1954–1985`; shader: `prepare_tile_sort.slang:31–32` (`index_buffer_offset[uniforms.num_splats - 1]`) |
| **Pattern** | Uniform-driven tail index into storage whose size came from cumsum(`tiles_touched_depth_ordered.deviceSize()`), not from a local `num_splats` check |
| **Evidence** | `executeCalculateIndexBufferOffset` early-outs on `uniforms.num_splats == 0`, then `executeCumsum(tiles_touched_depth_ordered → index_buffer_offset)` using **input.deviceSize()**. `executePrepareTileSort` then indexes with **`uniforms.num_splats`**. No host assert that `index_buffer_offset.deviceSize() >= num_splats`. Visible-bounded variant passes `visible_limit` into the same uniform field (`:3095–3105`) and is better coupled to `visible_capacity`, but classic path is dual-sourced. |
| **Severity** | **Medium** |
| **Minimal action** | Assert `index_buffer_offset.deviceSize() >= uniforms.num_splats` (classic) / `>= visible_capacity` (visible) before the single-thread dispatch. |

---

## 3. Guarded-write patterns in shaders (top candidates only)

Uniform (or uniform-driven) bounds that are **not** the bound buffer’s length API. No deep correctness proof — candidates for the same class as host dual-source contracts.

| # | File:line | Write / access | Guard source | Why it ranks |
|---|---|---|---|---|
| S1 | `prepare_tile_sort.slang:25–32` | `index_buffer_offset[tail-1]` / `[num_splats-1]` | `uniforms.num_splats` (+ optional `visible_count[0]`) | Host push constant, not `length()` of binding 0 |
| S2 | `prepare_visible_sort.slang:19–20` | `visible_prefix[num_splats-1]` | `uniforms.num_splats` | Same; single-thread tail read |
| S3 | `compact_visible_primitives.slang:33–35` | `out_compact_*[compact_idx]` | Dispatch/`num_splats` on inputs; **write index** is `visible_prefix[i]-1` with **no** clamp to out capacity | Out range is host sort capacity; compact_idx is data-dependent |
| S4 | `lod_select_threshold.slang:218–228` | `out_indices/logical/weights/levels[append_index]` | `uniforms.output_capacity` after atomic | Correct capacity guard pattern — listed as control contrast |
| S5 | `lod_compact_touch.slang:37–49` | `out_protected` / `out_misses` | `protected_capacity` / `miss_capacity` | Same — capacity from uniforms, host must match resizes (`kLodCompact*Cap`) |

**Not deep-dived:** tile/raster shaders (tile ranges, pixel dims from full uniforms + geometry).

---

## 4. Push-constant / PACK_STRUCT sizeof mismatches

### C4.1 — `PACK_STRUCT` without nearby `static_assert(sizeof…)`

| Struct | Location | sizeof (packed) | Nearby assert? | Used as executeCompute uniforms |
|---|---|---|---|---|
| `VulkanGSLodCompactUniforms` | `gs_renderer.h:55–60` | 16 B (4×`uint32_t`) | **Missing** (neighbors at L53 / L92 assert others) | Yes — `gs_renderer.cpp:1175–1182` → `lod_compact_touch` |
| `VulkanGSSelectionPolygonRasterizeUniforms` | `gs_renderer.h:124–133` | 32 B (8×`uint32_t`) | **Missing** | Yes — `gs_renderer.cpp:1740–1755` → `selection_polygon_rasterize` |

**Evidence:** All five `PACK_STRUCT` push types in `gs_renderer.h`; only three have `static_assert`. Shader-side layouts currently match field-for-field (`lod_compact_touch.slang` Uniforms; `selection_polygon_rasterize.slang` `PolygonRasterizeUniforms`), so this is **drift prevention**, not a proven live ABI bug.

| **Severity** | **Low–Medium** |
| **Minimal action** | `static_assert(sizeof(VulkanGSLodCompactUniforms) == 16);` and `static_assert(sizeof(VulkanGSSelectionPolygonRasterizeUniforms) == 32);` next to the structs (mirror sibling asserts). |

### C4.2 — Local executeCompute push structs without PACK_STRUCT / sizeof assert

Verified call-site locals (all-`uint32_t`, ABI-stable on current targets, but unguarded against future field inserts):

| Struct | Host | sizeof today | Shader twin |
|---|---|---|---|
| `Uniforms` (lod map) | `gs_renderer.cpp:1090–1094` | 16 | `lod_map_indices.slang` |
| `VisibleUniforms` | `:2237–2240` | 16 | `visible_flags` / `compact_visible_primitives` |
| `PrepareUniforms` (classic) | `:2266–2270` | 16 | `prepare_visible_sort.slang` |
| `PrepareUniforms` (visible chain) | `:2590–2594` | 16 | `prepare_visible_chain.slang` |
| `ApplyUniforms` | `:2385–2388` | 16 | `apply_depth_ordering.slang` |
| `CopyUniforms` | `:2646–2649` | 16 | `copy_visible_indices.slang` |

| **Severity** | **Low** (uniform scalar layout; risk rises if floats/padding added) |
| **Minimal action** | Optional: hoist to `PACK_STRUCT` + `static_assert` next to shader field counts, or one shared header.

**Not missing:** `VulkanGSRendererUniforms` (192), `VulkanGSLodSelectUniforms` (144), `VulkanGSSelectionMaskUniforms` (176) — asserts present.

---

## 5. Structural search notes (for reproducibility)

- **ast-grep / sg 0.39.6** used for operator=/struct shape probes; shallow-copy detection completed with a small Python structural pass over `src/`+`tests/` (brace-balanced struct bodies containing `VkBuffer`/`VmaAllocation`/`CUdeviceptr` + copy policy + nearby destroy).
- **RAII+destroy without deleted copy:** only `ScopedStagingBuffer` in `src/`.
- **Explicit shallow handle copy:** only `_VulkanBuffer` + `Buffer<T>` (production code).
- **`require_backing`:** single definition, only `executeCumsum` (classic).
- **tests/:** exercise `_VulkanBuffer` forges and barrier planner; no additional production ownership types. Test helpers that copy handle IDs are intentional fakes.

---

## 6. Priority order (if fixing)

1. **C1.1** — make `_VulkanBuffer`/`Buffer` non-copyable (or view-only copy).  
2. **C2.1–C2.3** — host contracts for dual-source scan/tail-index sites (cheap asserts).  
3. **C1.2–C1.3** — RAII/POD ownership hygiene in point-cloud path.  
4. **C4.1** — two `static_assert`s.  
5. **S1–S3** — only after host contracts; shaders already match intended host coupling.

---

*End of sweep C. No source modifications made.*
