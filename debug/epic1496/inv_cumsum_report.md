# Epic #1496 — Cumsum OOB Investigation (`VUID-vkCmdDispatch-storageBuffers-06936`)

**Mode:** read-only (no repo files modified)  
**Branch:** `epic-1496-computed-barriers`  
**Evidence:** `gui_val_p3.log` (RAD / LOD), `gui_val_p1.log` (3k.ply), source + binary disassembly of `build/LichtFeld-Studio`  
**Date:** 2026-08-06

---

## 1. Symptom (validated)

From `gui_val_p3.log` (first instrumented frame after loading `splat_30000.rad`):

| Dispatch | Shader | Binding | Bound VkBuffer (debug name) | VkBuffer size | Bound range | Sample OOB gid | OOB byte |
|----------|--------|---------|------------------------------|---------------|-------------|----------------|----------|
| Index 0 | `cumsum/block_scan` | 1 (`g_output`) | `vksplat.buffer.visible_count` | **8** | **8** | 2976–2981 | ~11907–11927 |
| Index 1 | `cumsum/scan_block_sums` | 2 (`g_blockSums`) | `vksplat.buffer.visible_sort_dispatch_args` | **12** | **12** | 736–739 | ~2947–2959 |

Same class of hit on plain PLY (`gui_val_p1.log`): binding 1 again named `visible_count` (8 B), high gids (88000+, 2944+). **Not LOD-exclusive.**

Shader source attribution is real: `cumsum.slang:136` `g_output[gid] = …` and `:190` `g_blockSums[gid] = …`.

---

## 2. What is proven

### 2.1 Labels are truthful (not lying names)

- Dedicated allocations are named only in `createBuffer` (`buffer.cpp` ~281–284):
  `vksplat.buffer.{label}` with `label` from `_VulkanBuffer::label`.
- Shared-scratch parent is named `interop.imported.buffer[{bytes}]` (`vulkan_context.cpp` ~3287–3290), **never** `vksplat.buffer.*`.
- Observed sizes match the **only** intended dedicated sizes of those fields:
  - `visible_count` → `resizeDeviceBuffer(..., 2)` → **8 bytes** (`gs_renderer.cpp` ~2334, ~2721)
  - `visible_sort_dispatch_args` → `VisibleSortDispatch::kLayout.word_count == 3` → **12 bytes**
- Therefore GPU-AV is not mis-labeling a large `tile_batch_offsets` / `_cumsum_blockSums` region. The **handles** at those descriptors really are those small dedicated buffers.

### 2.2 SPIR-V / pipeline binding numbers are correct

Disassembled `cumsum_block_scan.spv`:

| Id name | set | binding |
|---------|-----|---------|
| `g_input` | 0 | 0 |
| `g_output` | 0 | 1 |
| `g_blockSums` | 0 | 2 |

`pipeline_cumsum.block_scan = _ComputePipeline(3)` → layouts `[0,1,2]`.  
`recordComputeDispatch` writes `buffers[binding]` to `dstBinding = binding` (`gs_pipeline.cpp` ~1981–2015). No off-by-one in the push-descriptor path.

### 2.3 Classic `executeCumsum` call sites pass the right **fields**

| Site | Input | Output |
|------|-------|--------|
| `executeSortPrimitivesByDepth` ~2369 | `visible_flags` | `visible_prefix` |
| `executeCalculateIndexBufferOffset` ~2011–2014 | `tiles_touched_depth_ordered` | `index_buffer_offset` |
| Legacy waves ~1529–1530 | `tile_batch_counts` | `tile_batch_offsets` |
| Macro waves ~3088–3089 | `tile_batch_counts` | `tile_batch_offsets` |

Indirect path (~3227+) binds `index_buffer_offset` / `_cumsum_blockSums*` / `cumsum_counts` — also not `visible_count`.

### 2.4 Binary matches source at the first cumsum call

`executeSortPrimitivesByDepth` in `LichtFeld-Studio`:

```text
lea  0x6f8(%buffers), %rax   ; visible_flags   (off 1784)
mov  %rax, saved_input
resizeDeviceBuffer(visible_flags, N)
resizeDeviceBuffer(visible_count, 2)              ; 0x7a8
resizeDeviceBuffer(visible_sort_dispatch_args, 3) ; 0x800
...
lea  0x750(%buffers), %rcx   ; visible_prefix  (off 1872)  → output arg
call executeCumsum(this, buffers, visible_flags, visible_prefix, ...)
```

Struct stride `sizeof(Buffer) == 88`; offsets match a live layout probe of the field order in `buffer.h`.

Inside `executeCumsum` (binary):

1. `num_elements = input.deviceBuffer.size / 4` (`size` at `Buffer+0x38`)
2. `resizeDeviceBuffer(output, num_elements)` (**`no_shrink` default = true**, `gs_pipeline.h` ~77–79)
3. Phase vector built as copies of:
   - `input.deviceBuffer`
   - `output.deviceBuffer`
   - `buffers._cumsum_blockSums` at **`buffers+0x1200`** (4608) — correct field, not a neighbor of `visible_count`

So the C++/asm path is **not** “wrong argument at the call site.”

### 2.5 How `numElements` is chosen

```cpp
// gs_renderer.cpp ~1815, 1824–1830
size_t num_elements = input_buffer.deviceSize(); // deviceBuffer.size / sizeof(int32_t)
// push constant Uniforms.numElements = active_elements for every phase
```

Shader gates writes with `gid < numElements` (`cumsum.slang` ~135–136, ~189–190).  
OOB at `gid=2976` ⇒ **push-constant `numElements > 2976`**. That count comes from the **input** buffer’s active byte size, not from the tiny output.

### 2.6 Which cumsum is failing (extent analysis)

Two-level path (`num_elements ≤ block²`, `block=1024`):

| Phase | `active_elements` | Needs buffer |
|-------|-------------------|--------------|
| `block_scan` | `N` | `output` ≥ `4N` |
| `scan_block_sums` | `ceil(N/1024)` | `_cumsum_blockSums` ≥ `4·ceil(N/1024)` |

- OOB on `g_blockSums` at **gid ≈ 736** requires `active_elements ≳ 737` for that phase ⇒ `N ≳ 737·1024 ≈ 7.5e5`.
- Tile-grid-only sizes (~3k tiles) give `num_blocks = 3` → **cannot** produce scan OOB at 736.
- Matches **large-N** cumsum: first in the legacy path is **`visible_flags → visible_prefix`** with `N ≈ uniforms.num_splats` / selected LOD count (~1e6 on p3, ~5e6 on p1).

The earlier “~2977 ≈ tile count” reading is consistent with **one sampled OOB lane** on a much larger `N`, not with tile_batch cumsum’s scan phase.

### 2.7 ADD_OWNED / assignBufferLabels / shared-scratch order

- Two `ADD_OWNED` lists in `buffer.cpp` (~14–82 and ~96–164) are **identical** and only used for VRAM accounting — not positional assignment.
- `assignBufferLabels` (~318–392) assigns `#name` per named field — not a shifted table.
- `estimateSharedScratchBytes` and `bindSharedScratchBuffers` walk the **same** field sequence (verified by script). Macro conditionals match (`!macro_chain` skips the same three regions in both).
- Shared views use `makeResizableRegionView` → one parent `VkBuffer`, `allocation = NULL`. That path cannot produce an **8-byte** dedicated object named `vksplat.buffer.visible_count`.

---

## 3. Root cause (best supported)

### Statement

**At the cumsum `block_scan` / `scan_block_sums` dispatches, push descriptors for bindings 1 and 2 are the dedicated small buffers `visible_count` (8 B) and `visible_sort_dispatch_args` (12 B), while `Uniforms.numElements` is a large scan length from a large input (`visible_flags` / similar). The host call graph and binary for `executeCumsum` correctly *name* `visible_prefix` / `_cumsum_blockSums` as output/block-sums — so those fields’ `_VulkanBuffer` **values** at descriptor-build time are the small dedicated objects (or an equivalent alias of their handles/sizes).**

That is a **handle / view identity bug on the `Buffer` fields**, not a wrong SPIR-V binding table and not a wrong `executeCumsum(...)` argument list in source.

### Why this pair (shift pattern)

`visible_count` and `visible_sort_dispatch_args` are **adjacent** fields and are exactly prepare_visible_sort’s bindings 1 and 2:

```cpp
// gs_renderer.cpp ~2412–2416
{ visible_prefix, visible_count, visible_sort_dispatch_args }
```

while cumsum expects:

```cpp
{ input, output, _cumsum_blockSums }
```

So the **descriptor payload** looks like prepare_visible_sort’s last two slots, not a uniform “struct field shift” from `tile_batch_offsets`/`_cumsum_blockSums` (those are far apart in `VulkanGSPipelineBuffers`).

### What is *not* the root cause

| Hypothesis | Verdict |
|------------|---------|
| Wrong cumsum call-site arguments | **Rejected** — source + disassembly |
| SPIR-V / `buffer_layouts` off-by-one | **Rejected** — reflection + code |
| Label lie on a correctly large buffer | **Rejected** — sizes 8 and 12 match only those fields |
| Shared-scratch subrange of parent mis-offset alone | **Rejected as sole cause** — parent is multi‑MiB and named `interop.imported.*` |
| ADD_OWNED list order mismatch | **Rejected** — lists match; not used for binding |
| Epic #1496 tagged dispatch rewriting cumsum binds | **Rejected** — cumsum still untagged `vector<_VulkanBuffer>`; only `invalidate` |

### Mechanism still open (ownership / metadata)

No static assignment of the form  
`tile_batch_offsets.deviceBuffer = visible_count.deviceBuffer`  
exists. Grep of `.deviceBuffer =` only hits input-pool / LOD meta views.

Dangerous patterns that **can** put the wrong `VkBuffer` identity (or stale capacity/size) into a field:

1. **`_VulkanBuffer` / `Buffer` copy-assign shares handles with no refcount** (`buffer.h` ~45–67, ~108–116). Two fields can own one allocation; `destroyBuffer` on one frees it; the other keeps stale `capacity`/`allocSize`/`buffer`.
2. **`resizeDeviceBuffer` default `no_shrink = true`** (`gs_pipeline.h` ~77–79). Growth only when `capacity < need`. A **stale large capacity** on a handle that was recycled as an 8-byte buffer would skip recreate (see §4). `validateBufferRange` *should* catch `capacity > allocSize` via `hasValidViewBounds`, but any path that keeps `allocSize` stale-large with a recycled small handle would pass host checks and fail in GPU-AV with **large range** — which is *not* what p3 shows (range 8). So pure “stale capacity + recycled handle” is only a partial fit unless size also stayed 8.
3. **Shared-scratch bind/detach** (`vksplat_viewport_renderer.cpp` ~3437–3446, ~3621–3633): soft-clear owned allocs, install views, later detach. Correct when used alone; combined with (1) and mid-batch `HOST_GUARD` resizes it is the highest-risk lifecycle.

**Honest gap:** a single line that assigns `visible_count`’s handle into `visible_prefix.deviceBuffer` (or into the phase vector) was **not** found. The failure mode is established at the **descriptor / field-value** layer; the exact producer of that value needs a targeted runtime assert (below) on the next repro.

---

## 4. Important API fact (fix-relevant)

```cpp
// gs_pipeline.h
void resizeDeviceBuffer(_VulkanBuffer&, size_t, bool no_shrink = true);
template <typename T>
_VulkanBuffer& resizeDeviceBuffer(Buffer<T>&, size_t, bool no_shrink = true);
```

`executeCumsum` calls `resizeDeviceBuffer(output_buffer, num_elements)` with the **default `no_shrink=true`**.  
It will **not** recreate when `capacity >= need`. Binary confirms this (`cmp capacity, need` / `no_shrink` path at `resizeDeviceBuffer(_VulkanBuffer&,…)`).

Also: `Buffer` inherits `std::vector` and **copy-assigns `deviceBuffer` handles**. That is a footgun for any future/hidden `Buffer` copy of `VulkanGSPipelineBuffers` members.

---

## 5. Minimal fix proposal (no refactors)

### Fix A — harden `executeCumsum` (smallest, local, should green GPU-AV)

In `VulkanGSRenderer::executeCumsum` (`gs_renderer.cpp` ~1815–1870), after computing `num_elements` and before any phase dispatch:

1. Resize with an explicit contract against **backing** size, not only `capacity`:
   - `resizeDeviceBuffer(output_buffer, num_elements, /*no_shrink=*/false)` **or** keep no_shrink but then:
   - If `output_buffer.deviceBuffer.allocSize < num_elements * sizeof(int32_t)` **or** `!hasValidViewBounds()` **or** `size`/`capacity` disagree with `allocSize` for owned buffers (`allocation != NULL` ⇒ `offset==0` and `capacity <= allocSize`), force `destroyBuffer` + `createBuffer`.
2. Same check for `_cumsum_blockSums` / `_cumsum_blockSums2` after their resizes (use required element counts per level).
3. **Debug/contract assert (keep in release as `throw_renderer_contract`):**
   - `output.deviceBuffer.buffer != buffers.visible_count.deviceBuffer.buffer` unless `&output_buffer == &buffers.visible_prefix` is false… better:  
     `output.deviceBuffer.allocSize >= num_elements * 4` and label is not unexpectedly `"visible_count"` when resizing a non-count field.
   - Optionally: `output.deviceBuffer.label == expected` is too strict for shared views; prefer **allocSize / range**.

This does not require finding the producer: it makes a silent wrong small handle **fail host-side** or **self-repair** before `vkCmdDispatch`.

### Fix B — stop handle aliasing (minimal ownership rule)

Still small, slightly broader:

- In `destroyBuffer`, if desired later: do not allow two live `Buffer` fields to share `allocation != NULL` (census / debug only).
- Short term: **delete `Buffer` copy-assign / make `deviceBuffer` move-only** is a refactor — **out of scope** for “minimal.” Instead: ban `VulkanGSPipelineBuffers` value copies (already mostly reference-passed).

### Fix C — only if Fix A asserts fire on `visible_prefix` still holding `visible_count`

Then chase lifecycle around:

- `bindSharedScratchBuffers` / `detachSharedScratchBuffers` / `releasePrivateScratchBuffers`
- any path that `destroyBuffer`s one field while another still holds the same handle after a `_VulkanBuffer` copy

No evidence yet of a wrong line inside `bindSharedScratchBuffers` field order.

### Do **not** do

- Change cumsum.slang bindings or add robustness as the “fix.”
- Reorder `ADD_OWNED` / `assignBufferLabels` lists (not causal).
- Blindly grow every buffer on every frame without the allocSize check (hides ownership bugs).

---

## 6. Suggested verification after fix

```bash
scripts/run_vulkan_validation.sh -- splat_30000.rad
# and
scripts/run_vulkan_validation.sh -- 3k.ply
```

Expect: no `VUID-vkCmdDispatch-storageBuffers-06936` on `cumsum/block_scan` / `scan_block_sums` with `visible_count` / `visible_sort_dispatch_args`.

Optional one-frame host log (temporary): on enter to `executeCumsum`, print  
`output.label, output.buffer, output.allocSize, output.capacity, output.size, visible_count.buffer, visible_count.allocSize`  
— should show mismatch before Fix A and agreement after.

---

## 7. File:line index

| Item | Location |
|------|----------|
| Validation symptom | `gui_val_p3.log` ~131–231 |
| Cumsum shader OOB lines | `cumsum.slang` 136, 190 |
| `num_elements` + phase binds | `gs_renderer.cpp` 1815–1907 |
| First large-N cumsum call | `gs_renderer.cpp` 2369 (`visible_flags`→`visible_prefix`) |
| `visible_count` / sort-args sizing | `gs_renderer.cpp` 2334–2358 |
| Push descriptor bind | `gs_pipeline.cpp` 1981–2022 |
| `createBuffer` debug name | `buffer.cpp` 281–284 |
| `resizeDeviceBuffer` grow/no_shrink | `buffer.cpp` 345–370; default `gs_pipeline.h` 77–79 |
| `Buffer` handle copy-assign | `buffer.h` 45–67, 108–116 |
| Shared scratch bind | `vksplat_viewport_renderer.cpp` 3427–3508 |
| Field layout | `buffer.h` 165–168, 197–217 |

---

## 8. Bottom line

- **Root cause class:** wrong **dedicated** `VkBuffer` identity (and range) on cumsum storage bindings 1 and 2 — specifically the real `visible_count` (8 B) and `visible_sort_dispatch_args` (12 B) objects — while `numElements` is large from a correct large input.
- **Not** a mislabeled large `tile_batch_offsets`, and **not** a wrong `executeCumsum` argument list in source/asm.
- **Failing workload:** large-N classic two-level cumsum (primarily `visible_flags`→`visible_prefix` on the first legacy frame); tile-count-only scan phases do not explain binding-2 OOB at gid ~736.
- **Minimal fix:** contract-check + force-repair of `output` / `_cumsum_blockSums*` **backing** sizes inside `executeCumsum` before dispatch (Fix A); then use that assert to pin any remaining alias producer if it still fires.
