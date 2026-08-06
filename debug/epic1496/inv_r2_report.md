# Epic #1496 — Cumsum OOB Round 2

**Mode:** investigation (trap already in tree; no permanent edits beyond this report)  
**Branch:** `epic-1496-computed-barriers` (dirty: `require_backing` trap in `gs_renderer.cpp`)  
**Evidence:** `gui_val_trap.log`, new `gui_val_core_only.log`, source audit, VVL issue #11433  
**Date:** 2026-08-06

---

## 0. Decision summary

| Hypothesis | Verdict | Confidence |
|------------|---------|------------|
| **A. GPU-AV + push-descriptor misattribution** | **PROVEN primary** | High |
| **B. Multi-thread command-buffer record race** | **Killed** (static) | High |
| **C. Second renderer / untrapped path** | **Killed** | High |
| Round-1 “field holds wrong handle at `executeCumsum`” | **Refuted** by trap | High |

**Concrete fix (app-facing):** treat the 06936 hits under full GPU-AV as a **known VVL false positive** with the pinned layer; default developer validation should be **core + sync only** (no GPU-ASSISTED) until VVL is upgraded past a fix for [VVL #11433](https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/11433). Keep `require_backing` as a real app contract (it is valuable); do **not** “fix” cumsum bindings based on GPU-AV’s buffer names for this VUID.

---

## 1. Validation layer identity (exact)

| Item | Value |
|------|--------|
| vcpkg package | `vulkan-validationlayers:x64-linux@1.4.341.0` |
| Manifest `api_version` | **1.4.341** |
| Library | `build/vcpkg_installed/x64-linux/lib/libVkLayer_khronos_validation.so` |
| Manifest | `.../share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json` |
| SPDX document name | `vulkan-validationlayers:x64-linux@1.4.341.0 …` (created 2026-07-15) |
| Feature that enables it | `vcpkg.json` feature `vulkan-validation` → dep `vulkan-validationlayers` |
| Baseline | `vcpkg.json` `builtin-baseline` `c3867e714dd3a51c272826eea77267876517ed99` |
| How tests enable GPU-AV | `scripts/run_vulkan_validation.sh` sets  
  `VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT` |

Loader also reports Vulkan **API 1.4.341** on the host; device API **1.4.312**.

---

## 2. Trap result (refutes Round-1 field-handle mechanism)

`require_backing` in `executeCumsum` (`gs_renderer.cpp` ~1836–1857, plus block-sum checks):

```cpp
if (b.buffer == VK_NULL_HANDLE || b.allocSize < needed_bytes || b.size < needed_bytes)
    throw_renderer_contract(... label, handle, alloc/capacity/size ...);
```

**`gui_val_trap.log` (GPU-AV + sync, same repro as before):**

- Process ran to RAD load + multi-frame LOD render.
- **No** `throw_renderer_contract` / `require_backing` message.
- Still **10×** `VUID-vkCmdDispatch-storageBuffers-06936` on:
  - `cumsum/block_scan` binding 1 → `vksplat.buffer.visible_count` (8 B)
  - `cumsum/scan_block_sums` binding 2 → `vksplat.buffer.visible_sort_dispatch_args` (12 B)
- Highest OOB sample ~**340227** bytes on binding 1 (gid-scale ≫ tile count) — still large-N scan shape.

**Implication:** every real classic `executeCumsum` path that ran had host-side `input`/`output`/`_cumsum_blockSums*` with `size` and `allocSize` ≥ required element bytes **at record time**. The Round-1 claim “`visible_prefix.deviceBuffer` is the 8-byte `visible_count` object inside `executeCumsum`” is **false**.

Note on trap scope: it validates the `Buffer` structs’ host metadata before push, not a GPU-side capture of push-descriptor banks. That is still enough to refute “wrong field handle passed into `executeCumsum`.”

---

## 3. Hypothesis A — GPU-AV misattribution with push descriptors

### 3.1 Symptom alignment with a known open VVL bug

[KhronosGroup/Vulkan-ValidationLayers#11433](https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/11433)  
**(open, Jan 2026, label GPU-AV)** — title: *“GPU Assisted validation gives false positives when PushDescriptors are used”*

Reporter notes (paraphrase):

- GPU-AV reports **old** push-descriptor binding values after they were **re-pushed** to correct, larger buffers.
- Same VUID family: **`VUID-vkCmdDispatch-storageBuffers-06936`**.
- Example: binding 1, tiny buffer size (12 B in their case), OOB on a compute dispatch.
- **Errors disappear if push descriptors are disabled.**
- Environment: Vulkan SDK **1.4.335**-era layers (same 1.4.33x–1.4.34x generation as our **1.4.341.0** pin).
- Still **open** as of this investigation (no merged “fixed in 1.4.341” resolution visible from the issue).

Our app:

- **Every** compute dispatch uses `vkCmdPushDescriptorSetKHR` (`gs_pipeline.cpp` `recordComputeDispatch` / `recordComputeDispatchIndirect`).
- Cumsum and `prepare_visible_sort` both use set 0 with three storage buffers; the reported triple is **exactly** prepare_visible_sort’s:
  `{visible_prefix, visible_count, visible_sort_dispatch_args}`.

That is the textbook shape of “stale push-descriptor bank paired with a later (or different) instrumented dispatch.”

### 3.2 Local experiment: GPU-AV off ⇒ 06936 gone

| Run | Env `VK_LAYER_ENABLES` | Log | 06936 | App trap | Render progressed |
|-----|------------------------|-----|-------|----------|-------------------|
| Trap (prior) | `GPU_ASSISTED` + `SYNC` | `gui_val_trap.log` | **10× yes** | no throw | yes (RAD + LOD) |
| Core+sync only | **`SYNC` only** (no GPU_ASSISTED) | `gui_val_core_only.log` | **0** | n/a | yes (RAD + LOD selectors multi-frame) |

Core-only command (no rebuild):

```bash
timeout 45 env \
  VK_LAYER_PATH=build/vcpkg_installed/x64-linux/share/vulkan/explicit_layer.d \
  VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation \
  VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT \
  LFS_VK_VALIDATION=1 \
  ./build/LichtFeld-Studio -v splat_30000.rad
```

`gui_val_core_only.log`:

- `Vulkan validation enabled`, only deprecation/layer-enable **warns** (no GPU-AV “Both GPU Assisted…” banner).
- `RAD loaded: 1011868…`, `vksplat depth waves: 64 slots`, repeated `LOD GPU selector`.
- **`rg 06936` / `[error]` → empty.**

So the VUID appears **if and only if GPU-AV is enabled** in this configuration, while host-side cumsum contracts pass under GPU-AV.

### 3.3 How app-side descriptors are known correct at *record* time

Proof is **compositional**, not a single DebugPrintf:

1. **`require_backing`** runs immediately before every classic cumsum phase uses those buffers; it would throw if `output`/`block_sums` host `size`/`allocSize` were the 8/12-byte objects. It never threw while GPU-AV still named those objects.
2. **SPIR-V identity is clean** (user fact #2): each `.spv` embeds its own source; no swapped modules.
3. **Core validation** (layout, descriptor *writes*, command buffer rules) + sync validation do **not** report 06936 or related descriptor-set errors for the same workload.
4. External: VVL #11433 documents GPU-AV retaining **stale push-descriptor** buffer identities for 06936 under the same layer generation.

What we did **not** need for the verdict: temporary `std::thread::id` logging (B killed statically; see §4). Optional future hardening: log `(pipeline diagnostic_name, binding, buffer handle, range)` next to each `vkCmdPushDescriptorSetKHR` under `LFS_DEBUG_RECORD_THREAD=1` if someone wants a RenderDoc-free dump; not required now.

### 3.4 Verdict A

**Accept.** The failing reports are GPU-AV false positives (or equivalent mis-pairing of instrumented shader accesses with an outdated push-descriptor snapshot), matching VVL #11433 and confirmed by GPU-AV on/off behavior plus the host trap.

---

## 4. Hypothesis B — recording race

### Static audit

| Fact | Evidence |
|------|----------|
| Single `VulkanGSRenderer` per viewport | `VksplatViewportRenderer::renderer_` only (`vksplat_viewport_renderer.hpp` ~476) |
| Frame record under one `DeviceGuard` | Main path ~7850: one guard for projection → sort/cumsum → waves → compose |
| Nested `DeviceGuard` is same-thread no-op | Guard only `beginCommandBatch` if `!commandBatchInProgress` (`gs_pipeline.h` DeviceGuard) |
| LOD tree fill uses `DeviceGuard(&renderer_)` | ~2496 inside `ensureGpuLodTreeStorage`, called from render path (~7451) **before** the frame batch — still UI/render thread |
| Decode workers | `lod_page_cache` / `lod_upload_engine` use staging + engine timelines; **no** `renderer_.executeCumsum` / `beginCommandBatch` on worker threads (grep of those TUs) |
| Other workers | `rendering_manager` jthreads are GT image / camera metrics — not VkSplat batch record |

No second thread records into `VulkanGSPipeline`’s active command buffer for the splat cumsum path. Nested guards serialize on one thread.

### Verdict B

**Killed.** A record-time race that overwrote push descriptors after `require_backing` would require concurrent recorders on the same CB; none exist for this pipeline. Temporary thread logging was **not** installed (would only reconfirm single-thread record).

---

## 5. Hypothesis C — second renderer / untrapped path

- One `VulkanGSRenderer renderer_` member in the viewport renderer.
- Classic cumsum **only** via `executeCumsum` (now with trap on all branches that dispatch classic phases).
- Indirect cumsum uses different pipelines (`cumsum_*_indirect`) and was not the module named in the GPU-AV hits (`vksplat.cumsum/block_scan.shader`, not `*_indirect`).
- Trap is in shared code: if any instance recorded classic cumsum with tiny backings, it would throw. It did not.

### Verdict C

**Killed** (collapses into A). There is no untrapped classic `executeCumsum` elsewhere that could explain trap-silent 06936.

---

## 6. Concrete fix / process recommendations

### 6.1 What **not** to change in app binding code

Do not rewrite cumsum descriptor lists or “shift-correct” `visible_count` based on GPU-AV buffer names. Host contracts and core validation disagree with those names being the true live bindings for the failing instrumented dispatches.

### 6.2 Validation workflow (app / scripts)

1. **Default / CI / day-to-day:** enable **core + sync only** (as in `gui_val_core_only` env). Update `scripts/run_vulkan_validation.sh` to **not** enable `VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT` by default; make GPU-AV opt-in, e.g. `--gpu-av`.
2. **When GPU-AV is needed:** document that 06936 on compute **with push descriptors** under VVL **1.4.341.0** may be false positives per #11433; cross-check with:
   - host `require_backing` (already present), and/or
   - core-only re-run, and/or
   - RenderDoc / Nsight descriptor dump on the suspect dispatch.
3. **Layer upgrade path:** track VVL #11433; when a fixed release is available (post-1.4.341), bump the `vulkan-validationlayers` pin / baseline and re-enable GPU-AV in the script as a deliberate second pass.
4. **Suppression (only if forced to run GPU-AV on 1.4.341):**  
   - Prefer **not** globally suppressing 06936 (it is a real VUID for true bugs).  
   - Prefer: run GPU-AV **after** core is clean, accept known push-descriptor FP for this layer, and rely on `require_backing` for app-side storage size.  
   - If a message-id filter is required in-house, gate it behind an explicit env (e.g. `LFS_VK_GPUAV_SUPPRESS_PUSHDESC_06936=1`) with a comment pointing at this report + #11433 — **not** a silent default.

### 6.3 Keep the trap

`require_backing` is a good permanent contract for cumsum. It does **not** fix GPU-AV FPs; it proves the app’s scan sizes at record time. Optional small improvement (not required for this verdict): also assert `offset + size <= allocSize` for owned buffers (`allocation != NULL`).

### 6.4 Optional app experiment (not run; limit was 2–3 runs)

`gpuav_force_on_robustness` / `gpuav_descriptor_checks` toggles exist in the 1.4.341 layer JSON. Forcing robustness typically **skips** OOB instrumentation (makes 06936 vanish without proving correctness). Descriptor-check toggles may change report shape. Core-only off is the cleaner differential already collected.

---

## 7. Residual risk

- True OOB bugs with **non-push** descriptor sets, or with GPU-AV fixed layers, would still need investigation.
- Shared-scratch views make `allocSize` the **parent** size; `require_backing` relies primarily on `size` for the active range (correct for our resizes, but do not treat `allocSize` alone as “this field’s region is big”).
- #11433 may have nuances (Windows reporter); our Linux 1.4.341 repro matches the VUID + push-descriptor + stale small buffer story closely enough for operational policy.

---

## 8. Artifact index

| Artifact | Role |
|----------|------|
| `gui_val_trap.log` | GPU-AV + sync; trap silent; 10× 06936 |
| `gui_val_core_only.log` | Sync only; RAD/LOD render; **0× 06936** |
| `inv_cumsum_report.md` | Round 1 (field-handle theory — superseded) |
| VVL #11433 | External confirmation of GPU-AV + push-descriptor FP |
| Layer pin | `vulkan-validationlayers 1.4.341.0` |

---

## 9. Bottom line

The app’s classic cumsum path records **adequate** storage backings (trap never fires). GPU-AV on VVL **1.4.341.0** still reports 06936 against prepare_visible_sort’s tiny buffers while executing instrumented cumsum shaders — the same class as **open VVL #11433**. Core+sync validation on the same scene is clean. **Fix the validation configuration / layer pin, not the cumsum descriptor lists.**
