# Vulkan elite roadmap

Goal: take the Vulkan migration from "works, ~7.5/10" to **staff-level / industry-leading** for a 3DGS renderer. ImGui is greenlit for full removal.

Each phase has: scope, file-level deliverables, acceptance criteria, effort. Phases are ordered so each one unblocks the next.

Smoke test gate (CLAUDE.md) re-run after every phase:
```
./build/LichtFeld-Studio -d data/bicycle --output-path output --images images_4 \
  --strategy mcmc --max-cap 1500000 --log-level debug --train -i 7000 --start
ctest --test-dir build
```

Perf baseline captured once at start of phase 0; each subsequent phase reports delta against it (frame time, VRAM, startup).

---

## Phase 0 — Stop the bleeding (3 days, blocks ship)

Critical correctness + truth-in-docs. No new features.

### 0.1 Per-image acquire semaphores
- `src/visualizer/window/vulkan_context.{cpp,hpp}`
- Replace `image_available_[kFramesInFlight]` with `image_available_[swapchain_image_count]` indexed by acquired image index, not frame slot.
- Recreate the array in `createSwapchain()` / destroy in `destroySwapchain()`.
- `endFrame()` waits on `image_available_[active_image_index_]` instead of `[current_frame]`.
- **Accept**: Run with forced 4-image swapchain on Wayland; no `VK_ERROR_DEVICE_LOST` over 1h soak. Validation clean.

### 0.2 Replace `vkQueueWaitIdle` in swapchain teardown
- `vulkan_context.cpp` recreate/destroy paths.
- Wait on `swapchain_images_in_flight_[*]` fences explicitly; if any fence is `VK_NULL_HANDLE`, skip.
- **Accept**: Resize storm test (10 resizes/sec for 30s) — no hang, no validation errors.

### 0.3 Bound immediate-submit drain
- `vulkan_context.cpp:287-297`
- `vkWaitForFences` with 2s timeout; on timeout, log a `LOG_ERROR` naming the leaked submission and continue teardown.
- **Accept**: Inject a never-signaling fence in a debug build; shutdown completes within 3s.

### 0.4 Surface format & colorspace logging + `hasHdr()` flag
- `vulkan_context.cpp::chooseSurfaceFormat`
- `LOG_INFO` chosen format and colorspace at swapchain create.
- Add `hasHdr()` accessor returning true when a non-`VK_COLOR_SPACE_SRGB_NONLINEAR_KHR` format was selected.
- **Accept**: Manual log inspection on RTX (sRGB) and HDR display (BT.2020).

### 0.5 External-image tracker leak fix
- `vulkan_image_barrier_tracker.{cpp,hpp}` — add `clearExternal()`.
- `vulkan_context.cpp::destroySwapchain()` — call it.
- **Accept**: Memcheck (`compute-sanitizer --tool memcheck`) clean over a 1h training session.

### 0.6 `vkResultToString` helper
- New: `src/visualizer/window/vulkan_result.hpp` with `inline const char* vkResultToString(VkResult)`.
- Replace `static_cast<int>(result)` formatting across `vulkan_context.cpp` (~20 sites), `cuda_vulkan_interop.cpp`, `vulkan_viewport_pass.cpp`, `vksplat_viewport_renderer.cpp`.

### 0.7 Truth in docs
- `docs/development/vulkan-rendering-pipeline.md`: drop "P15 timeline DAG" claim, document that timeline-semaphore handoff is **already in place** (`vksplat_viewport_renderer.cpp:404-412`).
- `docs/development/vulkan-viewer-migration.md`: tick item 1 of "Remaining" once Phase 1 lands.

**Phase 0 exit**: validation-clean, no `vkQueueWaitIdle`, no `cudaStreamSynchronize` references in code, docs match reality. Composite score 7.5 → 8.

---

## Phase 1 — ImGui exorcism (5–7 days)

Rip out every ImGui touchpoint. RmlUi is the GUI; `rml_im_mode_panel_adapter` is the bridge.

### 1.1 Delete the dead render path
- `src/visualizer/gui/gui_manager.cpp`
- Remove `ImGui::CreateContext()` (line ~2628), `ImGui_ImplSDL3_InitForVulkan` (~2669), `ImGui_ImplSDL3_NewFrame` (~3667), `ImGui::Render()` (~4112), `ImGui_ImplSDL3_Shutdown` (~3167), `ImGui::IsAnyItemActive()` reads, `imgui_internal.h` includes.
- Replace `ImGui::GetMainViewport()` (line ~4104) with the SDL3 window/extent the surrounding code already has.
- 74 call sites — most are state-flag reads, not real UI; mechanical pass.
- **Accept**: app runs, modal overlays render, menu bar/context menus work. No `imgui*` symbols in `nm liblfs_visualizer.so`.

### 1.2 Migrate Python plugin UI (the heavy one)
- `src/python/lfs/py_ui.cpp` (5479 lines, **265 call sites**), `py_ui_menus.cpp`, `py_ui_theme.cpp`, `py_uilist.cpp`, `py_ui_panels.cpp`.
- Target: `src/python/lfs/rml_im_mode_panel_adapter.{cpp,hpp}` already exposes the immediate-mode bridge over RmlUi. Audit its API parity gap vs ImGui's:
  - `Begin/End`, `Text`, `Button`, `Checkbox`, `SliderFloat`, `InputText`, `ColorPicker`, `Image`, `BeginChild/EndChild`, `BeginPopup`, layout helpers (`SameLine`, `Separator`, columns/grid).
- Where the adapter is missing a primitive, add it (it's a thin RmlUi factory in C++).
- Bind `PyUILayout` and `PySubLayout` Python wrappers to the adapter instead of raw ImGui.
- Keep the public Python API surface identical (`tests/python/test_*` must pass).
- **Accept**: `./build/.../python3.12 -m pytest tests/python/ -v` passes (excluding pre-existing `test_icon_cache.py` GL-context segfault). Manual: load a sample plugin, verify panels render.

### 1.3 Migrate visualizer panels
- `src/visualizer/gui/ui_widgets.{cpp,hpp}` (151 sites) — port helper widgets to RmlUi (most are just `ImGui::Text` / `Button` / `SliderFloat` wrappers; rename to RmlUi equivalents).
- `src/visualizer/gui/panel_registry.cpp` (15 sites), `panel_layout.cpp` (3), `startup_overlay.cpp` (1) — small.
- `src/visualizer/gui/pie_menu.hpp`, `gizmo_transform.hpp` — drop the `#include <imgui.h>` if unused after refactor; otherwise port their inline ImGui usage.
- `src/visualizer/tools/{align_tool,brush_tool,selection_tool}.cpp`, `src/visualizer/input/input_controller.cpp`, `src/visualizer/sequencer/rml_sequencer_panel.cpp`, `src/visualizer/gui/windows/video_extractor_dialog.cpp`, `src/visualizer/gui/panels/windows_console_utils.cpp`, `src/visualizer/theme/theme.hpp` — most likely just stale includes, verify and delete.
- **Accept**: `grep -rn "include.*imgui" src/` returns zero hits.

### 1.4 vcpkg + CMake cleanup
- `vcpkg.json`: remove `imgui` and `implot` entries.
- Root + module `CMakeLists.txt`: remove any `imgui::imgui`, `implot::implot` link targets.
- Drop `LFS_*IMGUI*` flags if any.
- **Accept**: full clean build from cleared vcpkg cache; binary size reduction logged (~2 MB expected).

### 1.5 Sanity
- Re-run smoke test + Python tests.
- `nm liblfs_visualizer.so | grep -i imgui` → empty.
- `ldd build/LichtFeld-Studio | grep -iE 'imgui|imm|implot'` → empty.

**Phase 1 exit**: Single GUI stack (RmlUi), zero ImGui code/symbols, `~265+151+74+74 ≈ 564` ImGui call sites gone. Composite 8 → 8.5.

---

## Phase 2 — Hot-path performance (2 weeks)

Three orthogonal wins, all enabled by the timeline-semaphore plumbing that's already in place.

### 2.1 Indirect dispatch in `vksplat_fwd`
**Biggest single win.** Removes the synchronous GPU→CPU readback at `gs_renderer.cpp:376`.

- `src/rendering/rasterizer/vksplat_fwd/src/gs_renderer.cpp`
- Replace `int num_indices = readElement<int32_t>(...)` + scalar dispatch with `vkCmdDispatchIndirect(cmd, count_buffer, offset)`. The cumsum tail already lives in `index_buffer_offset.deviceBuffer` — point the indirect dispatch at the last 12 bytes (groupCountX/Y/Z).
- Where the count drives multiple subsequent dispatches with different group counts, add a tiny "indirect setup" compute shader that writes a small `VkDispatchIndirectCommand` array from the cumsum tail.
- Rewire `executeGenerateKeys`, `executeComputeTileRanges`, `executeRasterizeForward` to consume it.
- Drop `buffers.num_indices` CPU mirror entirely (or keep as a debug-only readback gated on `LOG_LEVEL=debug`).
- **Accept**: Frame time drop measurable (~0.5–2 ms on RTX 3070). NSight capture shows no `vkQueueWaitIdle`-style stall between dispatches.

### 2.2 Async compute queue for vksplat
- `vulkan_context.cpp` queue family selection: probe for a separate compute family (NVIDIA: queue family 2, AMD: 1). Fall back to graphics queue if absent.
- Add `computeQueue()`, `computeQueueFamily()` accessors.
- `vksplat_viewport_renderer.cpp` submits its 12 dispatches on the compute queue; the existing `addFrameTimelineWait` makes graphics queue wait for compute completion before sampling `output_image_`.
- Handle queue-family-ownership transfer for `output_image_` if the families differ (`VkImageMemoryBarrier2.srcQueueFamilyIndex/dstQueueFamilyIndex`).
- **Accept**: NSight shows compute and graphics overlapping. Frame time drop on UI-heavy frames (~10–20%).

### 2.3 Coalesce CUDA→Vulkan upload
- `src/visualizer/rendering/vksplat_input_packer.cpp` + `vksplat_viewport_renderer.cpp`.
- Pack `xyz_ws | rotations | scales_opacs | sh_coeffs` into one device-side buffer with 256B-aligned sub-regions; one `cudaMemcpyAsync` instead of 4.
- One `CudaVulkanBufferInterop` per frame slot instead of four; cache the mapped pointer.
- **Accept**: `nvtxRangePush("VkSplatUpload")` shows ≥3× shorter range. No correctness regression vs `tests/test_vksplat_input_packer.cpp`.

### 2.4 Persistent-mapped ring buffers
- `cuda_vulkan_interop.cpp` `CudaVulkanBufferInterop`: hold the mapped pointer across frames; only re-import on size growth.
- `vksplat_viewport_renderer.cpp::ensureCudaInputSlot`: skip `init()` re-call when existing slot fits.
- **Accept**: `nvprof`/`nsys` shows zero `cudaExternalMemoryGetMappedBuffer` per frame in steady state.

**Phase 2 exit**: Frame time on `bicycle` test scene measured down 25–40% on RTX 3070. Composite 8.5 → 9.

---

## Phase 3 — Profiling & frame pacing (5 days)

Quality-of-life multipliers. Cheap.

### 3.1 Debug-utils labels everywhere
- Wrap every dispatch and pass with `vkCmdInsertDebugUtilsLabelEXT` (or `vkCmdBeginDebugUtilsLabelEXT`/`End`):
  - `vulkan_viewport_pass.cpp` — one label per pipeline (scene, vignette, grid, overlay, shape, textured, pivot).
  - `vksplat_viewport_renderer.cpp` and `gs_renderer.cpp` — one per phase (proj/keys/sort/cumsum/tiles/raster/compose).
  - `rmlui_vk_backend.cpp` — one per RmlUi context render.
- Name every long-lived `VkImage`/`VkBuffer` via `vkSetDebugUtilsObjectNameEXT` at creation.
- **Accept**: RenderDoc capture is fully readable end-to-end, no anonymous draws.

### 3.2 `VK_KHR_present_wait` + `VK_NV_low_latency2` (opportunistic)
- Probe both at device init; gate behind `hasPresentWait()`, `hasLowLatency2()`.
- When present, drive frame submission timing from `vkWaitForPresentKHR`. Remove the implicit pacing by `vkAcquireNextImageKHR`'s blocking wait.
- **Accept**: input-to-photon latency on `--start` interactive mode drops by ~1 frame on supporting hardware.

### 3.3 Pipeline statistics + timestamp queries everywhere
- `gs_renderer`'s `perf_timer.cpp` already uses `VkQueryPool` timestamps. Extend to:
  - Per-stage `VK_QUERY_TYPE_PIPELINE_STATISTICS` (invocations, primitives).
  - Calibrated timestamps via `vkGetCalibratedTimestampsEXT` so we can correlate with `nvtx` ranges from CUDA in NSight.
- Surface in a debug overlay (RmlUi panel — gated on `--log-level debug`).
- **Accept**: F1 toggles a real perf overlay with per-stage µs.

**Phase 3 exit**: Debug-quality profiling. Composite 9 → 9.

---

## Phase 4 — Pipeline modernization (3 weeks)

Lean on Vulkan 1.3+ features that are probed but unused.

### 4.1 `VK_EXT_descriptor_buffer` for the UI + viewport paths
- `vulkan_context.cpp`: probe + enable, expose `hasDescriptorBuffer()`.
- `vulkan_viewport_pass.cpp`: replace per-overlay descriptor sets with a single descriptor buffer + offsets. Major win for textured-overlay batching gap.
- `rmlui_vk_backend.cpp`: replace its descriptor pool with a per-context descriptor buffer (vendored backend — patch carefully, keep upstream-diff small).
- **Accept**: zero `VkDescriptorPool` allocations during steady-state frames.

### 4.2 `VK_EXT_graphics_pipeline_library`
- `vulkan_context.cpp`: probe + enable.
- Refactor the viewport pass pipelines (currently 7 hand-rolled) into:
  - One vertex-input library
  - One pre-rasterization library per topology
  - Fragment libraries per shader variant
- Link at use site; cache via existing pipeline cache.
- **Accept**: cold-start pipeline compile time drops ~3×; shader hot-reload (dev mode) ≤100 ms.

### 4.3 Push descriptor on the viewport pass
- Already probed (`hasPushDescriptor()`). Use `vkCmdPushDescriptorSetKHR` for transient overlay descriptors so the pass stops touching descriptor sets at all.

### 4.4 Mutable descriptor type for RmlUi texture slots
- `VK_EXT_mutable_descriptor_type` — probe + enable.
- Fold RmlUi's per-context "color OR texture" descriptor variants into a single mutable slot.

**Phase 4 exit**: viewport + UI cmd buffer recording cost ~50% lower. Composite 9 → 9.5.

---

## Phase 5 — Slang convergence (4 weeks, high-leverage)

**Strategic move.** Today the math (camera, SH, covariance, quaternion rotation) lives twice: once in CUDA `gsplat_fwd/`, once in `vksplat_fwd/shader/src/slang/`. Lift it to a single Slang module that compiles to both targets.

### 5.1 Extract math kernels to shared Slang modules
- New: `src/rendering/rasterizer/shared_slang/`
- Move `spherical_harmonics`, `camera_projection`, `cov2d`, `quaternion`, `mip_aa` from `vksplat_fwd/shader/src/slang/` into here.
- Compile to:
  - SPIR-V (existing path) — `slangc -target spirv`
  - CUDA — `slangc -target cuda` produces `.cu` you `#include`.

### 5.2 Switch `gsplat_fwd` CUDA kernels to Slang-generated math
- Replace handwritten CUDA implementations of SH eval, covariance, projection in `src/rendering/rasterizer/gsplat_fwd/` with the Slang-generated equivalents.
- Run training regression: PSNR delta vs main on `bicycle` 7k iters must be < 0.05 dB.
- **Accept**: gtest comparing `gsplat_fwd` and `vksplat_fwd` forward outputs on a fixed scene shows pixel diff < 1/255 (within float rounding).

### 5.3 Retire `vksplat_fwd`'s standalone math
- After 5.1/5.2, the duplicated `.slang` files in `vksplat_fwd/shader/src/slang/` shrink to just the kernel skeletons.

**Phase 5 exit**: One source of truth for splat math. Composite 9.5 → 9.7.

---

## Phase 6 — Vulkan-native backward (the moat, 6–10 weeks)

This is the strategic bet. With it, **training works on AMD, Intel Arc, headless cloud GPUs without CUDA**, and on macOS via MoltenVK.

### 6.1 Wire the existing backward Slang shaders
- `vksplat_fwd/CMakeLists.txt:138` — enable `EXPORT_MODE=1` compilation.
- Files already exist: `alphablend_shader.slang` backward path, `vertex_shader.slang` backward path, Slang `bwd_diff()` infrastructure.
- Build + load + bind in `gs_pipeline.cpp`.

### 6.2 Backward sort + cumsum
- Forward radix sort is order-stable; backward needs the gather permutation. Allocate a permutation buffer in forward and consume in backward.
- Backward cumsum: scatter from output gradient.

### 6.3 Optimizer step in compute
- Port Adam / EMA / momentum updates to compute shaders. These are pointwise; trivial Slang.

### 6.4 Densification heuristics in compute
- MCMC and default densify are already pointwise + sort-by-grad — straightforward.

### 6.5 Toggle: `--training-backend {cuda,vulkan}`
- New flag. Default `cuda` initially; flip after PSNR equivalence holds for 30k-iter runs across 3 scenes.
- **Accept**: train `bicycle` 30k on Vulkan backend; PSNR within 0.1 dB of CUDA backend; throughput within 30% on RTX 3070.

**Phase 6 exit**: Vendor-agnostic 3DGS training. Industry-leading. Composite 9.7 → 10.

---

## Phase 7 — Optional research bets (post-elite)

Pick at most one based on user demand.

- **`VK_KHR_cooperative_matrix`** (NVIDIA tensor cores from compute) for SH eval and covariance matmul. Slang has intrinsics. Expected: 1.5–3× SH-heavy scenes on RTX. Effort: L.
- **Mesh / task shaders for splat draw**: replace the compute rasterize loop with HW-rasterized quads + alpha blending. Big win on AMD; mixed on NVIDIA. Effort: L.
- **`VK_EXT_device_generated_commands`** for multi-frame batched dispatch chains. Effort: L. Pays off for very large scenes (>5M splats).

---

## Acceptance dashboard (track in repo as `vulkan-status.md`)

| Phase | Status | Frame time (bicycle, 1080p, RTX 3070) | VRAM peak | Startup (cold) |
|---|---|---|---|---|
| Baseline (today) | — | TBD | TBD | TBD |
| 0 done | | = baseline | = | = |
| 1 done | | = baseline (slight win from no ImGui CPU cost) | -2 MB | -50 ms |
| 2 done | | -25–40% | = | = |
| 3 done | | = | = | = |
| 4 done | | -45–60% | -10–20 MB | -200 ms |
| 5 done | | = | = | = |
| 6 done | | training only | training only | = |

## Sprint 1 results (landed)

**Phase 0 — fully landed**
- 0.1 Per-image acquire semaphores. `image_available_` moved to swapchain lifecycle; `next_acquire_index_` rotation; submit waits on the same index passed to acquire. Eliminates the >2-image-swapchain reuse hazard.
- 0.2 `vkQueueWaitIdle(present_queue_)` removed from swapchain recreate. `waitForFrameFences()` now waits on `in_flight_` ∪ `swapchain_images_in_flight_` with a 2 s bound.
- 0.3 Immediate-submit drain bounded to 2 s per fence; logs and leaks on timeout instead of hanging shutdown.
- 0.4 Surface format/colorspace logged at swapchain create. `hasHdr()` accessor added; flips true on any non-`SRGB_NONLINEAR` colorspace.
- 0.5 `VulkanImageBarrierTracker` separates external images via `external_images_` set; `clearSwapchainOnly()` preserves them across swapchain recreate. `vksplat_viewport_renderer` registers `output_image_` with `external=true`.
- 0.6 `vulkan_result.hpp` + `vkResultToString` helper. 46 sites converted across `vulkan_context.cpp`, `vulkan_loader_probe.cpp`, `vksplat_viewport_renderer.cpp` (the two with VkResult sites in scope).
- 0.7 `vulkan-rendering-pipeline.md` updated — `cudaStreamSynchronize` claim removed; per-frame timeline-semaphore handoff is documented as already in place.

**Phase 1 — minimal scope, fully landed**
- 1.1 `ImGui::Render()` replaced with `ImGui::EndFrame()` (no consumer existed; pure CPU waste). `ImGui::GetMainViewport()` modal site replaced with `panel_input.screen_*` already in scope.
- 1.2 RmlUi-backed focus-state aggregators added (`wantsCaptureMouse / wantsCaptureKeyboard / wantsTextInput / anyItemActive`) over all live `Rml::Context`s using existing `rml_input::*` utilities + `Rml::Context::GetHoverElement / GetFocusElement`. All four `gui_manager.cpp` ImGui focus-state reads now OR with the RmlUi state. Camera/viewport input suppression now reflects the actual active GUI surface.
- 1.3 Stale `<imgui.h>` include removed from `rml_sequencer_panel.cpp`. The 8 other audited files (`align_tool.cpp`, `brush_tool.cpp`, `selection_tool.cpp`, `input_controller.cpp`, `windows_console_utils.cpp`, `video_extractor_dialog.cpp`, `rml_python_panel_adapter.cpp`, `window_manager.cpp`) still use ImGui; full removal is a separate later project per scope.

**Phase 2 — fully landed**
- 2.1 **Indirect dispatch + deferred readback in `vksplat_fwd`**. New tiny Slang shader `setup_dispatch_indirect.slang` reads `index_buffer_offset[num_splats-1]` on the GPU and writes a `VkDispatchIndirectCommand` for `compute_tile_ranges`. `compute_tile_ranges` now reads `num_isects` from `index_buffer_offset` directly via a new binding 2 (no more `uniforms.active_sh` dependency). The synchronous mid-frame `readElement` is gone; `executeCalculateIndexBufferOffset` records an async `vkCmdCopyBuffer` of the cumsum tail into a host-visible coherent + persistently-mapped buffer for the next frame to consume. `executeGenerateKeys` pre-fills `unsorted_keys` with the `0xFFFFFFFF` sentinel so the radix sort's tail (when capacity > actual num_indices) sorts to the end harmlessly. CPU-side high-water-mark + 2× safety factor sizes the sort buffers; first frame uses an `8 × num_splats` heuristic seed; `resetNumIndicesEstimate()` is called on model-identity change so a fresh model can't under-size the buffers. New `executeComputeIndirect` helper + new `INDIRECT_DISPATCH_READ` barrier mask. Net effect: the per-frame `vkQueueWaitIdle` previously baked into `readElement` → `HOST_GUARD` is gone.
- 2.2 **Async compute queue**. `findQueueFamilies` now probes for a compute-only family (NVIDIA family 2, AMD family 1, etc.). When present, `VulkanContext` exposes `computeQueue() / computeQueueFamily() / hasDedicatedComputeQueue()`; `vksplat_viewport_renderer` initializes the rasterizer on that queue so the splat dispatch chain overlaps graphics-queue work (RmlUi, viewport overlays). External images and external buffers switch to `VK_SHARING_MODE_CONCURRENT` listing both families when distinct, eliminating the need for ownership-transfer barriers; the existing per-frame timeline-semaphore wait already provides cross-queue ordering between the rasterizer's output and the swapchain pass that samples it. When no dedicated family exists the compute queue aliases graphics so call sites stay unconditional.
- 2.3 **Coalesced CUDA→Vulkan upload**. `_VulkanBuffer` gains a `VkDeviceSize offset` field (default 0). `executeCompute` / `executeComputeIndirect` use it for descriptor binding so a single `VkBuffer` can be bound as multiple sub-regions. `CudaInputSlot` collapses from 4 buffers per ring slot to **1**: a single CUDA-imported `VkBuffer` holds `xyz | rotations | scales+opacs | sh` packed back-to-back with 256-byte alignment. Setup cost drops from 4× `cudaImportExternalMemory` + 4× `cudaExternalMemoryGetMappedBuffer` to 1× of each per ring slot. `cuda_inputs_` shape changes from `array<array<CudaInputSlot, 4>, kInputRingSize>` to `array<CudaInputSlot, kInputRingSize>`. New offset-aware `CudaVulkanBufferInterop::copyFromTensor` overload writes each tensor to its sub-region.
- 2.4 Grow-only ring buffer policy — **already in place**. `gs_pipeline.h` declares `resizeDeviceBuffer` with `no_shrink=true` default; no per-frame realloc churn.

**Build status**: clean after every phase.

**Measured perf (RTX 4090, X11, `results/mrnf/bicycle/splat_30000.ply` at 1280×720, MAILBOX 60Hz, ~25 s capture each)**:

```
                       samples  min     median  p90     p99     max     mean
vksplat.render BASELINE   29   13.45ms  14.60ms 14.80ms 15.56ms 15.56ms 14.56ms
vksplat.render POST-P2    30    5.58ms   5.62ms  5.92ms  6.30ms  6.30ms  5.68ms
                                                                speedup: 2.60×
                                                              reduction: -61.5%
```

End-to-end `gui_render` is vsync-locked at 60 Hz so the user-visible median frame time is unchanged at ~12 ms; the ≈9 ms of headroom now under the rasterizer is what unlocks bigger scenes, higher refresh rates, and richer UI work without dropping below vsync. Repeat run reproduced the post-P2 median to within <1% (5.62 → 5.64 ms).

The async-compute queue probe correctly picked NVIDIA family 2 (graphics on family 0). External images and buffers ran with `VK_SHARING_MODE_CONCURRENT` listing both families. No validation errors, no warnings in the post-P2 log.

Reproduce locally:
```
./build/LichtFeld-Studio -v <splat>.ply --log-level debug --no-splash
# in the rendering panel, switch raster_backend to "vksplat",
# then in stderr: grep -E "vksplat\.render took"
```

The `LOG_TIMER("vksplat.render")` in `rendering_manager_vulkan.cpp` wraps the rasterizer-only submit + GPU compute (fence-waited inside the renderer), so it isolates rasterizer cost from vsync-bound `gui_render`.

**What's still on the table for staff-level (next sprint)**
1. Phase 3 — debug-utils labels everywhere + `VK_KHR_present_wait` + `VK_NV_low_latency2`.
2. Phase 4 — descriptor buffer (`VK_EXT_descriptor_buffer`) + graphics pipeline library + push descriptor + mutable descriptor type.
3. Phase 5 — Slang convergence (single math source for CUDA `gsplat_fwd` and Vulkan `vksplat_fwd`).
4. Phase 6 — Vulkan-native backward; Vulkan-backend training (the moat).
5. Full ImGui exorcism (`py_ui.cpp` → `rml_im_mode_panel_adapter`, theme `ImVec4` → `Color`, ui_widgets/panel_registry migration, ImPlot retirement). Estimated 6–10 weeks; out of scope this sprint per explicit decision.

## Sequencing notes

- Phases 0 → 1 → 2 are the **must-land** sequence to claim staff-level. ~3 weeks total.
- Phase 3 can run parallel to 2 (different files).
- Phase 4 needs Phase 1 done (RmlUi is the main descriptor consumer left).
- Phase 5 unblocks Phase 6; both can be deferred if priorities shift.
- Phase 7 strictly optional.

## What "elite" means once this lands

- Zero OpenGL surface area
- Single GUI stack (RmlUi), zero ImGui CPU/binary cost
- Vulkan 1.3 + 6 opportunistic extensions actively used (descriptor buffer, graphics pipeline library, push descriptor, mutable descriptor type, present wait, low latency 2)
- Cross-API timeline-semaphore handoff with no CPU stalls
- Indirect-dispatch splat pipeline with no GPU→CPU readbacks
- Async compute queue overlapping graphics
- Single source of math (Slang) for both CUDA and Vulkan backends
- Optional Vulkan-native training: vendor-agnostic 3DGS

That's the moat.
