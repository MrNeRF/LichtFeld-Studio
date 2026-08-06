I'll explore the visualizer, rendering, interop, and training-sharing paths read-only and build a quantified report with file:line evidence.Next I'll read the interop, viewport pooling, and VRAM HUD paths, plus how the trainer shares splat data with the viewer.# GUI cost during training vs headless — optimization analysis

Read-only analysis of `src/visualizer/`, `src/rendering/`, trainer↔viewer sharing, viewport RT pooling (`#1497/#1562`), and `vram_hud_overlay.cpp`. Claims below are code-backed; formulas use the same constants as the sources.

---

## Executive summary

| Cost class | GUI vs headless | Notes |
|---|---|---|
| **Splat tensors (means/SH/…)** | **Not duplicated** on the fast path | One CUDA VMM exportable block; Vulkan **imports** the same physical memory |
| **Rasterizer scratch arena** | **Shared**, not purely additive | Viewer installs exportable scratch as training’s `GlobalArenaManager` backing; **≥384 MiB committed floor** |
| **Viewport color/depth RTs** | **GUI-only additive** | Pooled external images, `ceil64` buckets, ring of 3 × up to 4 slots |
| **Swapchain / UI / pipelines** | **GUI-only additive** | `kFramesInFlight=2`, RmlUi textures, compiled VkSplat pipelines |
| **Per-frame splat memcpy** | **Avoided** when tensors are Vulkan-external | Full copy fallback is **hard-refused** |
| **Training stall risk** | Real but designed as **GPU-side waits**, not CPU sync | `beginModelRead` / viewer-release timeline / `shared_mutex` |

Headless training never constructs the visualizer/window/Vulkan context (`Application::run` branches at `params->optimization.headless`).

---

## 1. Extra VRAM when GUI is active

### 1.1 Splat parameter storage — zero-copy when interop works

**Intent (explicit):** one CUDA VMM block backs all six splat tensors; the Vulkan viewer imports that block and reads trainer writes with **no per-frame copy**.

```19:22:src/core/include/core/splat_exportable_storage.hpp
    // Coalesced exportable storage for the six per-primitive splat tensors. One
    // CUDA VMM allocation backs all six; each tensor is a view at a fixed offset
    // into the same physical memory. The Vulkan viewer imports this single block
    // and reads the trainer's writes directly — no per-frame copy.
```

**Allocation site (GUI training start only):** requires a live viewer + Vulkan external-memory interop:

```102:148:src/visualizer/training/training_manager.cpp
    lfs::core::SplatTensorAllocator TrainerManager::createTrainingSplatTensorAllocator(
        ...
        if (viewer_ && viewer_->getWindowManager()) {
            vk_ctx = viewer_->getWindowManager()->getVulkanContext();
        }
        ...
        if (vulkan_interop_available && exportable_capacity > 0) {
            auto storage_result =
                lfs::core::SplatExportableStorage::create(exportable_capacity, sh_degree);
            ...
                    LOG_INFO("Training tensors share one CUDA-exportable VMM block "
                             "imported into Vulkan ... — zero-copy viewer interop",
```

**Capacity formula:** `exportable_capacity = max(max_cap, min_capacity)` — sized to **configured max Gaussians**, not current live count:

```108:113:src/visualizer/training/training_manager.cpp
        const std::size_t configured_capacity =
            params.optimization.max_cap > 0
                ? static_cast<std::size_t>(params.optimization.max_cap)
                : 0;
        const std::size_t exportable_capacity = std::max(configured_capacity, min_capacity);
```

**Byte layout** (`SplatExportableStorage::create`):

| Region | Shape / packing | Bytes |
|---|---|---|
| Means | `N×3` f32 | `N×3×4` |
| Scaling | `N×3` f32 | `N×3×4` |
| Rotation | `N×4` f32 | `N×4×4` |
| Opacity | `N×1` f32 | `N×1×4` |
| Sh0 | `N×1×3` f32 | `N×3×4` |
| ShN | swizzled rest | `sh_swizzled_float_count(N, rest)×4` |

Regions are **256-byte aligned** (`kRegionAlignment = 256`).  
ShN packing: `sh_rest_coefficients_for_degree(d) ∈ {0,3,8,15}`; swizzle uses `kShReorderSize=32` and packed float4 slots (`src/core/include/core/cuda/sh_layout.cuh:44–93`, `src/core/splat_exportable_storage.cpp:58–81`).

**Example totals (from layout formulas):**

| N | SH | Total ≈ |
|---|---:|---:|
| 1e6 | 0 | 53 MiB |
| 1e6 | 3 | **237 MiB** |
| 5e6 | 3 | **1183 MiB** |

**Does the viewer keep its own positions/SH?**  
On the intended path: **no second copy**. `prepareInputs` **borrows** VkBuffer sub-views from `VulkanExternalTensorStorage` and refuses full-input copy fallback:

```4713:4827:src/visualizer/rendering/vksplat_viewport_renderer.cpp
                buffers_.xyz_ws.deviceBuffer = makeBorrowedBufferView(... means ...);
                buffers_.sh0.deviceBuffer = makeBorrowedBufferView(... sh0 ...);
                ...
        return std::unexpected(std::format(
            "VkSplat refusing full input-copy fallback; model tensors must use Vulkan-external storage ({})",
            input_copy_reason));
```

**Exceptions (small, not full model):**
- Soft-delete mask → per-ring **opacity** interop buffer of size `N×4` bytes (`cuda_opacity_copies_[ring]`, ring size 3) (`vksplat_viewport_renderer.cpp:4658–4709`, `kInputRingSize = kFrameRingSize = 3` at `vksplat_viewport_renderer.hpp:601,634–635`).
- Overlay/selection masks → separate ring interop buffers (`cuda_overlays_`).

Vulkan import of the exportable block is a **second mapping of the same physical memory** (`importExternalBuffer` of CUDA export handle in `makeSplatExportableInteropAllocator`, `vulkan_external_tensor.cpp:271–300`) — not a second host-side tensor store. Profiler tracks one exportable size via `setExportableSplatBytes` (`exportable_storage.cpp:294–296`).

**Headless:** `createTrainingSplatTensorAllocator` is not used by the headless path in `Application::runHeadless` (direct `Trainer` + default CUDA tensors). Comment on `TrainerManager::splatExportableStorage`: `nullptr if running headless or fallback` (`training_manager.hpp:143–148`).

---

### 1.2 Shared rasterizer scratch (large; shared with training)

Viewer maintains a CUDA-exportable scratch arena, **imported into Vulkan** and installed as training’s external arena backing:

```931:932:src/visualizer/rendering/vksplat_viewport_renderer.cpp
        constexpr std::size_t kSharedScratchPageBytes = std::size_t{2} << 20;
        constexpr std::size_t kSharedScratchMinBytes = std::size_t{384} << 20;
```

```3192:3194:src/visualizer/rendering/vksplat_viewport_renderer.cpp
        const std::size_t target_bytes = alignUp(
            std::max(required_bytes + required_bytes / 8, kSharedScratchMinBytes),
            kSharedScratchPageBytes);
```

- **Floor:** **384 MiB** committed, 2 MiB page align.  
- **Required size:** `estimateSharedScratchBytes(...)` walks projection/sort/tile/pixel buffers (sort keys×2, pixel_state `4×H×W` floats, depths, etc.) (`vksplat_viewport_renderer.cpp:3112–3179`).  
- **Headless still needs a training arena**; GUI difference is exportable + Vulkan alias + **min floor**. Pre-start prime at training start: `ensureVksplatTrainingSharedScratchReady` (`training_manager.cpp:457–473`).

**Not pure overhead vs headless**, but the 384 MiB floor and early prime can retain more VRAM than a lazy headless arena.

---

### 1.3 Viewport render targets (color + depth) — pooling `#1497/#1562`

Commit `21775357` (“Pool viewport render targets (#1497) (#1562)”) introduces `OutputImagePool` and `ceil64` bucketing.

**Constants:**

```600:603:src/visualizer/rendering/vksplat_viewport_renderer.hpp
        static constexpr std::size_t kOutputSlotCount = 4;
        static constexpr std::size_t kFrameRingSize = 3;
        std::array<std::array<OutputImageSlot, kFrameRingSize>, kOutputSlotCount> output_slots_{};
        OutputImagePool output_pool_{};
```

Slots: `Main, SplitLeft, SplitRight, Preview` (`vksplat_viewport_renderer.hpp:94–98`).

**Formats / usage** (`ensureOutputImages`):

- Color: `VK_FORMAT_R8G8B8A8_UNORM`, 4 B/px  
- Depth: `VK_FORMAT_R32_SFLOAT`, 4 B/px  
- Usage: sampled + storage + transfer (`vksplat_viewport_renderer.cpp:4980–4998`)  
- Extent: `bucket = (ceil64(W), ceil64(H))` (`output_image_pool.hpp:21–24`, `vksplat_viewport_renderer.cpp:4943–4945`)

**Per-image logical size (device may pad allocation_size):**  
\[
\text{color} \approx W_b \cdot H_b \cdot 4,\quad
\text{depth} \approx W_b \cdot H_b \cdot 4,\quad
\text{pair} \approx 8 \cdot W_b \cdot H_b
\]

**Live ring (Main only, 3 slots):**  
\[
\approx 3 \times 8 \cdot W_b H_b
\]

| Viewport | Bucket | Color | Depth | Main×3 | Worst all 4 slots×3 |
|---|---|---:|---:|---:|---:|
| 1280×720 | 1280×768 | 3.75 MiB | 3.75 MiB | **22.5 MiB** | 90 MiB |
| 1920×1080 | 1920×1088 | 8.0 MiB | 8.0 MiB | **48 MiB** | 191 MiB |
| 2560×1440 | 2560×1472 | 14.4 MiB | 14.4 MiB | **86 MiB** | 345 MiB |
| 3840×2160 | 3840×2176 | 31.9 MiB | 31.9 MiB | **191 MiB** | 765 MiB |

**Pool retention:** retired/free images counted by `idleBytes()`; free entries trimmed after **`kIdleTrimTicks = 240`** drain ticks (`output_image_pool.hpp:58`, `output_image_pool.cpp:148–165`). Idle pool is **extra** until trim.

Logged as `outputs` + `output_pool_idle` in `logVramBreakdownIfChanged` (`vksplat_viewport_renderer.cpp:4857–4918`).

---

### 1.4 Viewport interop image path (fallback / split / depth blit)

Primary scene path for VkSplat publishes **external VkImage** handles (no CUDA surface copy):

```1765:1774:src/visualizer/visualizer_impl.cpp
                if (vulkan_frame.external_image != VK_NULL_HANDLE) {
                    interop.setExternalSceneImage(...);
```

`ViewportInteropService` scene policy sets `external_handle_early_out = true` → **ExternalSkip** when external image present (`viewport_interop_service.cpp:71–81`, `decideViewportInteropEarly` in `viewport_interop_service.hpp:56–63`).

**Slow path (tensor → external image)** still exists for scene-without-external, **split_right**, **depth_blit**:
- Format: scene/split `R8G8B8A8_UNORM`; depth `R32_SFLOAT` (`viewport_interop_service.cpp:71–103`)  
- One external image + timeline semaphore **per channel × `framesInFlight()`** (2) (`viewport_interop_service.cpp:285–288`, `kFramesInFlight = 2` in `vulkan_context.hpp:458`)  
- Copy: CUDA surface write on **non-blocking** upload stream (`copyTensorToSurface`, `cuda_vulkan_interop.cpp:627–705`; stream creation `CudaVulkanUploadStream::init` with `cudaStreamNonBlocking`, `cuda_vulkan_interop.cpp:172–186`)

Size formula (exact logical, device-rounded allocation):  
\[
\text{bytes} \approx W \cdot H \cdot 4 \times \text{slots} \times \text{channels in use}
\]

---

### 1.5 Swapchain, depth-stencil, UI

| Resource | Sizing | Source |
|---|---|---|
| Frames in flight | **2** | `vulkan_context.hpp:458` |
| Swapchain images | ≥ `min_image_count_ = 2` (typically 2–3) | `vulkan_context.hpp:443` |
| Depth-stencil | **one per frame-in-flight** | `createDepthStencilResources` path |
| Default window | 1280×720 | `application.cpp:632–636` |
| RmlUi textures | `W×H×4` RGBA8, VMA GPU_ONLY | `rmlui_vk_backend.cpp:1193–1248` |

**Font host assets (disk, not VRAM until atlas):** Inter ~0.4 MiB each; NotoSans JP/KR **~16 MiB each** under `src/visualizer/gui/assets/fonts/`. GPU atlas cost is dynamic (RmlUi `GenerateTexture`); CJK load is gated (`cjk_fonts_loaded_` in `rmlui_manager.hpp:195–197`).

**VRAM HUD** (`vram_hud_overlay.cpp`): pure RmlUi DOM over profiler snapshots — **no GPU allocations of its own**; diagnostic only.

---

### 1.6 Rough additive GUI VRAM budget (steady training, Main only)

Assuming exportable splat + shared scratch **replace** (not double) headless model/arena:

| Component | Typical order |
|---|---|
| Viewport Main ring (color+depth)×3 @ ~1080p | **~48 MiB** |
| Pool idle (after resizes) | 0–tens of MiB until trim |
| Swapchain + UI DS + Rml textures | **~30–80 MiB** (resolution/UI dependent) |
| Overlay / opacity-copy rings | O(N) or small fixed |
| Vulkan pipeline/VMA misc | hard to formula from code alone |
| Shared-scratch **floor excess** vs headless | **0–384 MiB** if headless arena would be smaller |

**Speculation (marked):** total GUI-only additive at 1080p training is often **~100–250 MiB** plus any shared-scratch floor/max_cap exportable overshoot; not “2× model size”.

---

## 2. Speed impact on training

### 2.1 Streams and sync design

| Mechanism | Behavior | File:line |
|---|---|---|
| GUI upload stream | **Non-blocking**; comment: never use default stream (would serialize with trainer) | `cuda_vulkan_interop.hpp:67–69`, `cuda_vulkan_interop.cpp:180–181` |
| Timeline waits/signals | Explicit stream required | `cuda_vulkan_interop.hpp:132–135` |
| Input ready | CUDA signal → Vulkan timeline wait (no `cudaStreamSynchronize` on happy path) | `vksplat_viewport_renderer.cpp:4790–4805`, comment at `hpp:657–659` |
| Viewer release fence | Trainer `cudaWaitExternalSemaphoresAsync` on **training stream** before step writes | `trainer.cpp:2252–2261` |
| Model read epoch | `beginModelRead` / `endModelRead` via CUDA events on reader stream | `trainer.cpp:2173–2199` |
| `waitForModelReaders` | Start of each train step | `trainer.cpp:3786–3789` |

**Per-frame CPU host syncs that can hurt:**
- Interop **slow path** `waitForCurrentFrameSlot` (comment: previously dominated `gui_render` at ~10–12 ms with FIF=1) (`viewport_interop_service.cpp:323–327, 368–379`) — **skipped** on ExternalSkip/CacheHit.
- Opacity ring full → `cudaEventSynchronize` (`trainer.cpp:2189–2194`).
- Shared-scratch contention: arena “busy” → frame retries with cache (`rendering_manager_vulkan.cpp:3521–3556`).

### 2.2 Locks on shared data

| Lock | Role |
|---|---|
| `model_access_mutex_` shared | Viewer holds during live train render (`rendering_manager_vulkan.cpp:1804–1806`); metrics render same (`trainer.cpp:3100`) |
| `model_access_mutex_` unique | Topology/refining writes (`trainer.cpp:5337–5338`) |
| `render_mutex_` | Coarser render exclusion (`trainer.hpp:167`, uses at refining) |
| `stream_sync_mutex_` | Serializes read/release fence bookkeeping (`trainer.cpp:2174, 2203, 2238`) |

Locks are CPU-side mutual exclusion for **setup**, not full-step GPU blocking; comment stresses GPU edges order real R/W (`trainer.hpp:170–175`).

### 2.3 How often the viewport re-renders during training

Actual throttle is **`ViewportFrameLifecycleService::handleTrainingRefresh`**, not `FramerateController::shouldSkipSceneRender`:

```64:78:src/visualizer/rendering/viewport_frame_lifecycle_service.cpp
    DirtyMask ViewportFrameLifecycleService::handleTrainingRefresh(...) {
        ...
        if (now - last_training_render_ <= interval) {
            return 0;
        }
        last_training_render_ = now;
        return DirtyFlag::SPLATS;
    }
```

Default interval: **`training_frame_refresh_time_sec = 1.0`** (`framerate_controller.hpp:19`).  
Called from frame path with that setting (`rendering_manager_vulkan.cpp:1631–1637`).  
Dirty=0 + cached output → cache HIT, no re-render (`rendering_manager_vulkan.cpp:1769–1772`).

**Dead code:** `FramerateController::shouldSkipSceneRender` is **never called** outside its definition (only `framerate_controller.hpp/cpp`). Adaptive skip / training interval in that function is unused.

### 2.4 Training pause for viewport resize

If training and Main output needs resize, frame may **request temporary training pause** until trainer is paused (`rendering_manager_vulkan.cpp:1727–1751`).

### 2.5 Contention summary

- **GPU:** viewer and training **can contend** on SM/memory; ordering is via events/timeline, not continuous host sync.  
- **CPU:** `shared_mutex` can delay refining steps while a frame holds the shared lock; cached frames intentionally **do not** join the model-read epoch (`rendering_manager_vulkan.cpp:1775–1777`).  
- **Effective scene raster rate while training:** ~**1 Hz** by default (plus camera/overlay dirty).

---

## 3. Headless: what GUI costs are still paid?

| Cost | Headless? |
|---|---|
| Window / Vulkan device / swapchain / RmlUi / fonts | **No** — `runHeadless` / TCP path, no `runGui` (`application.cpp:741–753, 264–348`) |
| `SplatExportableStorage` / Vulkan import of model | **No** (no viewer Vulkan context) |
| Shared-scratch exportable + Vulkan import | **No** |
| Viewport RT pool, interop targets, VkSplat pipelines | **No** |
| Training CUDA kernels, image pipeline, CUDA modules | **Yes** (core training) |
| `GlobalArenaManager` raster arena | **Yes** (training rasterizers) |
| Progress bar terminal UI | Headless uses non-GUI progress (`trainer.cpp:2934–2935, 2983`) |
| TrainerManager completion reaper thread | Only if using TrainerManager (TCP headless does); plain headless uses `Trainer` directly |

**Headless residual “GUI-related” costs: essentially none** beyond shared training code.  
**Speculation:** dynamic library bloat of linking visualizer into the process is build-dependent and not quantified in runtime code.

---

## 4. Interop path — exact mechanism and per-frame cost

### 4.1 Model parameters (trainer → viewer)

```
CUDA VMM (cuMemCreate) → export OS handle (fd/Win32)
  → Vulkan importExternalBuffer (alias same physical)
  → Tensor::from_external_owner(CUDA ptr) for training writes
  → prepareInputs: borrow VkBuffer views (no memcpy)
  → CUDA timeline signal after any needed small copies
  → Vulkan compute waits timeline, rasterizes
  → render-complete external timeline
  → trainer cudaWaitExternalSemaphoresAsync(borrow value)
```

Evidence: `exportable_storage.hpp:32–40`, `splat_exportable_storage.hpp:19–22`, `vulkan_external_tensor.cpp:271–300`, `vksplat_viewport_renderer.cpp:4713–4805`, `trainer.cpp:2202–2261`, `rendering_manager_vulkan.cpp:1807–1825`.

**Per-frame model cost (happy path):**  
- No positions/SH memcpy  
- Optional opacity rebuild if soft-deleted  
- Stream wait for splat input producers  
- One CUDA external semaphore signal  
- GPU wait on trainer for borrow value  

### 4.2 Viewport color (viewer → UI)

**Fast path (VkSplat):** external color image sampled directly; interop **ExternalSkip** — **zero** CUDA surface copy (`viewport_interop_service` policy + `visualizer_impl.cpp:1765–1774`).

**Slow path:**  
`createExternalImage` → export memory → `cudaImportExternalMemory` + surface object → `copyTensorToSurface` (kernel launch) → timeline signal → Vulkan layout transition to `SHADER_READ_ONLY` (`viewport_interop_service.cpp:408–570`).

### 4.3 Buffers vs images

| Class | Mechanism |
|---|---|
| Model / scratch / overlays | External **buffer** memory (`CudaVulkanBufferInterop`) |
| Interop scene fallback / split / depth | External **image** + surface (`CudaVulkanInterop`) |
| Upload stream | Dedicated non-blocking CUDA stream |

---

## 5. Ranked concrete reductions (with file:line)

### R1 — **Size exportable splat block to live capacity / grow** (High VRAM)  
Today: `max(max_cap, min_capacity)` at start (`training_manager.cpp:108–124`).  
A 5M max_cap SH3 reserves **~1.2 GiB** even early training.  
Use grow (`growExportableDeviceBlock` already exists, `exportable_storage.hpp:58–63`) or step capacity.  
**Impact:** can dominate GUI “extra” when max_cap ≫ live N.

### R2 — **Lower `render_scale` during training** (High VRAM + speed)  
Setting already exists: `0.25–1.0`, “does not affect training” (`rendering_types.hpp:183`; clamp `rendering_manager_vulkan.cpp:1547`).  
Interactive resize already drops scale (`kInteractiveResizeRenderScale`).  
**Impact:** color+depth pair scales with \(W_b H_b\); 0.5× linear ≈ **0.25×** RT bytes + less raster fill.

### R3 — **Train-time: single Main slot, smaller ring** (Medium–High VRAM)  
`kFrameRingSize=3`, `kOutputSlotCount=4` (`vksplat_viewport_renderer.hpp:600–601`).  
During training without split/preview: keep **1 slot × 1–2 ring**, skip depth if no depth-blit/meshes.  
**Impact:** ~3–12× less RT memory vs worst-case all slots.

### R4 — **Increase training refresh interval / idle when unfocused** (Medium speed)  
Default 1 s (`framerate_controller.hpp:19`).  
Raising to 2–5 s or pausing when window minimized cuts SM contention.  
Also: **wire or delete** dead `shouldSkipSceneRender` (`framerate_controller.cpp:38–76`) so adaptive skip is real.

### R5 — **Lower `kSharedScratchMinBytes` or prime to training-camera size only** (Medium VRAM)  
384 MiB floor (`vksplat_viewport_renderer.cpp:932, 3192–3194`); prime uses window/viewport size (`training_manager.cpp:463–470`).  
**Impact:** large cards always pin ≥384 MiB early in GUI training.

### R6 — **Aggressive pool trim** (Low–Medium VRAM after resize)  
`kIdleTrimTicks = 240` (`output_image_pool.hpp:58`).  
Lower during training or force `trimIdle` when training starts.  
Already: `trimOutputImagePoolIdle` after frames (`vksplat_viewport_renderer.cpp:1819–1820`).

### R7 — **On-demand rendering (already partial — extend)** (Medium speed)  
Dirty-flag cache HIT already (`rendering_manager_vulkan.cpp:1769–1772`).  
Ensure camera idle + training doesn’t still force overlays/UI full path; suspend scene when `scene_render_suspended` (`visualizer_impl.cpp:1757–1758`).

### R8 — **Lower-precision preview** (Medium VRAM if applied to depth/scratch)  
Color is already **RGBA8**. Depth is **R32F** for both vksplat outputs and depth-blit interop.  
Optional: omit depth RT when not compositing meshes; R16F depth if quality allows.

### R9 — **Decoupled preview rate vs UI FPS** (Medium speed)  
UI can present cached external image at display rate while splat compute stays at 1 Hz (largely already true).  
Avoid `force_input_upload` / full re-raster unless `DirtyFlag::SPLATS` (`rendering_manager_vulkan.cpp:3501–3512`).

### R10 — **Font / UI atlas hygiene** (Low VRAM)  
Keep CJK load lazy; avoid permanent large atlases. Host fonts: ~32 MiB JP+KR on disk (`assets/fonts/`).

### R11 — **VRAM HUD sampling cost** (Low CPU)  
`isDueForProcessSample` gates process samples (`vram_hud_overlay.hpp:44`); keep HUD off in perf-sensitive training demos.

### Already implemented (do not re-do)
- Zero-copy splat block + refuse full copy (`vksplat_viewport_renderer.cpp:4825–4827`)  
- External scene image (no surface copy) for VkSplat  
- Non-blocking upload stream (avoids default-stream training stall)  
- Output image pool + ceil64 (`#1497/#1562`)  
- Training refresh dirty throttle (1 s)  
- Shared scratch with training arena  

---

## 6. Key architecture diagram

```
HEADLESS                          GUI TRAINING
────────                          ────────────
CUDA model tensors                CUDA VMM ExportableBlock (max_cap × SH layout)
CUDA raster arena                 same block → Vulkan import (alias)
  (GlobalArenaManager)            SharedScratch ExportableBlock (≥384 MiB)
                                  → install as training arena backing
                                  → Vulkan import for VkSplat
                                  Viewport RTs: 3× (RGBA8+R32F) pooled
                                  Swapchain + RmlUi + pipelines
                                  Timeline: viewer borrow ↔ trainer wait
```

---

## 7. Instrumentation already available

- `VksplatViewportRenderer::logVramBreakdownIfChanged` → `vksplat.memory` PERF log (owned, pipeline, input views, opacity copies, overlays, outputs, pool idle, sort, shared scratch) (`vksplat_viewport_renderer.cpp:4830–4918`).  
- `VramProfiler` + GUI `VramHudOverlay` (exportable_splat_bytes, Vulkan blocks, CUDA pool).  
- Interop PERF timers (`interop.cache-HIT-skip`, `interop.copyTensorToSurface`, …) (`viewport_interop_service.cpp:355–357, 523–527`).

For optimization planning, prefer A/B with HUD/profiler: **headless train step time & process VRAM** vs **GUI train with scene suspended** vs **GUI train with 1 Hz preview**, then isolate RT scale and `max_cap` exportable size.

---

## 8. Speculative / not proven from code alone

- Exact swapchain image count and allocation padding per GPU (driver-dependent).  
- Absolute ms stall of `cudaWaitExternalSemaphoresAsync` under concurrent SM load.  
- Whether NVML process VRAM double-counts exportable aliases (profiler tracks exportable once; OS may differ).  
- End-to-end % iteration slowdown with GUI open (requires measurement).
