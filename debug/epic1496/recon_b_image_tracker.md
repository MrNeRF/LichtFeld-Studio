# Recon B — Generation-keyed image tracker + leak report (epic #1496 sub-task 1)

Read-only recon of `/home/paja/projects/gaussian-splatting-cuda` for issues **#1478** (generation-keyed `VulkanImageBarrierTracker`) and **#1488** (shutdown leak report of External*). No files in the repo were modified.

Sources: tree as of recon date; issue text also in-repo at `GITHUB_ISSUES.md` (§4 = #1478, §14 = #1488, §22 = #1496).

---

## 0. Issue claim vs tree today

`GITHUB_ISSUES.md:185` claims **“14 register/forget call sites across 5 files.”**

**Today’s tree (shared `VulkanContext::imageBarriers_` only):**

| Metric | Count |
|---|---|
| `registerImage` / `forgetImage` call expressions on shared tracker | **22** |
| Files | **4** (`vulkan_context.cpp`, `vksplat_viewport_renderer.cpp`, `point_cloud_vulkan_renderer.cpp`, `mesh_offscreen_renderer.cpp`) |
| Plus local (non-shared) trackers | +12 expressions in 3 more files (see §1.2) |

Treat “14/5” as stale/approximate. Spec should target **every** site below, not the old count.

`VulkanImageBarrierTracker::reset()` is **never called** from production code (only defined). Swapchain rebuild uses `clearSwapchainOnly()` instead.

---

## 1. `VulkanImageBarrierTracker` usage

### 1.1 Tracker instances

| Instance | Location | Shared? |
|---|---|---|
| `VulkanContext::image_barriers_` | `vulkan_context.hpp:432`, accessor `imageBarriers()` `:201` | **Yes — device-wide** |
| `VulkanSceneImageUploader::Impl::scene_image_barriers` | `vulkan_scene_image_uploader.cpp:28` | Local |
| `VulkanUiTexture` member `image_barriers` | `vulkan_ui_texture.cpp:176` (approx; used throughout) | Local |
| `RenderInterface_VK::m_image_barriers` | `rmlui_vk_backend.hpp:658` | Local |
| Stack `upload_barriers` in RmlUi texture upload | `rmlui_vk_backend.cpp:1353` | Local, ephemeral |
| Stack `scene_image_barriers` is the uploader’s member (above) | — | — |

Generation-keying **must** cover the shared tracker. Local trackers have the same handle-reuse hazard on their own maps; include them if the API change is global.

---

### 1.2 Every `registerImage` / `forgetImage` site

#### A. Shared tracker — `context.imageBarriers()` / `image_barriers_`

##### `src/visualizer/window/vulkan_context.cpp` (2 register; no forget — uses `clearSwapchainOnly`)

| Line | Method | API | Image | Create / destroy relative to register/forget |
|---|---|---|---|---|
| **3995** | `VulkanContext::createSwapchain` | `registerImage(swapchain_images_[i], COLOR, UNDEFINED, external=false)` | Driver-owned swapchain images from `vkGetSwapchainImagesKHR` | Registered immediately after create; on rebuild `destroySwapchain` (**4432**) calls `clearSwapchainOnly()` (not per-image forget). Images re-registered on next `createSwapchain`. |
| **4133** | `VulkanContext::createDepthStencilResources` | `registerImage(resource.image, depthStencilAspectMask(), UNDEFINED, external=false)` | VMA depth/stencil images per frame-in-flight | Registered after `vmaCreateImage` + view. Destroyed in `destroySwapchain` (views/images freed **~4387–4408**); cleared from tracker via `clearSwapchainOnly`. |

`clearSwapchainOnly` at **4432** inside `destroySwapchain` — erases all **non-external** map entries (swapchain + depth). External entries survive.

##### `src/visualizer/rendering/vksplat_viewport_renderer.cpp` (2 register, 6 forget)

Images: `OutputImageSlot::{image, depth_image}` = `VulkanContext::ExternalImage` (color `R8G8B8A8_UNORM`, depth `R32_SFLOAT`), created with `external=true`.

| Line | Method | API | Image | Create / destroy order |
|---|---|---|---|---|
| **1769** | `releaseOutputSlot` | `forgetImage(slot.image.image)` | Output color | **Forget → then** `destroyExternalImage` 1774 |
| **1772** | `releaseOutputSlot` | `forgetImage(slot.depth_image.image)` | Output depth | **Forget → then** `destroyExternalImage` 1775; slot zeroed; `output_generations_[i]=0` |
| **2022** | `reset` | `forgetImage(slot.image.image)` | All ring slots color | Same: forget then destroy 2027–2028 |
| **2025** | `reset` | `forgetImage(slot.depth_image.image)` | All ring slots depth | |
| **4906** | `ensureOutputImages` | `forgetImage(slot.image.image)` | Old color before resize/recreate | After optional `waitForSubmittedFrames` when replacing; **forget → destroy** 4911–4912 → create 4921/4928 → **register** |
| **4909** | `ensureOutputImages` | `forgetImage(slot.depth_image.image)` | Old depth | Same |
| **4967** | `ensureOutputImages` | `registerImage(..., UNDEFINED, external=true)` | New color | After successful `createExternalImage` |
| **4971** | `ensureOutputImages` | `registerImage(..., UNDEFINED, external=true)` | New depth | Then `++slot.generation` at **4976** |

Ring: `kOutputSlotCount=4` × `kFrameRingSize=3` (`vksplat_viewport_renderer.hpp:582–584`). Slots: `Main/SplitLeft/SplitRight/Preview`.

##### `src/visualizer/rendering/point_cloud_vulkan_renderer.cpp` (4 register, 2 forget)

Images: `OutputSlotResources::{color_image, depth_image}` — **VMA** images (not ExternalImage), registered as non-external (default `external=false`).

| Line | Method | API | Image | Create / destroy order |
|---|---|---|---|---|
| **1108** | `ensureOutputImages` (nested in `Impl`) | `registerImage(color, COLOR, UNDEFINED)` | New color | After create image+view; then `++slot.generation` **1118** |
| **1111** | `ensureOutputImages` | `registerImage(depth, DEPTH, UNDEFINED)` | New depth | |
| **1130** | `destroySlot` | `forgetImage(color)` | Color | **Forget →** destroy view + `vmaDestroyImage` |
| **1143** | `destroySlot` | `forgetImage(depth)` | Depth | Same |
| **1797** | restore lambda in render path | `registerImage(color, COLOR, previous_color_layout)` | Color | **Not lifecycle** — on failed `vkEndCommandBuffer` / submit, rewinds tracker layout to pre-transition value without destroying the VkImage |
| **1800** | same | `registerImage(depth, DEPTH, previous_depth_layout)` | Depth | Same |

##### `src/visualizer/rendering/mesh_offscreen_renderer.cpp` (4 register, 2 forget)

Images: `Impl::{color_image, depth_image}` VMA offscreen targets; registered **`external=true`** (odd for non-export images — means they **survive** `clearSwapchainOnly`).

| Line | Method | API | Image | Create / destroy order |
|---|---|---|---|---|
| **143** | `destroyTargets` | `forgetImage(color_image)` | Color | Forget then destroy views/images |
| **144** | `destroyTargets` | `forgetImage(depth_image)` | Depth | |
| **325–327** | `ensureTargets` | `registerImage(..., UNDEFINED, true)` color+depth | After create; `destroyTargets` first if size changed |
| **378–380** | restore lambda in `render` | `registerImage(..., previous_*_layout, true)` | On failure paths after transitions started | Layout rewind only |

---

#### B. Local trackers (not `VulkanContext::image_barriers_`)

##### `src/visualizer/gui/vulkan_ui_texture.cpp` (local `image_barriers`)

| Line | Context | API | Image |
|---|---|---|---|
| **510** | `transitionImageLayout` helper | `registerImage(image, COLOR, old_layout)` then transition | CPU or interop image; re-seeds layout each transition |
| **540** | `ensureImage` resize | `forgetImage` then `vmaDestroyImage` | CPU-upload path resize |
| **609** | `ensureImage` after create | `registerImage(..., UNDEFINED)` | New VMA CPU-upload image |
| **718** | `ensureInteropImage` after create+CUDA import | `registerImage(..., GENERAL)` | `ExternalImage` interop (`createExternalImage` 662, scope `"vulkan.ui_texture.interop_image"`) |
| **961** | `destroyImage` interop branch | `forgetImage` then `destroyExternalImage` | |
| **987** | `destroyImage` CPU branch | `forgetImage` then `vmaDestroyImage` | |

##### `src/visualizer/rendering/passes/vulkan_scene_image_uploader.cpp` (local `scene_image_barriers`)

| Line | Method | API | Image |
|---|---|---|---|
| **64** | `clearSceneImageBinding` | `forgetImage(scene_image)` | Clears handle without necessarily destroying (external borrow) |
| **209** | owned-image create path | `registerImage(..., UNDEFINED)` | After VMA create + view |
| **247** | `bindExternalSceneImage` | `registerImage(..., params.external_scene_image_layout)` | Borrows external image; keys also on `scene_image_external_generation` (**33**, compared **227**, set **238**) |

##### `src/visualizer/gui/rmlui/rmlui_vk_backend.cpp`

| Line | Method | API | Image |
|---|---|---|---|
| **1354** | texture upload lambda | stack tracker `registerImage` + transitions | One-shot upload; tracker discarded after submit |
| **2234** | `Destroy_Texture` | `m_image_barriers.forgetImage` | Before `vmaDestroyImage` |
| **2411** | `TransitionImageLayout` | `m_image_barriers.registerImage(old_layout)` then transition | Re-seeds layout every transition (same pattern as UI texture) |

---

### 1.3 `transitionImage` / `imageLayout` / `reset` / `clearSwapchainOnly`

#### `clearSwapchainOnly`

| Line | Caller |
|---|---|
| **4432** | `VulkanContext::destroySwapchain` |

#### `reset()`

- Implemented `vulkan_image_barrier_tracker.cpp:64–67` — clears both maps.
- **No production callers.**

#### `imageLayout` (reads)

| Line | Caller | Use |
|---|---|---|
| `mesh_offscreen_renderer.cpp:375–376` | `render` | Snapshot layouts before transitions for restore lambda |
| `vksplat_viewport_renderer.cpp:5163` | `composePixelState` / `transitionToProducer` | Detect “has previous contents” (`!= UNDEFINED`) to choose source scope |
| `vulkan_scene_image_uploader.cpp:226` | `bindExternalSceneImage` | Compare tracked layout to external param for early-out |

#### `transitionImage` — layout-only overload (inherits scopes from `layoutAccess` / last access)

Shared tracker (non-exhaustive but complete for shared instance):

| File:line | Enclosing | Image → layout |
|---|---|---|
| `vulkan_context.cpp:1213` | `beginFrame` | depth_stencil → `DEPTH_STENCIL_ATTACHMENT_OPTIMAL` |
| `vulkan_context.cpp:1345` | `endFrame` | swapchain → `PRESENT_SRC_KHR` |
| `vulkan_context.cpp:1656` | `captureAndEndActiveFrameRgba` | swapchain → `TRANSFER_SRC_OPTIMAL` |
| `point_cloud_vulkan_renderer.cpp:1703–1708` | render | color→COLOR_ATTACHMENT, depth→DEPTH_ATTACHMENT |
| `point_cloud_vulkan_renderer.cpp:1787–1792` | render | both → `SHADER_READ_ONLY_OPTIMAL` |
| `point_cloud_vulkan_renderer.cpp:2040,2059` | readback path | color → TRANSFER_SRC then restore |
| `mesh_offscreen_renderer.cpp:384–393,431–440` | `render` | attach then TRANSFER_SRC |
| `vksplat_viewport_renderer.cpp:5937,6094,6264` | readback restore paths | restore layout after transfer |

Local trackers: UI texture helper `:511`; RmlUi `:1355,:1370,:2412`.

#### `transitionImage` — **scope-taking overload** (hpp:49–57 / cpp:159–221)

Only **cross-queue / semaphore-edge** call sites pass explicit scopes. Callers:

| File:line | Enclosing | Scopes |
|---|---|---|
| **`vulkan_context.cpp:1193–1204`** | `beginFrame` | **src:** `COLOR_ATTACHMENT_OUTPUT` + `ACCESS_NONE` (match acquire wait stage); **dst:** `COLOR_ATTACHMENT_OUTPUT` + color attach R/W. Image: swapchain. |
| **`vksplat_viewport_renderer.cpp:396–401`** | anonymous `acquireOutputImageForReadback` | **src:** `external_producer{}` (empty — semaphore wait is TOP_OF_PIPE on submit); **dst:** `TRANSFER` + `TRANSFER_READ`. Image: output color for graphics-queue readback. |
| **`vksplat_viewport_renderer.cpp:5167–5172`** | `composePixelState` → lambda `transitionToProducer` | **src:** either empty `external_dependency` (cross-queue / no previous contents) or `fragment_sample` (`FRAGMENT_SHADER` + `SHADER_READ`); **dst:** producer (`TRANSFER_WRITE` for clear path, or `COMPUTE` + `SHADER_STORAGE_WRITE` for compose). |
| **`vksplat_viewport_renderer.cpp:5180–5186`** | `composePixelState` → lambda `releaseToFragmentSampling` | **src:** producer; **dst:** empty if `hasDedicatedComputeQueue()`, else `fragment_sample`. |

Internal: layout-only overload delegates to scope-taking at `vulkan_image_barrier_tracker.cpp:151–156`.

---

## 2. Existing generation counters near image recreation

### 2.1 Primary example (issue text)

**`OutputImageSlot::generation`** — `vksplat_viewport_renderer.hpp:570–581`

```cpp
struct OutputImageSlot {
    VulkanContext::ExternalImage image{};
    VulkanContext::ExternalImage depth_image{};
    ...
    std::uint64_t generation = 0;   // line 577
    std::uint64_t completion_value = 0;
};
```

| Who bumps | When | Tracker access? |
|---|---|---|
| `ensureOutputImages` **`++slot.generation` at :4976** | After successful create + **registerImage** of both images | **Yes** — same function has forget/register; generation is on the slot, available to all methods that hold `OutputImageSlot&` |
| `composePixelState` sets `output.generation = ++output_generations_[output_index]` at **:5229** and **:5368** | After successful compose/clear of pixel state (logical content generation, not VkImage recreate) | Has `output` slot; **overwrites** `slot.generation` with a **separate** counter `output_generations_[]` |
| `releaseOutputSlot` / `reset` | Zero slot via `slot = {}` (generation → 0); also `output_generations_[i]=0` on release | Yes |

**Important dual-use:**  
- Recreate path bumps at 4976 (resource identity).  
- Compose path reassigns from `output_generations_` (content identity for consumers).  
A generation-keyed tracker needs a **stable image-identity generation** that is not clobbered by compose, or compose must stop reusing the same field for content gens. Spec must resolve this.

Also: color and depth share one `slot.generation` today — fine if both recreated together (they always are in `ensureOutputImages`).

### 2.2 Other ad-hoc counters near image / interop recreation

| Counter | Definition | Bumped when | Tracker access? |
|---|---|---|---|
| `output_generations_[kOutputSlotCount]` | `vksplat_viewport_renderer.hpp:586` | Compose success 5229/5368; cleared on `releaseOutputSlot` 1779 | Same renderer; not passed to tracker |
| `point_cloud` `OutputSlotResources::generation` | nested struct ~line 462 in `.cpp` | `ensureOutputImages` **1118** after register; again **1904** after successful submit (content gen) | Yes inside `Impl` — same dual-use pattern |
| `VulkanSceneInteropTarget::generation` | `viewport_interop_service.cpp:26` | `++` on `destroy` (:45) and after successful upload (:566) | **Interop images are NOT registered** with barrier tracker (use `transitionImageLayoutImmediate` only) |
| `VulkanSceneInteropTarget::uploaded_source_generation` | `:31` | Set to `channel.source_generation` after upload | Skip re-upload; not a VkImage key |
| `Channel::source_generation` / `published_image_generation` | `:53`, `:60` | From renderer / publish path | Downstream consumers |
| `scene_image_external_generation` | `vulkan_scene_image_uploader.cpp:33` | From `params.external_scene_image_generation` on bind | **Local** tracker only; already compared for early-out with layout |
| `shared_scratch_.generation` | vksplat shared CUDA arena | Grow/import **3278, 3341, 3380** | Buffer, not image tracker |
| `artifact_generation_` | `viewport_artifact_service.cpp` | Capture invalidation | Host tensors, not VkImage |
| Scene/selection/LOD generations | various | Content versioning | Unrelated to barrier tracker |

### 2.3 Images **without** a generation field today

| Image class | Notes for #1478 |
|---|---|
| Swapchain images | No generation; recreated wholesale; `clearSwapchainOnly` + re-register. Could use swapchain-epoch counter on `VulkanContext`. |
| Depth/stencil frame images | Same as swapchain recreate path |
| Mesh offscreen color/depth | No generation on `Impl` |
| UI texture CPU/interop images | No generation; local tracker |
| RmlUi textures | No generation; local tracker |
| Viewport interop ExternalImages | Have `generation` but **not** barrier-tracker registered |

---

## 3. Tracker semantics (`vulkan_image_barrier_tracker.{hpp,cpp}`)

### 3.1 Data model

```cpp
// hpp:25-30, 60-61
struct ImageState {
    VkImageAspectFlags aspect_mask;
    VkImageLayout layout;              // last known layout
    VkPipelineStageFlags2 last_stage;  // last destination stage written
    VkAccessFlags2 last_access;        // last destination access written
};
std::unordered_map<VkImage, ImageState> images_;
std::unordered_set<VkImage> external_images_;
```

Key is **bare `VkImage`** — the #1478 bug.

### 3.2 `registerImage(image, aspect, layout, external=false)` (cpp:86–106)

- Null image → no-op.
- Overwrites `images_[image]` with layout + `layoutAccess(layout, Source)` as initial last_stage/access.
- If `external`: insert into `external_images_`; else erase from set.
- **Does not** emit a barrier.

### 3.3 `forgetImage` (cpp:79–84)

- Erase from both maps. Null no-op.
- **Does not** destroy the VkImage.

### 3.4 External vs non-external

- Only difference: membership in `external_images_`.
- `clearSwapchainOnly` (cpp:69–77): erase from `images_` **iff not** in `external_images_`. Does **not** clear the external set itself.
- Intent: swapchain resize must not drop layout tracking for long-lived interop/output images that stay alive across swapchain rebuild.
- Mesh offscreen marks VMA images external so they survive swapchain clear (they share the device-wide tracker).

### 3.5 `reset` (cpp:64–67)

Clear both containers. Unused by callers.

### 3.6 `layoutAccess(layout, direction)` (cpp:10–62)

Maps layout → stage/access for Source vs Destination:

| Layout | Source | Destination |
|---|---|---|
| COLOR_ATTACHMENT | COLOR_OUT + WRITE | COLOR_OUT + R/W |
| DEPTH(_STENCIL)_ATTACHMENT | early/late tests + WRITE | early/late + R/W |
| TRANSFER_SRC/DST | TRANSFER + READ/WRITE | same |
| SHADER_READ_ONLY | FRAGMENT_SHADER + SHADER_READ | same |
| GENERAL | ALL_COMMANDS + MEM R/W | same |
| PRESENT_SRC | COLOR_OUT + NONE (src); empty (dst) | — |
| UNDEFINED | empty | empty |
| default | ALL_COMMANDS + MEM R/W | same |

### 3.7 `transitionImage` layout-only (cpp:112–157)

1. Require non-null cmd/image; debug-assert image is tracked.
2. If `state.layout == new_layout` → **no-op** (no barrier, last_stage unchanged).
3. Source scope: prefer `state.last_stage/access` if either non-NONE; else `layoutAccess(old, Source)`.
4. Dest scope: `layoutAccess(new, Destination)`.
5. Delegate to scope-taking overload.

### 3.8 `transitionImage` with scopes (cpp:159–221)

1. Same null/tracked checks; same early-out if layout already matches.
2. Record single `VkImageMemoryBarrier2` (queue family IGNORED both sides — no ownership transfer).
3. `vkCmdPipelineBarrier2`.
4. Update state: aspect, layout, **last_stage = destination.stage**, **last_access = destination.access**.

### 3.9 Write-after-read sharp edge (hpp:49–51)

> Layouts do not identify the queue or shader stage that produced/consumes an image. Cross-queue users must provide the scopes represented by the submission's semaphore edges instead of inheriting a graphics-only scope.

**What goes wrong with layout-only:**

- After compose, last access becomes `FRAGMENT_SHADER|SHADER_READ` (or empty on async compute release).
- Next producer transition using layout-only would take that as source — wrong queue/stage for compute, and **only one** last reader is stored (no multi-reader union).
- Epic sub-task 4 explicitly wants reader accumulation; today only a single `last_*` pair exists.

**Who uses the scope-taking overload today:** see §1.3 table — only `beginFrame` (swapchain acquire chain) and `VksplatViewportRenderer` compose/readback cross-queue paths.

### 3.10 Related: `transitionImageLayoutImmediate` does **not** use the tracker

`vulkan_context.cpp:3517+` builds barriers via `layoutAccess` only, with caller-supplied `old_layout`/`new_layout`. Used by viewport interop + UI interop. Tracker state for those images (if any) is **not** updated by this path — interop targets intentionally stay off-tracker.

---

## 4. Leak report (#1488)

### 4.1 Types (`vulkan_context.hpp:105–131`)

```cpp
struct ExternalImage {
    VkImage image;
    VkDeviceMemory memory;
    VkImageView view;
    VkExtent2D extent;
    VkFormat format;
    VkDeviceSize allocation_size;
    std::string diagnostic_scope;   // line 112  ← issue cites ~107,117 (scope fields)
    std::string diagnostic_label;
    ExternalNativeHandle native_handle;
};

struct ExternalBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size, allocation_size;
    std::string diagnostic_scope;   // line 122
    std::string diagnostic_label;
    ExternalNativeHandle native_handle;
};

struct ExternalSemaphore {
    VkSemaphore semaphore;
    std::uint64_t initial_value;
    ExternalNativeHandle native_handle;
    // NO diagnostic_scope / diagnostic_label today
};
```

Issue #1488 asks to count all three per `diagnostic_scope`. **Semaphore has no scope field** — leak report must add one (or use a fixed scope like `"vulkan.external.semaphore"`).

### 4.2 Lifecycle API (`vulkan_context.hpp` / `.cpp`)

| API | Definition | Behavior relevant to leak report |
|---|---|---|
| `createExternalImage` | cpp:2820–3031 | Sets scope/label; `recordCurrentVulkanBytes(scope, label, size)`; exports OS handle |
| `destroyExternalImage` | cpp:3033–3050 | `recordCurrentVulkanBytes(..., 0)`; destroy view/image/memory; close handle; zero struct |
| `releaseExternalImageNativeHandle` | cpp:3052–3056 | Transfers OS handle ownership (CUDA import); object remains live |
| `createExternalBuffer` | cpp:3058+ | Same pattern as image |
| `destroyExternalBuffer` | cpp:3350+ | Bytes → 0; destroy buffer/memory |
| `importExternalBuffer` | cpp:3195+ | Import foreign memory; default scope `"vulkan.external.imported_buffer"` |
| `createExternalTimelineSemaphore` | cpp:3372+ | **No VRAM profiler / no scope** |
| `destroyExternalSemaphore` | cpp:3458–3470 | Destroy VkSemaphore; scrub timeline trackers; close handle |

**There is no live-object registry** — only VRAM **byte** gauges via `VramProfiler::recordCurrentBytes` for image/buffer. Semaphores are invisible to that path. A true object-count leak report needs create/destroy refcounts (or a set of live labels) inside `VulkanContext`.

### 4.3 Create/destroy call sites (owners)

| Owner | Create | Destroy | Typical scope string |
|---|---|---|---|
| `VksplatViewportRenderer` | output images 4921/4928; many buffers; timelines 1715, 4227 | `releaseOutputSlot` / `reset` / ensure path | `"vulkan.vksplat.output_image"`, buffer scopes via helpers, `"shared.scratch"` import |
| `ViewportInteropService` | `createExternalImage` 415 + semaphore 422 | `VulkanSceneInteropTarget::destroy` 33–46 | `"vulkan.gui.interop_image"` |
| `vulkan_ui_texture` | interop image+sem 662–667 | `destroyImage` 965–966 | `"vulkan.ui_texture.interop_image"` |
| `vulkan_external_tensor` | buffer 147; import 275 | destructor / fail paths | `"vulkan.external_tensor.buffer"`, `".alias"` |
| Point cloud / mesh offscreen | **not** External* (VMA) | n/a | n/a for #1488 counters |

### 4.4 Device teardown hook (right place for #1488)

```
VulkanContext::~VulkanContext()  →  shutdown()   // cpp:471–473, 586–672
WindowManager drops unique_ptr   →  vulkan_context_.reset()  // window_manager.cpp:369 etc.
```

`shutdown()` sequence (cpp:586+):

1. Latch `context_shutdown_started_`
2. `vkDeviceWaitIdle`
3. Destroy frame fences
4. **`destroySwapchain()`** (includes `clearSwapchainOnly`)
5. Drain/destroy immediate pool, command pools
6. Pipeline cache, VMA allocator
7. **`vkDestroyDevice`**
8. Surface, instance

**Recommended leak-report hook:** start of `shutdown()` after idle (or just before `vkDestroyDevice`), logging any External* still live **by diagnostic_scope**.  

Caveat: ownership is **external** to `VulkanContext` — consumers must destroy before context shutdown. Order today relies on GUI/visualizer teardown calling renderer/`reset` before `WindowManager` destroys the context (`gui_manager.cpp` shutdown ~3740+ tears down Rml/viewport before window death). Report makes failed ordering **visible**.

Also note: if counters live on `VulkanContext`, create/destroy paths are the single instrumentation point — no need to touch every owner.

---

## 5. Where visualizer creates/destroys tracked images; resize behavior

### 5.1 On shared tracker

| Image class | Create | Destroy | Recreates on window resize? | Recreates on viewport/content resize? |
|---|---|---|---|---|
| **Swapchain color** | `createSwapchain` (~3791, register 3995) | `destroySwapchain` + clearSwapchainOnly | **Yes** via `recreateSwapchain` (4520) driven by `notifyFramebufferResized` / out-of-date | N/A |
| **Swapchain depth/stencil** | `createDepthStencilResources` (register 4133) | with swapchain | **Yes** with swapchain | N/A |
| **VkSplat output color/depth** (ExternalImage ring) | `ensureOutputImages` 4850+ | forget+destroy same fn / `releaseOutputSlot` / `reset` | **Indirectly** — viewport size change calls ensure with new size (render ~7815, overlay ~6921); `nextOutputImagesNeedResize` 4418 | **Yes** when `slot.size != size` |
| **Point-cloud output color/depth** (VMA) | `ensureOutputImages` ~960 | `destroySlot` | Only if render size changes | **Yes** on size change |
| **Mesh offscreen color/depth** (VMA, marked external) | `ensureTargets` | `destroyTargets` | Only if requested w/h changes | **Yes** |

### 5.2 Not on shared tracker (but create ExternalImage / local track)

| Image class | Create | Destroy | Resize |
|---|---|---|---|
| **Viewport interop** Scene / SplitRight / DepthBlit | `viewport_interop_service.cpp:409–470` ExternalImage + semaphore | `target->destroy` | Recreate if size changes or interop invalid (**398–400**) |
| **GT compare** | Not a dedicated ExternalImage class — GT is a **CUDA tensor** uploaded via interop/split path (`rendering_manager_vulkan.cpp` ~2227+), often through Preview/SplitRight **OutputSlot** or interop channel | Same as those pipelines | Follows preview/split size |
| **UI texture CPU** | VMA in `ensureImage` | `destroyImage` | Yes on dimension change |
| **UI texture CUDA interop** | `createExternalImage` | `destroyImage` | Yes on dimension change |
| **RmlUi textures** | CreateTexture path | `Destroy_Texture` | Per-texture |
| **Scene image uploader owned image** | VMA create path | `destroySceneImage` | Size change |
| **Scene image external bind** | Borrow only | `clearSceneImageBinding` | When external handle/generation/layout changes |

### 5.3 OutputSlot map (VkSplat)

| Slot | Use |
|---|---|
| Main | Primary splat viewport |
| SplitLeft / SplitRight | Split-view panes (GT compare often drives right/preview tensors) |
| Preview | Preview / GT-related paths in `rendering_manager_*` |

Point-cloud has Main/SplitLeft/SplitRight only (no Preview).

---

## 6. Spec implications (recon only — not design decisions)

### #1478 generation-keyed tracker

1. Change map key from `VkImage` → `{VkImage, uint64_t generation}` (and same for `external_images_`).
2. **All** register/forget/transition/imageLayout signatures need a generation (or a handle wrapper).
3. Existing counters are **inconsistent**:
   - Some mean resource identity (bump on recreate).
   - Some mean content identity (compose overwrites `slot.generation`).
   - Swapchain/mesh/UI lack counters.
4. Dual-bump on point-cloud and vksplat will break keying if content gen advances without re-register — **must split fields** (e.g. `image_generation` vs `content_generation`).
5. Restore-on-failure `registerImage(previous_layout)` sites (point_cloud 1797, mesh 378) must re-register with the **same** generation.
6. `clearSwapchainOnly` must drop only non-external keys (by generation epoch or external flag).
7. Local trackers: either migrate to shared API with generation, or leave local maps keyed the same way for consistency.

### #1488 leak report

1. Instrument `createExternal*` / `importExternalBuffer` / `destroyExternal*` with live counts per scope (and add scope to `ExternalSemaphore`).
2. Log non-zero counts in `VulkanContext::shutdown()` after idle, before device destroy.
3. VRAM profiler bytes ≠ object counts; do not rely on bytes alone (label reuse, zero-size, semaphores).

### Cross-links to epic sub-tasks 2–4

- Scope-taking overload already used for async-compute output path; sub-task 4 (reader accumulation) is still open.
- Immediate layout transitions bypass the tracker entirely — generation keys on tracker do not protect interop-only images unless those paths also track.

---

## 7. Quick inventory tables for implementers

### Shared-tracker register/forget (22 expressions, 4 files)

```
vulkan_context.cpp:3995 register (swapchain loop)
vulkan_context.cpp:4133 register (depth loop)
vksplat_viewport_renderer.cpp:1769,1772 forget (releaseOutputSlot)
vksplat_viewport_renderer.cpp:2022,2025 forget (reset)
vksplat_viewport_renderer.cpp:4906,4909 forget (ensureOutputImages)
vksplat_viewport_renderer.cpp:4967,4971 register (ensureOutputImages)
point_cloud_vulkan_renderer.cpp:1108,1111 register (ensureOutputImages)
point_cloud_vulkan_renderer.cpp:1130,1143 forget (destroySlot)
point_cloud_vulkan_renderer.cpp:1797,1800 register (layout restore)
mesh_offscreen_renderer.cpp:143,144 forget (destroyTargets)
mesh_offscreen_renderer.cpp:325,327 register (ensureTargets)
mesh_offscreen_renderer.cpp:378,380 register (layout restore)
```

### Scope-taking `transitionImage` (production)

```
vulkan_context.cpp:1193 beginFrame (swapchain acquire chain)
vksplat_viewport_renderer.cpp:396 acquireOutputImageForReadback
vksplat_viewport_renderer.cpp:5167 transitionToProducer (composePixelState)
vksplat_viewport_renderer.cpp:5180 releaseToFragmentSampling (composePixelState)
```

### Teardown hook

```
VulkanContext::shutdown()  @ vulkan_context.cpp:586
  ← insert live External* report after vkDeviceWaitIdle, before vkDestroyDevice
```

---

*End of recon B.*
