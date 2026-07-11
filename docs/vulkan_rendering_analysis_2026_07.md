# Vulkan rendering analysis and overnight implementation plan

Date: 2026-07-10

Audited checkout: `master` / `origin/master` at `bd9dcf76c1a4`

Scope: `src/rendering/**`, `src/visualizer/**`, and the CUDA/Vulkan interop bridge

Out of scope for edits: `src/core/tensor`, `src/io`, `src/fusion`, `src/training`

## Audit basis and verdict

This is a static, current-checkout audit plus a clean build and focused CPU/gtest baseline. The GUI was not launched and no training was run. Prior profiling numbers are labeled as prior measurements; current code-path existence was re-verified, but runtime magnitudes must be re-measured in the implementation phases.

Baseline validation:

- `cmake --build build -j8` passes.
- The focused rendering/service suite ran 55 tests: 52 passed and three pre-existing `ViewportFrameLifecycleServiceTest` cases failed on unchanged master (`ResizeActiveDefersFullRefreshUntilDebounceCompletes`, `PassiveWindowResizeDefersFullRefreshUntilDebounceCompletes`, and `ExplicitRefreshDeferralCompletesAfterDebounce`). The common mismatch was dirty mask `16` versus expected `9`, plus incomplete deferral state in the explicit-refresh case. These are the before-state, not Vulkan changes to absorb into this effort.
- No direct Vulkan runtime integration test exists in this filtered suite. Debug validation-layer runs, fixed-camera captures, and Nsight/Vulkan timestamp captures therefore remain mandatory implementation gates.
- Tests are intentionally not added or modified in this plan. Existing focused tests, validation layers, fault-path checks, image comparisons, and profiling are the validation surface.

Staff verdict:

1. The training rasterizer is CUDA, not this Vulkan compute rasterizer. The Vulkan path still affects training materially because it reads the live CUDA model, borrows the training rasterizer arena, and launches competing work on the same physical GPU.
2. The hottest actionable viewport defect remains the synchronous cumsum-tail read in the legacy/live-training GS path. It splits the command batch and busy-spins the CPU. The checkout already contains almost all machinery needed for a GPU count plus indirect sort; using it in the legacy path is the highest-confidence large performance win.
3. The most serious correctness defects are not “sync2 with submit1.” They are release-mode Vulkan errors being compiled out, an early return that skips shared-arena/trainer release publication, mutable descriptors/UBOs reused across draws and in-flight frames, and reset-before-submit fences left permanently unsignaled when submission fails.
4. Seven of the eight named `vulkan-fixes` changes are already present semantically in master. The release-only NULL guard landed narrowly, but the surrounding `_THROW_ERROR` infrastructure is still disabled in release and can publish an unsignaled timeline value after a failed submit. None of the seven stale commits applies cleanly to this checkout; semantic reimplementation is safer than cherry-picking.
5. The current device creation path enables five unused capability families. Delete them rather than building speculative bindless/shader-object/cooperative-matrix paths tonight.

Severity in the issue table means: **Critical** = plausible hang, memory corruption, device loss, or wrong normal-path output; **High** = serious correctness/reliability defect or dominant hot path; **Medium** = bounded performance/robustness issue; **Low** = maintainability/dead code with small immediate runtime impact.

## 1. Architecture map

### 1.1 Execution and ownership topology

```text
CPU/UI thread
  RenderingManager::renderVulkanFrame()
    |-- builds SceneRenderState / dirty decision
    |-- installs trainer <-> viewer model-read handshake
    |-- VksplatViewportRenderer::render()
    |     |-- CUDA non-blocking render_stream_
    |     |     packs/copies inputs and signals upload-ready timeline
    |     |-- VulkanGSRenderer command batch
    |           waits upload-ready timeline
    |           projection -> depth sort -> cumsum -> tile sort -> raster -> compose
    |           submits on dedicated compute-only queue when available
    |           signals render-complete timeline
    |-- returns external color/depth image + completion value
    |
    +-> GUI frame
          VulkanContext::beginFrame() [2 frame slots]
          graphics submit waits Vksplat render-complete timeline
          viewport pass graph -> RmlUi -> present

CUDA training stream (blocking/default stream semantics)
  fast_rasterize_forward() or gsplat_rasterize_forward()
  records params-ready event
  waits reader-done events and Vksplat render-complete timeline before next writes

Shared objects
  CUDA-exported model buffers <-> Vulkan imported buffers
  training rasterizer arena <-> Vulkan shared scratch import
  CUDA/Vulkan external timeline semaphores in both directions
```

### 1.2 Grounded component map

| Component | Current role and topology | Current evidence |
|---|---|---|
| Training rasterizer | Training selects `fast_rasterize_forward` for normal 3DGS and `gsplat_rasterize_forward` for GUT. There is no Vulkan raster call in the training step; the Vulkan rasterizer library is linked into the visualizer target. | `src/training/trainer.cpp:3274`, `src/training/trainer.cpp:3441-3499`, `src/visualizer/CMakeLists.txt:251-259`, `src/visualizer/CMakeLists.txt:307-312` |
| Training CUDA streams | `training_stream_` is created with blocking/default stream flags; callback and metrics streams are non-blocking. | `src/training/trainer.cpp:1783-1791`, `src/training/trainer.cpp:1805-1813` |
| Model WAR handshake | Viewer waits the trainer's params-ready event; trainer consumes reader-done events and waits the viewer's external completion timeline before overwriting borrowed data. | `src/training/trainer.cpp:1934-1957`, `src/training/trainer.cpp:1983-2013` |
| Viewer CUDA lanes | VkSplat owns a non-blocking `render_stream_`; live input work signals a CUDA timeline, and the Vulkan compute batch waits it at compute stage. The separate GUI texture and scene-image upload paths currently enqueue on the legacy NULL stream, which synchronizes against the blocking training stream. | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:1513-1517`, `:4462-4493`; `src/visualizer/gui/vulkan_ui_texture.cpp:609-615`; `src/visualizer/gui/gui_manager.cpp:4193-4205` |
| Interop copy bridge | Imported Vulkan images are CUDA surfaces; 16x16 CUDA kernels convert contiguous HWC/CHW UInt8/Float32 tensors to RGBA8 or copy channel 0 to R32F. Imported external buffers use device-to-device async copies. | `src/rendering/cuda_vulkan_interop.cu:9-103`, `src/rendering/cuda_vulkan_interop.cu:107-159`, `src/rendering/cuda_vulkan_interop.cpp:461-506`, `src/rendering/cuda_vulkan_interop.cpp:746-786` |
| Device selection | Vulkan 1.3 is required and the preferred discrete Vulkan device is UUID-matched to CUDA device 0. | `src/visualizer/window/vulkan_context.cpp:1462-1536` |
| Vulkan queues | One graphics+compute queue, one present queue, and one compute-only queue when the hardware exposes it; all are requested at priority 1.0. | `src/visualizer/window/vulkan_context.cpp:1392-1423`, `src/visualizer/window/vulkan_context.cpp:1553-1580`, `src/visualizer/window/vulkan_context.cpp:1929-1939` |
| GS compute submit | VkSplat selects the compute-only queue when available and reuses the application pipeline cache. | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:3940-3968` |
| Compute-to-graphics dependency | Each GS batch signals `render_complete_timeline_`; the GUI graphics frame waits that value before sampling the external image. | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:7207-7224`, `src/visualizer/gui/gui_manager.cpp:6406-6411` |
| Graphics frame | Two frames in flight. One graphics command buffer records the declarative viewport pass graph and RmlUi, then submits and presents. | `src/visualizer/window/vulkan_context.hpp:334-365`, `src/visualizer/gui/gui_manager.cpp:6398-6456` |
| Pass graph | Ordered prepare/scene/overlay/UI phases; active passes get CPU NVTX ranges. The graph records into the current graphics frame. | `src/visualizer/rendering/passes/viewport_pass_graph.hpp:23-62`, `src/visualizer/rendering/passes/viewport_pass_graph.hpp:71-98`, `src/visualizer/rendering/passes/viewport_pass_graph.hpp:146-160`, `src/visualizer/rendering/passes/vulkan_viewport_pass.cpp:1883-1922` |
| Cross-queue resources | External images and buffers use concurrent sharing between graphics and the compute-only family, avoiding ownership-transfer barriers. | `src/visualizer/window/vulkan_context.cpp:2295-2309`, `src/visualizer/window/vulkan_context.cpp:2463-2477` |
| VkSplat output buffering | Four logical outputs (`Main`, two split panels, `Preview`), each with a three-entry color+depth image ring. | `src/visualizer/rendering/vksplat_viewport_renderer.hpp:93-98`, `src/visualizer/rendering/vksplat_viewport_renderer.hpp:568-580` |

The dedicated Vulkan compute queue is not a separate GPU. It can overlap graphics scheduling, but its kernels still compete with CUDA training for SM issue slots, registers/shared memory, L2, DRAM bandwidth, copy engines, and VRAM. The timeline handshakes provide necessary ordering; they do not isolate throughput.

### 1.3 Sync model

- CUDA -> Vulkan: input packing/copy is enqueued on `render_stream_`, then CUDA signals an upload timeline; the compute batch waits it at `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT` (`vksplat_viewport_renderer.cpp:4479-4493`).
- Vulkan -> graphics: the compute batch signals a monotonically increasing render-complete value, and the graphics frame waits it before sampling (`vksplat_viewport_renderer.cpp:7207-7224`, `gui_manager.cpp:6406-6411`).
- Vulkan -> CUDA training: after a successful submit, the renderer publishes the same completion semaphore/value to the shared arena and trainer (`vksplat_viewport_renderer.cpp:7541-7549`); the training stream waits it before writes (`trainer.cpp:2006-2013`).
- Synchronization2 is used for barriers, while every submission remains `vkQueueSubmit`; the repository has zero `vkQueueSubmit2` calls. This is legal Vulkan. Migration to submit2 would make stage/access semantics more coherent but is not itself a correctness repair.

## 2. Re-audit of `vulkan-fixes`

`git log --reverse master..vulkan-fixes` contains seven commits based at merge base `ef097e248f94`. The first commit contains two of the eight named fixes. `git apply --check` against current master fails for all seven, as expected after five weeks of overlapping changes.

| Named fix | Branch commit | Current-master status | Current evidence | Tonight action |
|---|---|---|---|---|
| External-buffer dedicated-allocation import flag | `b40384364` | **Already fixed.** Allocation requirements drive dedicated VMA allocation and CUDA import receives the matching flag. | `src/visualizer/window/vulkan_context.cpp:2493-2505`; `src/visualizer/rendering/vulkan_external_tensor.cpp:126-132` | Do not cherry-pick. Add to interop smoke gate. |
| `oldSwapchain` on recreation | `b40384364` | **Already fixed and more robust.** The old handle stays live through dependent teardown/new creation, then is destroyed; failed creation restores it. | `src/visualizer/window/vulkan_context.cpp:3613-3677` | Do not cherry-pick. Keep current failure rollback. |
| Point-cloud color cache key | `0825b738f` | **Already fixed.** Cache keys use the source tensor pointer, dtype, count, and revision, not a converted temporary. | `src/visualizer/rendering/point_cloud_vulkan_renderer.cpp:868-896` | Do not cherry-pick. |
| Reachability-uniform cumsum barriers | `81a81470c` | **Already fixed.** All launched threads reach workgroup barriers; work is gated by `active`. | `src/rendering/rasterizer/vulkan/shader/src/slang/cumsum.slang:89-137`, `src/rendering/rasterizer/vulkan/shader/src/slang/cumsum.slang:150-186` | Do not cherry-pick. Retain shader validation in P3. |
| Release NULL-buffer guards | `01fd189ff` | **Partially fixed.** `_CHECK_FATAL` is always on and guards dispatch buffers, but `_THROW_ERROR` remains a no-op under `ENABLE_ASSERTION=0`; creation, mapping, batch, limit, and submit failures still continue in release. | `src/rendering/rasterizer/vulkan/src/config.h:5`, `src/rendering/rasterizer/vulkan/src/config.h:47-67`; guarded sites at `gs_pipeline.cpp:898`, `gs_pipeline.cpp:962-974` | **P0 semantic reimplementation.** Do not cherry-pick the stale diff. |
| Persisted GS pipeline-cache reuse | `01fd189ff`, `3988ae813` | **Already fixed.** Disk cache is loaded/saved and the app cache is passed into GS initialization/pipeline creation. | `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:124-147`, `src/visualizer/rendering/vksplat_viewport_renderer.cpp:3958-3968`, `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:874-875` | Do not cherry-pick. |
| Persistent readback pool + fence | `3988ae813` | **Already fixed for command resources.** Pool, command buffer, and fence are persistent. Staging buffers are still allocated per call. | `src/visualizer/rendering/vksplat_viewport_renderer.hpp:460-473`, `src/visualizer/rendering/vksplat_viewport_renderer.cpp:4940-4976` | Do not cherry-pick. P4 may finish staging reuse. |
| Shadow dirty-gating | `e7669ada5` | **Already fixed.** Shadow renders are keyed/gated and skipped when unchanged. | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:1548-1583` | Do not cherry-pick. Preserve in P2. |

The unlabelled `bb9b97fde` commit mixes Python, mesh normals, point-cloud rendering, viewport behavior, shaders, and tests. Its relevant point-cloud/viewport effects have since been superseded. It is not a coherent cherry-pick candidate and includes out-of-scope files.

## 3. Issues inventory

| ID | Severity | Area | File:line | One-line description | Evidence / failure mode |
|---|---|---|---|---|---|
| COR-01 | Critical | GS error handling | `src/rendering/rasterizer/vulkan/src/config.h:5`, `:56-67` | `_THROW_ERROR` compiles to nothing in release. | Failed Vulkan creation, begin/end, map, bounds, and submit checks continue with invalid state. Only the narrow `_CHECK_FATAL` sites survive. |
| COR-02 | Critical | GS batch state | `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:539-649` | A failed queue submit can still be treated as a pending timeline signal. | Submit failure calls the release-no-op `_THROW_ERROR`, then clears waits, closes the batch, and stores `pending_signal`; later code/training can wait forever on a value never signaled. |
| COR-03 | High | RAII/exception safety | `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:118-121`, `src/rendering/rasterizer/vulkan/src/gs_pipeline.h:270-337` | Pipeline, `DeviceGuard`, and `HostGuard` destructors perform fallible submit/begin/end work. | A second exception during stack unwinding terminates the process; batch state has no explicit cancel transaction. |
| COR-04 | Critical | Shared scratch handoff | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:7400-7405`, `:7541-7549` | Insufficient shared sort capacity returns from inside the active batch before publishing the arena/trainer release. | The guard still submits at scope exit, but `noteVulkanRelease` and the live-submit callback are skipped, allowing scratch reuse/overwrite while Vulkan reads it. |
| COR-05 | High | CUDA timeline | `src/rendering/cuda_vulkan_interop.cpp:398-402`, `:509-521`, `:572-575`, `:606-618` | Timeline monotonicity is assert-only and bookkeeping advances before CUDA enqueue succeeds. | Release builds accept non-increasing values; a failed signal poisons `last_signaled_` and can make subsequent handshakes inconsistent. |
| COR-06 | High | VkSplat initialization | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:3940-4118` | `ensureInitialized` creates the renderer and multiple exported/imported timelines incrementally without transactional rollback. | A mid-sequence failure returns with earlier live handles while `initialized_` is false; retry can overwrite/leak partial state. |
| COR-07 | High | Main-frame submit | `src/visualizer/window/vulkan_context.cpp:787-807`, `:1018-1049` | The frame fence is reset before submit and left unsignaled if `vkQueueSubmit` fails. | Reusing that frame slot blocks forever in `beginFrame`; the acquired binary semaphore also needs explicit abort/replacement semantics. |
| COR-08 | High | Transfer submits | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:5188-5202`, `:5374-5389`, `:5575-5590`; `vulkan_depth_blit_pass.cpp:518-566`; `vulkan_split_view_pass.cpp:621-669` | Persistent transfer fences have the same reset-before-success poison path. | Any command reset/begin/end/submit failure after the fence reset can make the next reuse or teardown wait forever. Point-cloud transfer state repeats the pattern. |
| COR-09 | Critical | Mesh descriptors/UBO | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:118-121`, `:1355-1386`, `:1618-1645` | One light UBO and descriptor set are rewritten for every mesh, then rebound for every draw. | All recorded draws can observe the final mesh's light/shadow state; descriptor update while bound/pending violates descriptor lifetime rules. Wrong multi-mesh lighting occurs even with shadows off. |
| COR-10 | High | Mesh material state | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:1389-1411` | Selection emphasis rewrites persistent material UBO memory while earlier frames may read it. | Two frames are in flight; no per-frame material storage or retirement edge protects the mapped writes. |
| COR-11 | High | Depth/split/environment descriptors | `src/visualizer/rendering/passes/vulkan_depth_blit_pass.cpp:55`, `:572-598`, `:633-635`; `vulkan_split_view_pass.cpp:129`, `:678-720`, `:764-766`; `vulkan_environment_pass.cpp:84`, `:526-537`, `:582-584` | Each child pass owns one mutable descriptor set despite two frames in flight and rotating external views. | Descriptor updates can mutate a set referenced by an older pending graphics submission. Parent viewport resources are per-frame; these children are not. |
| COR-12 | High | Resource lifetime | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:455-456`, `:658-682`, `:1312`; `vulkan_environment_pass.cpp:382-383`, `:543-562`; `vulkan_depth_blit_pass.cpp:422-440`; `vulkan_split_view_pass.cpp:437-480` | Mesh/environment/depth/split buffers, images, and views can be destroyed after waiting only their upload work. | `beginFrame` waits only the current frame slot; the other graphics frame can still sample/use these objects. This is use-after-free/device-loss territory. |
| COR-13 | Medium | Descriptor exhaustion | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:394-409`, `:658-667` | Mesh material descriptor sets are never freed from a fixed 256-material pool. | Repeated mesh replacement/eviction permanently consumes sets until allocation fails. The pool lacks `FREE_DESCRIPTOR_SET`. |
| COR-14 | High | RmlUi release robustness | `src/visualizer/gui/rmlui/rmlui_vk_backend.hpp:41-48`, `rmlui_vk_backend.cpp:1204-1215`, `:1251-1252`, `:1270-1316`, `:1354-1356` | RmlUi Vulkan checks become expression-only in release and several results are ignored. | OOM/invalid texture data can flow through null image/allocation/map/view handles. Host-image-copy transition/copy results are ignored. |
| COR-15 | Medium | External tensor lifetime/API | `src/visualizer/rendering/vulkan_external_tensor.cpp:49-68`, `:200-228`, `:287-307` | Storage/factory retain a raw `VulkanContext*`, and CUDA-exported parent `cudaPtr()` contradicts its comments and returns the no-op interop pointer. | A tensor escaping context lifetime can UAF on destruction; the latent parent/subview pointer API is internally inconsistent. |
| COR-16 | Medium | Image state tracking | `src/visualizer/window/vulkan_image_barrier_tracker.cpp:51-148` | Global host layout state is mutated when commands are recorded, not when submission succeeds. | Record failure, submit failure, or future concurrent recording can desynchronize tracked versus actual layout; the map has no locking or rollback. |
| COR-17 | Medium | Shadow instances | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:1449-1454`, `:1548-1645` | Shadow target/key live on cached mesh data, not a draw instance. | Two instances of one mesh with different transforms/lights overwrite the same shadow state. Latent because shadows default off (`vulkan_mesh_pass.hpp:43-45`). |
| COR-18 | Medium | Index width | `src/rendering/rasterizer/vulkan/shader/src/slang/cumsum.slang:41-50`, `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp:1461-1499`, `prepare_tile_sort.slang:38-45` | Tile-instance prefix/count is signed 32-bit and negative overflow is clamped to zero. | Extreme instance counts can silently render nothing. VRAM normally limits first, but the path needs an explicit ceiling/error. |
| COR-19 | High | Device-lost handling | `src/visualizer/window/vulkan_result.hpp:22`, `src/visualizer/window/vulkan_context.cpp:1039-1088`, `src/visualizer/rendering/vksplat_viewport_renderer.cpp:1648-1662` | Device loss is formatted as an error but has no terminal context state or recovery boundary. | Subsequent frames can keep entering submit/wait paths, and reset ignores the `vkDeviceWaitIdle` result. Full device recreation is out of scope, but the renderer must fail closed without poisoning fences/timelines. |
| COR-20 | High | GS readback coherency | `src/rendering/rasterizer/vulkan/src/buffer.cpp:157-190`, `:377-414` | The cumsum-tail staging allocation requests host access but is read after GPU copy without `vmaInvalidateAllocation`. | HOST_VISIBLE is not guaranteed HOST_COHERENT; a stale tile-instance count can mis-size or skip the frame. Other readback paths correctly invalidate. P3 deletes this hot read, but P0 must make the interim path portable. |
| COR-21 | Critical | WSI signal lifetime | `src/visualizer/window/vulkan_context.cpp:999-1016` | `render_finished_` is indexed by frame slot although presentation consumes it per swapchain image. | On a swapchain with more images than frame slots, a binary semaphore can be re-signaled while an earlier present wait is still pending. The already-fixed acquire side documents the same independent rotations at `:978-981`. |
| COR-22 | High | GUI interop failure | `src/visualizer/gui/gui_manager.cpp:4031-4039`, `:4183-4217` | Required interop failures throw from `prepareVulkanSceneInterop`, with no frame-boundary catch. | A transient transition/copy/signal failure escapes the GUI render path and can terminate the application instead of failing one frame with a stable diagnostic. |
| COR-23 | High | Render settings race | `src/visualizer/rendering/rendering_manager_vulkan.cpp:606-2690` | `renderVulkanFrame` repeatedly reads mutable `settings_` without one synchronized snapshot. | Other threads update settings under `settings_mutex_`; unsynchronized field/string reads are a C++ data race and can produce internally inconsistent frame decisions. |
| COR-24 | High (likely) | LOD teardown | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:1634-1639`, `:1648-1713`; `src/visualizer/rendering/lod_upload_engine.cpp:358-370`; `src/visualizer/rendering/spark_lod_controller.cpp:388-410` | LOD pool detachment/reset may race live decode/upload producers. | `drainAndSync` waits acquired staging slots but does not itself stop new producers; scene release destroys the external pool while the upload engine remains configured. P1 must trace producer ownership and either establish a stop/generation barrier or record a concrete refutation. |
| PERF-01 | High | Legacy GS hot path | `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp:1468-1501`, `src/rendering/rasterizer/vulkan/src/buffer.cpp:377-414`, `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:614-635` | The cumsum tail is synchronously copied and the CPU busy-spins on a fence mid-forward. | It splits the batch before sort/raster, burns a CPU core, holds shared arena/model access longer, and serializes the critical path. |
| PERF-02 | High | GPU contention | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:7225-7524`; `src/training/trainer.cpp:3441-3499` | A full projection/sort/raster/compose frame competes with training on the same device. | Separate queues/streams do not partition SM/L2/DRAM. Cost scales with viewport frequency and tile instances. |
| PERF-03 | Medium | Cached training frames | `src/visualizer/rendering/rendering_manager_vulkan.cpp:823-899`, `:1043-1046` | Trainer handshake, model read lock, and CUDA reader event are installed before the whole-frame cache hit is known. | Static/cached UI frames create avoidable synchronization and lock pressure without launching a new viewport batch. |
| PERF-04 | High | Output-image VRAM | `src/visualizer/rendering/vksplat_viewport_renderer.hpp:568-580`, `vksplat_viewport_renderer.cpp:1523-1568` | Twelve color+depth output pairs can remain allocated; only preview has a targeted release API. | At RGBA8+R32F, one three-image logical slot is 47.5 MiB at 1080p or 189.8 MiB at 4K. Two dormant split slots can retain about 94.9/379.7 MiB. |
| PERF-05 | Medium | Shared scratch VRAM | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:2899-2961`, `:2974-2976` | Legacy scratch scales approximately with splats, tile instances, and pixels, with a 384 MiB floor and 12.5% growth headroom. | This is a borrowed replacement for the training arena backing, not all duplicate VRAM, but over-reserved tile-instance high-water directly reduces training headroom. |
| PERF-06 | Medium | Point clouds | `src/visualizer/rendering/point_cloud_vulkan_renderer.cpp:848-896`, `:1174-1178` | Changed CUDA point clouds round-trip through CPU vectors and staging; command resources wait synchronously on reuse. | Large dynamic clouds stutter on D2H + CPU conversion + H2D. The fixed cache makes steady unchanged frames cheap. |
| PERF-07 | Medium | CPU recording | `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:880-999`, `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp:1045-1107` | Dispatch recording repeatedly constructs heap-backed descriptor/binding vectors. | Small per-dispatch costs accumulate across the many GS stages and are visible in CPU NVTX at high frame rate. |
| PERF-08 | Medium | Readbacks/picking | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:5120-5206`, `:5290-5394`, `:5491-5595` | Every depth/image/pixel readback allocates a mapped staging buffer and then synchronously submits/waits. | The branch fixed pool/cmd/fence churn, but allocator churn and GPU/CPU round-trip remain. |
| PERF-09 | Medium | Resize | `src/visualizer/window/vulkan_context.cpp:3613-3643` | Swapchain recreation waits all frame fences and then calls `vkDeviceWaitIdle` for untracked compute work. | Correct but produces resize stalls proportional to outstanding compute. |
| PERF-10 | Low | Dirty shadows | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:1456-1524`, `:1548-1583` | A genuinely dirty shadow uses a standalone command buffer, submit, and blocking fence wait. | Dirty gating prevents steady cost, but changed shadows still stall the UI thread instead of joining the frame command buffer. |
| PERF-11 | Medium | Compute/graphics overlap | `src/visualizer/gui/gui_manager.cpp:6406-6411`, `src/visualizer/window/vulkan_context.cpp:983-1013` | The graphics frame waits for VkSplat completion at `ALL_COMMANDS` although the external color/depth images are sampled in fragment stages. | The conservative mask prevents independent earlier graphics work from overlapping the compute tail. Narrow only after enumerating every consumer. |
| PERF-12 | High | CUDA stream serialization | `src/visualizer/gui/vulkan_ui_texture.cpp:609-615`, `src/visualizer/gui/gui_manager.cpp:4193-4205`; `src/training/trainer.cpp:1783-1791` | UI and scene-image CUDA uploads use the legacy NULL stream while training uses a blocking stream. | Legacy-default-stream work synchronizes with blocking streams, turning otherwise independent UI uploads into device-wide ordering points against training. Use a dedicated non-blocking upload stream and preserve the external-semaphore ordering. |
| PERF-13 | High | Mesh geometry residency | `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:1205-1215` | Vertex and index buffers request host-preferred memory and are used directly for drawing. | On discrete GPUs VMA may place geometry in host-visible system memory, so every draw can fetch static geometry over PCIe. Upload through staging into device-local buffers; retain host-visible material UBOs at `:1118-1127`. |
| PERF-14 | Medium | GT comparison | `src/visualizer/rendering/rendering_manager_vulkan.cpp:1392-1423` | Each dirty GT-comparison frame reloads, optionally GPU-undistorts, copies to CPU, and flips the static camera image. | Interaction/training refresh repeats expensive static preparation. Cache the prepared host tensor by camera UID, render width, and undistort state. |
| DESIGN-01 | Medium | Submission API | `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp:559-607`, `src/visualizer/window/vulkan_context.cpp:999-1039` | Sync2 barriers coexist with legacy submit structs everywhere. | Legal, but stage-mask conversion remains split across sync1/sync2 APIs and error handling is duplicated. Zero `vkQueueSubmit2` uses exist. |
| DESIGN-02 | High | Rendering manager | `src/visualizer/rendering/rendering_manager_vulkan.cpp:606-2690` | `renderVulkanFrame` is a ~2,085-line lifecycle/interop/render/split/cache/publish state machine. | Six repeated external-output clear/publish blocks appear around `:786-792`, `:957-963`, `:2201-2211`, `:2494-2501`, `:2643-2652`, and `:2656-2666`; correctness fixes must cover every exit. |
| DEAD-01 | Low | Device features | `src/visualizer/window/vulkan_context.cpp:163-214`, `:1652-1674`, `:1785-1853`, `:1946-1970`, `:1979-1981` | BDA, descriptor indexing, shader object, extended dynamic state 3, and cooperative matrix are enabled but unused. | No `vkGetBufferDeviceAddress` or non-context consumer of their `has*` accessors exists. BDA also unnecessarily restricts device selection and sets the VMA BDA flag. |
| DEAD-02 | Low | RmlUi backend | `src/visualizer/gui/rmlui/rmlui_manager.cpp:99`, `:681`; `rmlui_vk_backend.cpp:1485-1625`, `:1780-2084` | The self-owned RmlUi instance/device/surface/swapchain/frame path is unreachable in the application. | Application call sites use only `InitializeExternal` and `BeginExternalFrame`; the standalone backend duplicates VulkanContext. |
| DEAD-03 | Low | VkSplat helpers | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:602-622` | `ScopedCommandPool` and `ScopedFence` are defined but never instantiated. | Pure dead scaffolding. |
| DEAD-04 | Low | VkSplat input fallback | `src/visualizer/rendering/vksplat_viewport_renderer.hpp:272`, `:342-354`, `:415-417`, `:621-624`; `vksplat_viewport_renderer.cpp:2785-2881`, `:4508-4523`, `:7193-7205` | The full input-copy fallback is refused, but its three-slot storage, plug/alias/release helpers, always-false result flag, branch, and VRAM accounting remain. | No path allocates `cuda_inputs_`; missing external storage returns an error. Delete the unreachable fallback scaffold while keeping LOD page input regions and soft-delete opacity copies. The private `synchronize_upload` argument is also unread. |
| DEAD-05 | Low | Comments/API drift | `src/visualizer/rendering/vksplat_viewport_renderer.cpp:7527-7528`; `vulkan_mesh_pass.cpp:118`; `vulkan_split_view_pass.cpp:621-624`; `vulkan_external_tensor.cpp:200-218` | Comments describe a fence wait, a per-frame light UBO, an unsignaled first-use split fence, and CUDA pointer behavior that the code does not implement. | Misleads future synchronization work; delete or correct with the owning fix. |

## 4. Training-impact findings

### 4.1 What actually steals training throughput

The unavoidable cost is GPU duty, not the host timeline call itself. For a viewport frame with GPU residency `T_view` rendered at `f_view`, the first-order device-demand ceiling is:

```text
viewport GPU duty ~= min(1, f_view * T_view)
```

Example: a 25 ms viewport frame at the default static-training cadence of 1 Hz consumes roughly 2.5% device duty. The same frame while interaction drives 30 Hz requests demands roughly 75% device duty. Actual training slowdown is not exactly the duty number because CUDA and Vulkan can overlap when they use different bottlenecks, and the handshake can serialize otherwise-overlappable work. Measure it.

The current default limits training refreshes to one per second (`framerate_controller.hpp:12-20`, `framerate_controller.cpp:61-71`). Interaction, resize, selection, capture, and explicit dirty events can force additional work. Whole-frame cache gating exists at `rendering_manager_vulkan.cpp:1043-1046`.

Ranked training costs:

| Rank | Cost | Estimated recoverable amount | Why |
|---|---|---|---|
| 1 | Full GS GPU work on the shared device | Workload-dependent; up to most of the interactive slowdown | Projection is O(splats), radix/sort and emission scale with tile instances, raster scales with covered pixel-instance work, and compose scales with pixels. Dedicated compute queue does not reserve GPU capacity. |
| 2 | Legacy-NULL-stream UI/image uploads | Direct serialization removed for every affected upload; steady-state recovery depends on UI texture churn and must be measured. | Legacy default-stream semantics synchronize these uploads against the blocking training stream even though VkSplat already demonstrates a non-blocking render lane. |
| 3 | Legacy cumsum-tail host read/fence wait | Prior captures saw roughly 16-25 ms in affected zoomed-out frames; re-measure before/after. At 1 Hz that alone caps recoverable training duty around 1.6-2.5%; interactive recovery can be much larger. | It forces a submit/wait between cumsum and sort and lengthens the arena/model borrow. Current code still executes it for live training because HiGS is disabled by `synchronize_input_upload` (`vksplat_viewport_renderer.cpp:7007-7023`). |
| 4 | Overlay-capable raster variant | If current A/B confirms a 20-50% raster-stage penalty and raster is ~50% of frame GPU time, total frame opportunity is about 10-25%. | Plain pipelines exist and are selected when no visible overlay feature is active (`gs_renderer.cpp:1007-1010`, `:1045-1107`, `:1202-1204`; overlay predicate at `vksplat_viewport_renderer.cpp:3600-3620`). This is already implemented; the job is to verify it is active in normal training and avoid regressions. |
| 5 | Handshake/model lock before cache hit | Expected 0-3% training improvement on static/UI-active workloads; could be noise at true 1 Hz. | Cached frames still enqueue a params-ready wait/reader-done event and take the model lock before returning. Move only the handshake after the cache decision; do not weaken real render ordering. |
| 6 | VRAM pressure from output rings and scratch high-water | No direct it/s promise; can prevent OOM and reduce allocator/eviction pressure. | Two dormant full-size split slots can retain ~95 MiB at 1080p or ~380 MiB at 4K. Scratch has a 384 MiB floor and scales with measured tile-instance peaks. |

The handshakes are otherwise conceptually correct and should not be removed. `beginModelRead` orders the viewer after the last stable training parameters, and the reader-done/external-timeline waits prevent write-after-read on live model and shared arena memory. The optimization target is their duration and unnecessary cached-frame use, not correctness edges.

### 4.2 Memory accounting

Output images use RGBA8 color plus R32F depth: 8 bytes/pixel. With a three-image ring:

- One logical slot: `width * height * 8 * 3` = 47.46 MiB at 1920x1080, 189.84 MiB at 3840x2160.
- All four slots at full size: 189.84 MiB at 1080p, 759.38 MiB at 4K.
- Two dormant split slots: 94.92 MiB at 1080p, 379.69 MiB at 4K.

Shared scratch (`vksplat_viewport_renderer.cpp:2899-2961`) includes roughly 84 bytes per legacy splat before alignment and auxiliary arrays, 16 bytes per sort instance (two 32-bit key buffers plus two 32-bit index buffers), dense-batch state, and about 24 bytes per pixel for pixel/depth/contributor state. Allocation uses at least 384 MiB, adds 12.5% headroom, and rounds to 2 MiB pages (`:2974-2976`). It replaces/borrows the training arena allocation when active; do not report the entire block as duplicate VRAM.

The normal non-LOD model-input path is zero-copy Vulkan-external storage. If required model tensors are not external, `prepareInputs` now refuses the old full-copy fallback (`vksplat_viewport_renderer.cpp:4508-4523`), so normal training does not maintain three full model copies. A soft-delete mask can allocate a per-ring opacity copy (`:4374-4403`), overlays have their own bounded per-ring buffers, and RAD/LOD pages use their separate page-input storage path.

### 4.3 Measurement protocol

Use the same model, build, camera matrices, viewport size, and settings before/after. Capture three cases: static training at the one-second cadence, continuous camera orbit at a fixed request rate, and viewport hidden. Use at least one small model and one model large enough that tile-instance/raster cost dominates.

1. Warm 30 rendered frames; collect 300 rendered frames or at least 60 seconds of training.
2. Nsight Systems: CUDA + Vulkan + NVTX. Track `lfs.train`, `rasterize`, `vksplat.render.*`, `vksplat.command_batch.*`, graphics queue, compute queue, semaphore gaps, and per-iteration duration.
3. Report median and p95 viewport CPU record time, GPU frame time, cumsum/count stage, RasterizeForward, end-to-end present interval, training iterations/s, and training p95 iteration time.
4. Use existing GS Vulkan timestamps (`gs_pipeline.cpp:476-523`) for compute stages. Current pass-graph NVTX ranges measure CPU record time, not GPU duration; use Vulkan timestamp queries around graphics phases for GPU claims.
5. Capture VRAM from `VramProfiler`/`vksplat.memory` at steady state, after opening split view, after closing it, and after resize.
6. A/B overlay pipelines with a fixed camera and identical image output: no overlays; selection; rings/markers; crop/ellipsoid desaturation. Defaults keep rings/markers off (`rendering_types.hpp:219-222`), while training suppresses selection but still honors explicitly enabled markers (`viewport_request_builder.cpp:171-229`).

## 5. Visualizer performance findings, ranked

### 5.1 What limits FPS at large splat counts

1. **Legacy/live-training host serialization.** `executeCalculateIndexBufferOffset` reads the last prefix element (`gs_renderer.cpp:1495-1499`); `readElement` closes/submits the active batch (`buffer.cpp:377-414`); `endCommandBatch` busy-spins (`gs_pipeline.cpp:614-635`). This is a CPU and GPU bubble, not merely four bytes of transfer.
2. **Raster and tile-instance work.** Prior profiling put RasterizeForward with overlays near 50% of viewport cost at scale. Current code still contains overlay/plain variants and a correct activation predicate, so the number must be revalidated, not assumed. Large-count behavior is driven by visible splats, tile instances, overdraw, and viewport pixels—not splat count alone.
3. **Projection/depth/tile sorts.** The compute chain records projection, primitive depth sort, coverage/cumsum, tile sort, raster, and compose at `vksplat_viewport_renderer.cpp:7225-7524`. Radix work and scratch traffic grow with capacity even when the final raster is cheap.
4. **Dynamic point-cloud upload.** Changed point data makes a GPU->CPU->GPU trip. This is usually irrelevant for a cached static cloud but severe for live/edited large clouds.
5. **CPU record allocation.** Heap-backed binding/descriptor vectors are rebuilt for many dispatches. Expected opportunity is roughly 0.2-1 ms CPU/frame, to be proven with NVTX/allocation profiling.
6. **Synchronous readback/picking.** Full image/depth reads and even a four-byte depth sample allocate staging and wait the GPU. These are latency spikes rather than steady raster FPS unless tools poll them.
7. **Conservative compute-to-graphics wait.** `ALL_COMMANDS` blocks the complete graphics submit on VkSplat even though the external image is consumed by viewport fragment sampling. A precise consumer-stage mask can reduce latency through overlap.
8. **Resize drain.** `vkDeviceWaitIdle` is safe but can turn outstanding compute into visible resize stutter.
9. **Host-resident mesh geometry.** Static vertex/index buffers can be fetched over PCIe on discrete GPUs instead of from device-local VRAM.
10. **GT panel rebuild.** Dirty GT-compare frames repeat image load, undistort, device-to-host copy, and vertical flip for a static camera image.

### 5.2 Ranked opportunities and proof criteria

| Rank | Change | Expected payoff | Proof required |
|---|---|---|---|
| 1 | GPU-resident legacy tile count + indirect sort; remove mid-forward host wait | Remove one split submit and the observed 16-25 ms stall in affected frames; plausible 1.3-2x viewport gain when that stall dominates, not a blanket promise | No `readElement`/fence wait between cumsum and sort; Vulkan timestamps and nsys; fixed-camera image/count parity; training it/s |
| 2 | Ensure plain raster variant is selected whenever overlays are visually inactive | 10-25% total frame potential only if the current A/B confirms prior raster share/penalty | Per-stage timestamps, identical images, overlay feature matrix |
| 3 | Move live handshake/model-read work after whole-frame cache gate | Small CPU/throughput win, expected 0-3%; reduces unnecessary synchronization | CUDA event counts and training it/s with static viewport/UI animations; no missing WAR edge on a real render |
| 4 | Release inactive split output rings | Exact VRAM recovery of up to ~95 MiB 1080p / ~380 MiB 4K for two full-size slots | VramProfiler before/open/close/retire; validation clean; no premature destroy |
| 5 | Direct/ringed point-cloud GPU upload | Estimated 2-5x faster changed-frame upload; little steady cached-frame change | D2H eliminated in nsys, upload p50/p95, identical positions/colors/deletion behavior |
| 6 | Reuse readback staging buffers | Remove VMA allocation spikes; expected 0.1-1 ms CPU/call, but the synchronous GPU wait remains | allocator trace and pick/readback p95; pixel/image parity |
| 7 | Fixed-capacity/small binding arrays | Expected 0.2-1 ms CPU/frame at high dispatch count | CPU NVTX and allocation count; no shader binding changes |
| 8 | Narrow VkSplat graphics wait to actual consumer stages | Small-to-moderate visual latency win when graphics and compute overlap; no blanket training win | Queue overlap in nsys; validation clean; identical output across base/depth/split paths |
| 9 | Replace resize device-idle with exact compute timeline wait | Smaller resize p95 when compute backlog exists | resize trace; no swapchain-tied resource use after free |
| 10 | Stage mesh geometry into device-local buffers | Potentially large mesh-pass GPU win on discrete GPUs; exact gain scales with triangle count | VMA memory properties, mesh-pass GPU timestamps, PCIe traffic, image parity |
| 11 | Cache prepared GT comparison images | Removes repeated load/undistort/D2H/flip; expected multi-ms dirty-frame win for high-resolution GT | CPU NVTX/p95, cache hit counters, camera/width/undistort invalidation, image parity |

The checkout already contains a HiGS macro chain for eligible non-training, non-3DGUT, non-depth-capture frames (`vksplat_viewport_renderer.cpp:7002-7023`). The user-provided side-branch 1.76x result is context only. This effort must not merge a side branch or broaden into a HiGS redesign; the high-value incremental work is to reuse its GPU-count/indirect machinery for the legacy live-training chain.

## 6. Design critique and incremental direction

### 6.1 Batch lifecycle is implicit and exception-hostile

`DeviceGuard` currently owns begin, implicit submit in its destructor, optional fence wait, and timeline publication. `HostGuard` can close and reopen the batch inside helper calls. This makes a four-byte read unexpectedly split the frame and makes error propagation depend on destructor behavior.

Incremental direction: introduce explicit batch states (`Idle`, `Recording`, `Submitted`, `Failed`), a no-throw cancel path for unwinding, and an explicit `finish()` that returns/throws before a completion value is published. Keep the public renderer structure; do not rewrite the rasterizer.

### 6.2 Frame lifetime policy stops at the parent pass

The parent viewport pass already has per-frame resources (`vulkan_viewport_pass.cpp:240-253`, frame lookup at `:474-479`), but depth, split, environment, and mesh state keep single descriptors/UBOs. Resource replacement likewise destroys immediately instead of retiring behind a frame serial.

Incremental direction: thread the existing `frame_slot` through child `prepare`/`record`, allocate per-frame descriptor state, and use per-frame/per-draw immutable dynamic UBO ranges. For rare replacement tonight, waiting `VulkanContext::waitForSubmittedFrames()` before destruction is acceptable; a reusable serial-keyed retirement queue is the later design.

### 6.3 `renderVulkanFrame` owns too many state machines

At `rendering_manager_vulkan.cpp:606-2690`, one method owns dirty state, resize pause, model locks, handshake, backend choice, split panels, output publication, PPISP/readback, and failure cleanup. The repeated external-output blocks demonstrate the cost.

Incremental direction: first extract behavior-preserving helpers for (a) clearing/publishing viewport output, (b) live-render handshake scope, and (c) split-panel render/publish. Each extraction must be its own diff with existing behavior and tests unchanged. Do not redesign the renderer in one phase.

### 6.4 Submission and failure recovery are duplicated

Main frames, GS compute, point clouds, depth/split uploads, mesh shadows, RmlUi uploads, captures, and readbacks each implement their own reset-submit-wait sequence. Several poison fences on failure.

Incremental direction: fix each active path with the same invariant first—never expose an unsignaled reset fence as reusable, never publish a timeline value until submit succeeds, and classify `VK_ERROR_DEVICE_LOST` terminally. Only then factor a small submission helper or migrate central paths to `vkQueueSubmit2`.

### 6.5 Capability discovery contains speculative architecture

BDA, descriptor indexing, shader objects, extended dynamic state 3, and cooperative matrix increase pNext/extension complexity and minimum device requirements without a caller. The descriptor hazards should be solved with per-frame lifetime, not by enabling update-after-bind retroactively.

Incremental direction: delete unused features and accessors. Reintroduce a capability only in the same change that consumes it and proves a benefit.

### 6.6 Dead-code deletion inventory

Schedule deletion of:

- Unused BDA requirement/VMA flag, descriptor-indexing feature block, shader-object extension/feature/accessor, extended-dynamic-state-3 extension/feature/accessor, and cooperative-matrix extension/feature/accessor.
- `ScopedCommandPool` and `ScopedFence` in `vksplat_viewport_renderer.cpp:602-622`.
- The unread private `synchronize_upload` argument.
- Duplicated output reset/publish blocks after extracting helpers.
- Stale fence/per-frame/cudaPtr comments as their owning fixes land.
- The standalone RmlUi-owned Vulkan backend after the external path is hardened and a separate deletion-only build/validation diff proves no application call site.

Keep:

- Synchronization2: barriers actively use `vkCmdPipelineBarrier2`.
- Push descriptors: the GS compute pipeline consumes them.
- Float16 storage: the current macro chain consumes it.
- `VK_EXT_host_image_copy`: RmlUi uses it at `rmlui_vk_backend.cpp:1220-1310`, with an explicit pre-Volta disable at `vulkan_context.cpp:1855-1868`.
- Plain/overlay shader variants: both are active and are a real performance control.

## 7. Phased implementation plan

### Common validation gate

Every phase is independently landable and revertable. Never exceed eight build jobs.

Execution rule for the overnight run: do not stack new work on a phase that has missed its validation gate. Park it on its own commit/branch point and move to the next independent phase. Suggested timeboxes are P0 2 hours, P1 2 hours, P2 3 hours, P3 4 hours, P4 2 hours, and P5 1.5 hours; they are stop-loss limits, not estimates to pad. If time forces a choice after P1, prefer P3 for measured throughput/FPS impact or P2 for multi-mesh/validation correctness based on the plan-review priority.

Build:

```bash
cmake --build build -j8
```

Focused existing tests:

```bash
build/tests/lichtfeld_tests \
  --gtest_filter='ViewportRequestBuilderTest.*:ViewportFrameLifecycleServiceTest.*:SplitViewServiceTest.*:RenderSettingsDefaults.*:RenderSettingsProxy.*:RenderSettingsBackendNormalization.*:SceneManagerRenderStateTest.*:VksplatInputPackerTest.*:VkSplatLayouts/*:VkSplatDeviceLayouts/*'
```

The three baseline lifecycle failures listed above may remain only if their signatures are identical. Any new failure blocks the phase. For runtime phases, also run a debug validation-layer viewport smoke on NVIDIA with: no scene, one mesh, two meshes, point cloud, normal VkSplat, live training at the default cadence, continuous orbit, split view open/close, resize/minimize/restore, and clean shutdown. Use fixed-camera captures for visual changes.

### P0 — Revalidated `vulkan-fixes` residual: release-safe GS batch errors

**Goal:** retain the seven already-landed fixes, reimplement the one incomplete fix safely, and make a failed GS batch impossible to publish as successful.

**Exact changes:**

1. Cherry-pick nothing. Record the semantic matrix above in the implementation PR/commit message.
2. Replace `ENABLE_ASSERTION`-gated `_THROW_ERROR` with explicit always-on result/invariant handling; delete the dead compile-time switch. Preserve `_CHECK_FATAL` only if it remains a useful naming distinction.
3. Make `beginCommandBatch` set `commandBatchInProgress` only after `vkBeginCommandBuffer` succeeds.
4. Make `endCommandBatch` transactional: on end/submit failure, clear/cancel recording state, retain or deliberately discard pending waits, do not set `slot.pending_signal`, and return/throw the exact `VkResult`.
5. Add a no-throw cancel path used when a guard sees active exception unwinding; make normal completion explicit enough that a submit error reaches the caller. Remove destructor-driven double-exception termination.
6. Replace the unbounded `vkGetFenceStatus` spin's “anything other than success means keep spinning” logic with correct `VK_NOT_READY` handling and terminal propagation for device-lost/other errors. P3 removes this hot wait from the frame path; P0 makes it safe.
7. Invalidate the mapped cumsum-tail staging allocation after GPU completion and before reading it; propagate invalidate/map failure. P3 may delete this one remaining `readElement` caller later.
8. Verify all allocation/map/dispatch-limit checks in `buffer.cpp`, `gs_pipeline.cpp`, and `gs_renderer.cpp` now stop before invalid Vulkan calls.

**Files:**

- `src/rendering/rasterizer/vulkan/src/config.h`
- `src/rendering/rasterizer/vulkan/src/gs_pipeline.h`
- `src/rendering/rasterizer/vulkan/src/gs_pipeline.cpp`
- `src/rendering/rasterizer/vulkan/src/buffer.cpp`
- `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp`
- `src/visualizer/rendering/vksplat_viewport_renderer.cpp` only for explicit batch completion/cancel call sites

**Risk:** Medium. Hot synchronization code, but localized and directly auditable.

**Validation gate:** Common build/tests; release build smoke, not only debug; validation layers; force/temporarily instrument one begin/end/submit failure in a local diagnostic run and prove no timeline publication, no hang, and a clean subsequent reset. Remove any fault injection before landing. Fixed-camera image parity on normal and 3DGUT paths.

**Expected payoff:** Eliminates a release-only hang/invalid-handle class and completes the only still-relevant branch fix. Little steady-state performance change.

### P1 — Interop handoff and submit-failure integrity

**Goal:** make every timeline/fence state reflect work actually enqueued, and close the shared-scratch early-return hole.

**Exact changes:**

1. Hoist the insufficient-shared-sort-capacity result out of the active batch scope. A submitted completion must always update `last_signaled_render_value_`, `noteVulkanRelease`, and the live trainer callback before the error returns; an unsubmitted batch must cancel without publishing.
2. Make both CUDA timeline wrappers reject nonzero initial values and non-increasing signals in all builds. Advance `last_signaled_` only after `cudaSignalExternalSemaphoresAsync` returns success.
3. Make `ensureInitialized` transactional: construct renderer/timelines/imports into local/temporary ownership or add a single rollback scope; a failed attempt must leave the object in the same reset state as entry.
4. Repair reset-before-success failure paths. Main-frame failure must replace/retire the reset fence and acquired binary semaphore or mark device loss terminal; persistent readback/upload fences must be recreated signaled (or tracked as not submitted) on every command-reset/begin/end/submit failure after fence reset.
5. Add a terminal device-lost state to `VulkanContext`: the first `VK_ERROR_DEVICE_LOST` stops new acquire/submit/wait work, propagates one stable diagnostic, and lets teardown avoid pretending `vkDeviceWaitIdle` succeeded. Do not attempt in-process Vulkan/CUDA device recreation tonight.
6. Apply the same invariant to active depth/split/point-cloud transfer submit paths; do not build a large generic abstraction until behavior is correct.
7. Delete unused `ScopedCommandPool`, `ScopedFence`, the unread private `synchronize_upload` parameter, and stale batch-fence comments while touching these files.
8. Delete the unreachable full-model `CudaInputSlot` fallback: storage array, plug/alias/release helpers, always-false `uses_temporary_upload_slot`, dead branch, and zero-only VRAM accounting. Keep the separately active LOD page input layout and soft-delete opacity-copy ring.
9. Index render-finished binary semaphores by swapchain image, not frame slot, so a semaphore cannot be re-signaled while presentation still owns its prior wait. Keep acquire semaphores on their independent acquire rotation.
10. Keep PRESENT/UNDEFINED source scopes empty: this is intentional because the acquire semaphore wait at `COLOR_ATTACHMENT_OUTPUT` orders the same-submit transition and UNDEFINED discards contents. Add the invariant in code rather than inventing a fake source access.
11. Give GUI CUDA image uploads a dedicated non-blocking stream and pass it to both copy and external-semaphore signal operations. Catch required-interop exceptions at the GUI frame boundary, log, and fail that frame without unwinding the application.
12. Audit LOD producer shutdown/generation ordering before freeing the page input pool. Stop new submits before drain/free if a live race is confirmed; otherwise record the producer-lifetime proof in this log.

**Files:**

- `src/rendering/cuda_vulkan_interop.cpp`
- `src/rendering/cuda_vulkan_interop.hpp`
- `src/visualizer/rendering/vksplat_viewport_renderer.cpp`
- `src/visualizer/rendering/vksplat_viewport_renderer.hpp`
- `src/visualizer/window/vulkan_context.cpp`
- `src/visualizer/window/vulkan_context.hpp`
- `src/visualizer/rendering/point_cloud_vulkan_renderer.cpp`
- `src/visualizer/rendering/passes/vulkan_depth_blit_pass.cpp`
- `src/visualizer/rendering/passes/vulkan_split_view_pass.cpp`
- `src/visualizer/gui/gui_manager.cpp`
- `src/visualizer/gui/vulkan_ui_texture.cpp`

**Risk:** Medium. Error paths and rare capacity growth, with no shader/output change.

**Validation gate:** Common build/tests; validation layers; fixed-camera large-model run that forces scratch growth/clamping; start/stop training, resize during training, model switch, and shutdown; local failure injection at CUDA timeline signal and each submit family proving bounded error return and no next-frame fence hang. Nsight must show the same successful-path semaphore ordering.

**Expected payoff:** Removes one plausible training/viewport memory race, partial-init leaks, and several post-error permanent hangs. Negligible normal-path frame-time change.

### P2 — Per-frame descriptors, per-draw mesh state, and safe replacement

**Goal:** eliminate active descriptor/UBO races and prevent resources from being destroyed while either frame slot can still reference them.

**Exact changes:**

1. Thread existing `ViewportRecordContext::frame_slot` into depth, split, environment, and mesh child prepare/record calls.
2. Allocate depth/split/environment descriptor sets per frame slot. Cache bound view/generation per slot; update only the slot whose fence `beginFrame` already retired.
3. Replace the mesh singleton light UBO/set with per-frame growable, aligned per-draw UBO storage and per-draw descriptor sets from a frame-local pool reset only after that frame slot retires. Shadow image binding must be immutable for the recorded draw.
4. Move emphasis/flash values out of mutable persistent material UBO writes into per-draw state (the light/draw UBO or push constants). Keep material texture/base parameters immutable after upload.
5. Enable individual material descriptor frees and free them on mesh teardown. Size pools from actual frame/material needs or grow predictably instead of silently exhausting at 256.
6. Before rare mesh/environment/depth/split image or buffer replacement, wait all submitted graphics frames or queue retirement behind known frame serials; upload fences alone do not cover graphics sampling. Use the simple wait-all cold path tonight if a shared retirement helper would expand scope.
7. Preserve the existing dirty shadow key/gate. Do not attempt multi-instance shadow redesign in this phase.
8. Allocate static vertex/index buffers device-local with transfer-destination usage and upload through the existing staging path. Keep frequently rewritten material/light UBOs host-visible; audit their coherent flushes separately.

**Files:**

- `src/visualizer/rendering/passes/vulkan_viewport_pass.cpp`
- `src/visualizer/rendering/passes/vulkan_depth_blit_pass.{hpp,cpp}`
- `src/visualizer/rendering/passes/vulkan_split_view_pass.{hpp,cpp}`
- `src/visualizer/rendering/passes/vulkan_environment_pass.{hpp,cpp}`
- `src/visualizer/rendering/passes/vulkan_mesh_pass.{hpp,cpp}`
- Mesh shader source/generated resource inputs only if per-draw layout changes

**Risk:** Medium-high. It changes binding layouts and normal multi-mesh rendering, but is bounded to visualizer passes.

**Validation gate:** Common build/tests; validation layers with GPU-assisted descriptor validation; two differently lit meshes, repeated mesh add/delete/reload beyond 256 cumulative materials, emphasis/flash while rendering two frames in flight, split/depth external views rotating every frame, environment switch/disable, resize, and shutdown. Capture deterministic images for one mesh, two meshes, emphasis, shadows off, and shadows on. No descriptor update-while-pending messages are acceptable.

**Expected payoff:** Fixes wrong multi-mesh lighting and the largest active validation/device-loss hazard in the presentation stack. May reduce map/flush churn; performance is secondary.

### P3 — Remove the legacy cumsum readback with GPU count and indirect sort

**Goal:** remove the top hot-path CPU stall and split submit without adopting a side-branch renderer.

**Exact changes:**

1. Keep legacy projection/depth ordering/cumsum semantics. After cumsum, run the existing `prepare_tile_sort` GPU kernel with a bounded sort capacity; allocate its count buffer at two words, write clamped count to `tile_sort_count[0]`, raw count to `[1]`, and write indirect dispatch args.
2. Pass the chosen capacity explicitly through legacy key generation instead of resizing from host `buffers.num_indices`; set `uniforms.sort_capacity` to that capacity so `tile_shader.slang:94-102` bounds every key write. Use `executeSortTileInstances`/the existing indirect radix machinery for the legacy tile sort instead of host count dispatch sizing.
3. Record tile-range construction against capacity plus its sentinel invocation; retain the shader's GPU-tail clamp (`tile_shader.slang:233-260`) so only the clamped sorted prefix is read. No downstream stage may use a stale host count as a memory bound.
4. Record the existing two-word deferred instance-count readback at the end of the unsplit batch; poll it on a later completed timeline value to update `num_indices`/high-water and grow after a clamped frame.
5. Seed first-frame capacity conservatively, preserve the 12.5%/high-water policy, and mark a clamped one-shot capture unsettled until a complete retry. Interactive frames may self-heal; exports/captures may not return partial output.
6. Delete the hot-path `readElement` call and its special cumsum-tail staging use. Keep generic readback only if another live caller remains.
7. Preserve the legacy path for live training/3DGUT/depth capture; this phase changes dispatch/count transport, not raster math or HiGS eligibility.
8. Make `prepare_tile_sort` distinguish a negative cumsum tail (impossible for nonnegative coverage without signed overflow) from a genuine zero count, emit a reserved overflow sentinel in the raw-count word, suppress key writes, and surface a clear deferred error instead of silently rendering zero indices. Do not convert the whole sort format to 64-bit tonight.

**Files:**

- `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp`
- `src/rendering/rasterizer/vulkan/src/gs_renderer.h`
- `src/rendering/rasterizer/vulkan/src/buffer.cpp` if the now-unused hot read helper can be deleted
- `src/rendering/rasterizer/vulkan/shader/src/slang/prepare_tile_sort.slang`
- `src/rendering/rasterizer/vulkan/shader/src/slang/tile_shader.slang`
- `src/visualizer/rendering/vksplat_viewport_renderer.cpp`
- `src/visualizer/rendering/vksplat_viewport_renderer.hpp`

**Risk:** Medium-high. Highest performance payoff, with one-frame capacity/clamp semantics to validate.

**Validation gate:** Common build/tests; shader rebuild under `-j8`; validation layers; fixed-camera color/depth image comparison for normal, 3DGUT, overlay on/off, depth capture, split view, and live training. Test camera jumps from low to extreme overdraw and one-shot export after the jump. Assert raw/clamped counts and eventual full recovery. Nsight must show no host fence wait between cumsum and sort, one GS submit per frame, and no CPU `_mm_pause` hot span. Report Vulkan stage timestamps, viewport median/p95, and training iterations/s using the protocol in section 4.3.

**Expected payoff:** Removes the prior 16-25 ms affected-frame stall if reproduced; likely the largest overnight FPS/training-throughput win. No guaranteed gain when raster GPU time already dominates.

### P4 — Cheap training-throughput and VRAM recovery

**Goal:** stop synchronizing cached frames and release large inactive allocations without weakening live-render ordering.

**Exact changes:**

1. Move creation/installation of the live trainer handshake scope, model read lock, and reader event after the whole-frame cache decision. Keep scene-state/dirty computation sufficient to make the decision, and prove every real VkSplat submit still has both forward and reverse edges.
2. Add safe release/retirement for inactive `SplitLeft` and `SplitRight` output rings after split mode closes or a panel stops using VkSplat. Wait each ring completion and submitted graphics frames before destruction; do not free the image currently sampled by a pending frame.
3. Reuse high-water mapped staging buffers for full color, full depth, and four-byte pick/depth readback. Preserve the persistent command pool/buffer/fence already on master.
4. Enumerate every graphics consumer of the VkSplat completion semaphore and narrow the main-frame wait from `ALL_COMMANDS` to the earliest actual consumer stage (normally fragment shader, OR-ing any proven transfer/attachment consumer). Keep `ALL_COMMANDS` if validation or traces show an unmodeled consumer.
5. Replace repeated viewport output clear/publish blocks with small behavior-preserving helpers; do not otherwise split `renderVulkanFrame` tonight.
6. At `renderVulkanFrame` entry, copy `settings_` once under `settings_mutex_` and use that immutable snapshot for the entire frame.
7. Cache the prepared GT-comparison host tensor by camera UID, render width, and undistort state; invalidate on key changes and scene/camera teardown.

**Files:**

- `src/visualizer/rendering/rendering_manager_vulkan.cpp`
- `src/visualizer/rendering/rendering_manager.hpp` if helper declarations are needed
- `src/visualizer/rendering/vksplat_viewport_renderer.cpp`
- `src/visualizer/rendering/vksplat_viewport_renderer.hpp`
- `src/visualizer/gui/gui_manager.cpp`

**Risk:** Medium. Cache/handoff ordering and external-image lifetime need careful exit coverage.

**Validation gate:** Common build/tests, including confirming the same three baseline failures only; validation layers; static training with UI animation and no dirty viewport, then real camera/selection/resize renders; nsys event/semaphore counts; split open/close/reopen at 1080p and 4K; VramProfiler proves expected retirement; readback/pick pixel parity and repeated-call p95.

**Expected payoff:** Expected 0-3% training throughput in synchronization-heavy static cases, exact recovery of dormant split-ring VRAM, lower readback allocation spikes, and less exit-path duplication.

### P5 — Delete unused capabilities and harden the active RmlUi path

**Goal:** finish the low-risk dead-code order and prevent UI texture OOM/error paths from continuing with null handles.

**Exact changes:**

1. Remove BDA from required-feature selection and device enablement; remove the unused VMA BDA allocator flag.
2. Remove descriptor-indexing enablement/runtime flag/accessor and the false comment that RmlUi/viewport use it.
3. Remove shader-object, extended-dynamic-state-3, and cooperative-matrix extension discovery, queried/enabled feature structs, flags/accessors, and logging.
4. Keep push descriptor, float16, host image copy, synchronization2, timeline semaphore, dynamic rendering, subgroup controls, and atomic features that have consumers.
5. In the active external RmlUi path, validate source/dimensions/allocator in release, check every create/map/host-copy/transition/view result, unwind partially created texture resources, and return a failed texture handle without poisoning renderer state.
6. Correct comments/API drift identified as DEAD-05.
7. Prove `VulkanExternalTensorStorage::cudaPtr()` has no caller, then delete the broken/dead parent-pointer API and its contradictory adapter comments. If a real caller appears during implementation, store the exported CUDA base pointer explicitly instead; do not leave a null-returning latent API.

**Files:**

- `src/visualizer/window/vulkan_context.cpp`
- `src/visualizer/window/vulkan_context.hpp`
- `src/visualizer/gui/rmlui/rmlui_vk_backend.cpp`
- `src/visualizer/gui/rmlui/rmlui_vk_backend.hpp`
- `src/visualizer/rendering/vulkan_external_tensor.cpp`
- `src/visualizer/rendering/vulkan_external_tensor.hpp`

**Risk:** Low-medium. Mostly deletion, plus an active UI error path.

**Validation gate:** Common build/tests; device initialization log/capability diff; debug validation smoke; RmlUi texture/font/image load, reload, host-image-copy on supported hardware, staging fallback with host image copy disabled, and local allocation-failure injection. Confirm no removed `has*` accessor callers and no BDA API use with `rg`.

**Expected payoff:** Smaller device-selection surface and feature chain, better compatibility, less dead architecture, and bounded UI OOM behavior. Negligible steady frame-time change.

## 8. Stretch if time remains

Ordered by value after P0-P5 are green:

1. **Small-vector/fixed-array GS bindings.** Replace per-dispatch heap vectors where binding counts are compile-time bounded. Gate on an allocation trace showing material churn; target 0.2-1 ms CPU/frame.
2. **Direct/ringed point-cloud upload.** For CUDA tensors, use an external Vulkan buffer or CUDA->Vulkan staging copy plus timeline; keep CPU tensors on the staging path. Ring command/fence resources. No `src/core/tensor` edits—flag an API limitation rather than crossing scope.
3. **Graphics-pass GPU timestamps.** Add a per-frame query ring around environment/base scene/mesh/overlay/RmlUi phases. Keep existing CPU NVTX. This is measurement infrastructure, not a claimed optimization.
4. **Central submit2 migration.** Convert VulkanContext main-frame and GS compute submission first, preserving exact waits/signals and fixing failure state in the same small commits. Do not mass-convert every one-shot path.
5. **Delete standalone RmlUi Vulkan ownership path.** Separate deletion-only change after application call-site proof. Keep external backend and shared resource managers.
6. **Dirty-shadow integration.** Record dirty shadow rendering into the frame command buffer or a tracked async submission rather than submit+wait. Preserve the current dirty key.

## 9. Not tonight

- Merge or transplant the HiGS side branch. Current master already has a macro chain; the side-branch benchmark is context, not an integration target.
- Rewrite `renderVulkanFrame`, VulkanContext, the pass graph, or GS rasterizer wholesale.
- Adopt descriptor-indexing/bindless to paper over descriptor lifetime. Per-frame immutable descriptors are simpler and safer.
- Replace all submissions with `vkQueueSubmit2` in one mechanical sweep. Correct failure/lifetime invariants first.
- Build global GPU QoS, queue priorities, or adaptive training/viewer scheduling without measurements. Queue priority does not partition SM/DRAM.
- Replace swapchain `vkDeviceWaitIdle` until VkSplat compute completion is registered in a context-owned, bounded retirement model.
- Redesign `VulkanImageBarrierTracker` into submission-transactional, queue-aware state. Current recording is serialized; a correct rollback/commit model should follow centralized submission ownership rather than be patched into each command buffer tonight.
- Convert tile keys/counts to 64-bit. Add the explicit ceiling now; a format-wide conversion touches shaders, sort memory, and performance.
- Redesign multi-instance shadows. Shadows are default-off; fix active light/descriptor correctness first and schedule per-instance shadow ownership separately.
- Replace the raw `VulkanContext*` ownership in external tensor storage with a general shared context-lifetime system. Tonight, preserve and assert shutdown ordering; do not invent broad shared ownership around the window/device. Flag any escaping tensor found during P5 for a dedicated lifetime fix.
- Use `VK_EXT_host_image_copy` for general readback. The extension is used for UI upload, is disabled on pre-Volta due a driver crash, and needs measured readback support/benefit before expansion.
- Edit `src/core/tensor`, `src/io`, `src/fusion`, or `src/training`. If direct point-cloud upload or arena ownership exposes an API gap, document it for the owning stream.
- Add/update tests in this effort. Use the existing gtests and runtime validation/profiling gates specified above.

## 10. Considered and rejected

| Proposal | Decision | Reason |
|---|---|---|
| Per-pass `frame_dirty` caching in the viewport graph | **Rejected** | The prior dependency analysis showed the required mask degenerates to all bits; current whole-frame gating already returns cached output at `rendering_manager_vulkan.cpp:1043-1046`. Per-pass state would add invalidation complexity without saved work. |
| Treat sync2 barriers + `vkQueueSubmit` as a validation bug | **Rejected** | It is legal. Submit2 is a clarity/modernization task, not the root of the observed hangs/races. |
| Cherry-pick all `vulkan-fixes` commits | **Rejected** | None applies cleanly; seven named semantics are present, the remaining macro fix is incomplete, and the mixed final commit crosses scope. Reimplement only the residual against current lifecycle code. |
| Disable async compute to protect training | **Rejected as default** | It may reduce overlap with graphics without reducing total GPU demand. Keep as an A/B diagnostic; choose policy from nsys/training it/s, not queue folklore. |
| Remove the trainer/viewer timeline handshake | **Rejected** | It protects live-model and shared-arena write-after-read. Shorten unnecessary/cache-held time instead. |
| Force HiGS for live training | **Rejected tonight** | Current code explicitly disables it because deferred readback lacked safe ordering against training mutation (`vksplat_viewport_renderer.cpp:7007-7013`). Reuse only the GPU count/indirect mechanics while keeping the legacy live path. |
| Always use overlay-capable raster shaders | **Rejected** | Plain variants are present and should remain the normal no-overlay path. Measure and enforce correct activation instead. |
| Always use plain shaders during training | **Rejected** | Explicit rings/markers/crop desaturation can remain visually requested during training. The activation predicate, not training status alone, is authoritative. |
| Free every cached output immediately when inactive | **Rejected** | Images can still be referenced by the compute ring or one of two graphics frames. Release only behind both completion domains. |
| Make every replacement call `vkDeviceWaitIdle` | **Rejected** | It is safe but turns ordinary scene edits into global stalls. Use submitted-frame waits for rare graphics-owned resources and timeline/frame retirement where already available. |
| Introduce a general render graph/resource allocator tonight | **Rejected** | Too broad and hard to validate unattended. Per-frame child resources and small helpers solve the active defects incrementally. |
| Optimize point clouds before GS cumsum | **Rejected ordering** | Point-cloud upload is conditional on changes; the cumsum stall is in the live large-splat critical path and has existing indirect machinery ready to reuse. |
| Claim the prior 1.76x HiGS or ~50% RasterizeForward numbers as current | **Rejected** | Code paths were verified, not runtime numbers. They are hypotheses/baselines until the fixed-camera timestamp/nsys protocol reproduces them on this checkout. |

## Morning success criteria

Minimum good night: P0 and P1 land independently, release/failure injection cannot publish unsignaled values, and no new baseline/test/validation failures exist.

Strong night: P0-P3 land; multi-mesh/descriptors are clean; the legacy host read and split submit disappear; fixed-camera output matches; and measured viewport/training results substantiate the win.

Excellent night: P0-P4 land, dormant split VRAM retires, cached training frames stop creating reader edges, and the report contains before/after nsys, Vulkan timestamp, FPS, it/s, p95, and VRAM numbers. P5/standalone RmlUi deletion remains independently optional rather than jeopardizing the hot-path work.

## Overnight execution log

### P0 — release-safe GS batches

- Commits: `2a9acd756` (`fix(vulkan): make GS command batches fail closed`).
- Landed: always-on release checks; transactional begin/end/submit state; no-throw cancellation during exception unwinding; checked timestamp/fence/queue results; mapped-readback invalidation; no-throw renderer teardown. The former `_mm_pause` poll is now one checked `vkWaitForFences` call, preserving the completion point without burning a CPU core.
- Gates: `cmake --build build -j8` passed. Focused filter ran 55 tests: 52 passed; only the three documented `ViewportFrameLifecycleServiceTest` failures remained with the baseline signatures (dirty `16` versus `9`, deferral incomplete). `git diff --check` passed before commit.
- Measurements: no GUI/runtime capture by contract. P0 changes CPU waiting behavior but does not remove the split submit; P3 remains the performance measurement point.
- Deviations: no stale `vulkan-fixes` commit was cherry-picked. Fault injection, release viewport smoke, validation layers, and fixed-camera image parity require interactive validation and are deferred.
- Amendment status: all seven round-2 additions are now represented in the inventory/plan. The NULL-stream upload fix is assigned to P1 because it is both a direct training-throughput serialization point and part of the same CUDA/Vulkan semaphore failure boundary. WSI barrier review preliminarily supports the counter-argument: an empty PRESENT source scope is correct when the same submit waits acquire at the transition's `COLOR_ATTACHMENT_OUTPUT` destination stage; P1 will document that invariant in code. The per-image render-finished semaphore issue is confirmed. LOD teardown remains under active audit.

### P1 — interop, WSI, submit failure, and teardown integrity

- Commits: `16d05c009` (`fix(vulkan): isolate CUDA GUI uploads from training`), `5562b3ac2` (`fix(vulkan): harden WSI and transfer submission lifetimes`), and `0d6a86c9c` (`fix(viewport): close VkSplat handoff and LOD teardown races`).
- Landed: strict runtime timeline monotonicity with bookkeeping only after successful CUDA enqueue; transactional VkSplat initialization rollback; dedicated non-blocking CUDA streams for UI textures and all three GUI scene/split/depth uploads; GUI-frame exception containment; per-swapchain-image render-finished semaphores; frame-fence recovery and swapchain retirement after failed submit; recoverable depth/split/point-cloud transfer fences; shared-scratch release publication on the capacity error path; producer-first LOD shutdown; Spark traversal quiescence before detach.
- Dead code deleted: `ScopedCommandPool`, `ScopedFence`, unused `copyCudaBytes`, unread `synchronize_upload`, and the unreachable three-slot full-model `CudaInputSlot` fallback, its alias/release helpers, result flag, branch, and zero-only VRAM accounting.
- Gates: `cmake --build build -j8` passed. Focused filter remained 52/55 with exactly the three baseline lifecycle failures and signatures. Additional `LodUploadEngine.*` gate passed 5/5, including multi-worker monotonic signaling and reconfigure-drain coverage. `git diff --check` passed before each commit.
- Measurements: static proof only by contract. `rg` confirms no remaining GUI `copyTensorToSurface` or associated signal call uses a defaulted/NULL stream. No live nsys or presentation trace was collected.
- Amendment verdicts: NULL-stream training serialization **confirmed and fixed** for UI texture, main scene, split-right, and depth-blit uploads. Per-frame-slot render-finished reuse **confirmed and fixed** per swapchain image. Interop exceptions escaping the GUI frame **confirmed and fixed**. LOD teardown **confirmed**: draining before `LodPageCache::reset()` allowed a decode worker to acquire/submit after the drain; shutdown now joins producers, then drains/unconfigures, then frees storage. Spark detach also cleared traversal vectors while its worker traversed them and now waits for cancellation completion.
- WSI barrier verdict: the proposed PRESENT-source-scope bug is **refuted**. The acquire semaphore wait at `COLOR_ATTACHMENT_OUTPUT` directly orders the same-submit transition whose destination is `COLOR_ATTACHMENT_OUTPUT`; PRESENT has no source access performed by this queue, and UNDEFINED discards contents. The invariant is now documented in `vulkan_image_barrier_tracker.cpp`. No fake source stage/access was added.
- Deviations/refutations: the persistent VkSplat readback fence does not have the claimed next-use hang: those synchronous paths reset without first waiting on a previous failed submit, which is legal for an unsignaled fence not in use, and teardown idles the device. The actually poisoned wait-before-reuse paths (depth, split, point-cloud, and main frame) were repaired. A broad terminal `device_lost_` state was not added without an interactive fault-injection gate; existing failures now fail closed/recover local fence state, while full device recreation remains not-tonight.
- Needs interactive validation: validation-layer WSI smoke on a three-image swapchain; forced main/transfer submit failure; CUDA upload failure; live RAD model switch/reset under decode load; training with UI texture churn in nsys; resize/minimize/restore; and clean shutdown.

### P2 — immutable descriptors, mesh draw state, and device-local geometry

- Commits: `ef907a3a8` (`fix(vulkan): make mesh draw state immutable`) and `a772975b3` (`fix(vulkan): isolate viewport descriptors by frame`).
- Landed: frame-slot descriptor sets for depth blit, split view, and environment sampling; explicit frame-slot routing through child prepare/record/ready queries; per-frame, per-draw light/shadow UBO descriptors sized for every split-view panel occurrence; selection emphasis moved from mutable material UBOs into immutable draw state; growable/freeable material descriptor pools; checked material upload failure; submitted-frame retirement before replacing sampled depth/split/environment resources; and dead one-shot helper deletion from depth/split passes.
- Mesh geometry amendment: **confirmed and fixed**. Vertex and index buffers now use `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE` plus `TRANSFER_DST`, one paired staging upload, and explicit transfer-to-vertex/index memory barriers. The allocation formerly near `vulkan_mesh_pass.cpp:1125` is the 64-byte material UBO, not streamed geometry; it remains host-visible because it is a tiny one-time upload and making every material use a separate staging submit would regress scene-load latency for negligible steady-state bandwidth. Its write/flush result is now checked, and it is no longer mutated per frame.
- Correctness details: each mesh draw and each split panel records against a distinct descriptor/UBO, so later CPU writes cannot retroactively change earlier commands in the same command buffer. Material descriptor sets are freed on mesh teardown and additional 256-set pools are allocated predictably instead of accumulating dead sets until a silent hard ceiling. Rare shared-image replacements use the existing submitted-frame wait rather than trusting transfer fences that do not cover later graphics sampling. Static shadow dirty gating is unchanged.
- Gates: `cmake --build build -j8` passed after both source commits. The exact focused filter ran 55 tests: 52 passed; only the three documented `ViewportFrameLifecycleServiceTest` failures remained with identical dirty `16` versus `9` and incomplete-deferral signatures. `git diff --check` passed before both commits.
- Measurements: shader compilation and full link passed; no live frame/GPU measurement was taken because the contract forbids launching the GUI. The expected steady-state win is elimination of PCIe vertex/index fetches on discrete GPUs; magnitude remains unclaimed until a fixed-camera mesh trace.
- Deviations: frame-local light resources retain descriptor sets at a per-slot high-water mark rather than resetting a pool every frame, avoiding allocator churn while preserving slot-fence lifetime. Material UBOs were not moved device-local for the reason above. No multi-instance shadow ownership redesign was attempted.
- Needs interactive validation: GPU-assisted descriptor validation with two differently lit meshes and split panels; emphasis/flash while two frames are in flight; shadows off/on and dirty gating; repeated add/delete/reload beyond 256 cumulative materials; rotating external depth/split image views; environment switch/disable; resize; fixed-camera image parity; and a discrete-GPU trace confirming device-local vertex/index residency and draw throughput.

### P3 — GPU-resident tile-instance count and indirect dispatch

- Commit: `de667ae37` (`perf(vulkan): keep tile instance counts on GPU`).
- Landed: the legacy cumsum, GPU count preparation, key generation, radix sort, range construction, and raster record in one command batch without the mid-forward host readback. Both legacy and macro chains now consume the same two-word GPU count: word zero is capacity-clamped and word one is the raw count or signed-overflow sentinel. Radix and range work are dispatched indirectly; key generation and raster scratch use an explicit `INT32_MAX`-bounded capacity rather than a stale host count. Deferred timeline-gated readback updates only the next frame's high-water policy and one-shot settle state.
- Correctness details: negative offsets are rejected before key writes; range kernels synthesize their terminal boundary without reading `sorted_keys[capacity]`; first-frame capacity is seeded at four instances per active splat and grows by deferred high-water after a clamped frame; one-shot capture settles only after both visibility and tile-count readbacks prove a complete steady-state chain. A negative signed prefix tail fails the following frame with a clear overflow error instead of being interpreted as an empty render. Training scratch priming is capped at the same signed prefix limit.
- Dead code deleted: generic `readElement` and its explicit instantiations, the host-sized/direct radix entry point and pipeline objects, three direct radix shader variants, and the compile-time direct/count-buffer shader branches. The only remaining tile radix implementation is GPU-count-driven.
- Gates: `cmake --build build -j8` passed, including recompilation of all changed Slang/GLSL shaders and the full application/test link. The exact focused filter ran 55 tests: 52 passed; only the three documented `ViewportFrameLifecycleServiceTest` failures remained with identical dirty-state and incomplete-deferral signatures. `git diff --check` passed before commit.
- Measurements: static evidence confirms no `readElement`, fence busy-wait, direct radix pipeline, or mid-forward batch split remains. No live nsys, Vulkan timestamps, FPS, training iterations/s, or image captures were taken because the contract forbids launching the GUI. The earlier 16–25 ms stall is therefore still a hypothesis, not a claimed measured gain.
- Deviation from plan: tile/macro range construction also became indirect instead of recording a capacity-wide dispatch. This avoids potentially enormous no-op dispatches while retaining the terminal sentinel invocation and bounded reads. The signed 32-bit ceiling was enforced rather than converting prefix/sort formats to 64 bit.
- Needs interactive validation: validation layers; fixed-camera color/depth parity for normal, 3DGUT, overlays on/off, depth capture, split view, and live training; low-to-extreme-overdraw camera jumps and convergence after a clamped frame; one-shot export/capture convergence; nsys proof of one GS submit and no cumsum-to-sort host wait; Vulkan stage timestamps, median/p95 FPS, VRAM high-water, and training iterations/s.

### P4 — cached-frame synchronization, GT preparation, and dormant VRAM

- Commit: `b5f4d7b93` (`perf(viewport): avoid redundant frame synchronization`).
- Landed: `renderVulkanFrame` now takes one immutable `RenderSettings` copy under `settings_mutex_` and uses it throughout; the training scratch-prime entry point also uses the locked accessor. The trainer model lock, CUDA read epoch, renderer stream guard, and forward/reverse timeline handshake are installed only after all whole-frame cache/resize-defer returns. Real render work retains the same `beginModelRead` edge, per-submit prompt publish, and scope-exit `endModelRead` edge.
- GT amendment: **confirmed and fixed**. The prepared, vertically flipped CPU GT tensor is cached by camera UID, render width, undistort request state, and source path; cache entries are replaced on key changes and cleared on model/render-resource teardown. Repeated dirty frames still rerender the comparison panel but no longer reload, GPU-undistort, synchronize `.cpu()`, and flip the static photograph.
- VRAM/readback: inactive split-left/right output rings now retire only after the GS batch, every renderer ring value, and submitted graphics frames complete. Full color, full depth, and four-byte depth-pick readbacks share one mapped 64 KiB-aligned, grow-only staging allocation instead of three per-call allocation/destruction paths. The old scoped staging helper and repeated viewport-output reset blocks were deleted; a single reset helper now also clears stale GT content bounds.
- Submission overlap: the main-frame VkSplat timeline wait moved from `ALL_COMMANDS` to `FRAGMENT_SHADER`, the earliest consumer proven by the viewport, split, and depth-blit pass graph. This allows earlier graphics work to record/execute while async compute finishes without weakening the image-sampling dependency.
- Gates: `cmake --build build -j8` passed after the final source state. The exact focused filter ran 55 tests: 52 passed; only the three documented `ViewportFrameLifecycleServiceTest` failures remained with the same dirty-state and incomplete-deferral signatures. `git diff --check` passed before commit.
- Measurements: static proof only. Source ordering confirms every cache return precedes `ensureHandshakeReady`/`beginModelRead`; all direct frame-body `settings_` reads are gone; and no per-call `ScopedStagingBuffer` remains. No GUI, nsys, Vulkan timestamp, FPS, iterations/s, or live VRAM trace was run under the no-GUI contract, so the planned 0–3% static-frame training recovery and readback p95 reduction remain unmeasured.
- Deviations: the GT key additionally includes the image path to prevent same-UID reuse across scene replacement. Split images are released on the first successful non-split publication, not immediately on a mode event, so a failed transition can safely keep presenting its last valid frame. No broad `renderVulkanFrame` decomposition was attempted beyond deleting the repeated reset block.
- Needs interactive validation: training with an unchanged camera to prove cached frames emit no reader events and improve/hold iterations/s; a dirty live-training frame to prove both handshake directions remain; GT compare camera/width/undistort changes and scene reload; split open/close/reopen at 1080p and 4K with VramProfiler; repeated color/depth/pick readback parity and p95; validation-layer sampling with the fragment-stage wait; resize/minimize/restore; and screenshot/lazy-capture behavior immediately after closing split view.
