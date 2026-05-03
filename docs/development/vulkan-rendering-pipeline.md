# Vulkan rendering pipeline

How a frame goes from CUDA tensors to pixels on screen.

## Flow

```
trainer (CUDA)            VulkanContext               viewport pass            RmlUi backend           SDL3 window
     │                         │                           │                       │                        │
     │  splat_data tensors     │                           │                       │                        │
     ├────────────────────────►│                           │                       │                        │
     │                         │                           │                       │                        │
     │       VksplatViewportRenderer (per frame_slot)      │                       │                        │
     │                         │                           │                       │                        │
     │  ring_inputs[slot] ◄────┤  packDeviceInputs +       │                       │                        │
     │  CUDA → VkBuffer        │  cudaMemcpyAsync +        │                       │                        │
     │  (external memory)      │  cudaStreamSynchronize    │                       │                        │
     │                         │                           │                       │                        │
     │                         │  vksplat compute pipeline (12 dispatches)         │                        │
     │                         │  → output_image_ (VkImage, R8G8B8A8_UNORM,        │                        │
     │                         │     external memory) in SHADER_READ_ONLY layout   │                        │
     │                         │                           │                       │                        │
     │                         │                           │  vulkan_viewport_pass.record():                 │
     │                         │                           │   - bind viewport / quad                        │
     │                         │                           │   - clearViewport                               │
     │                         │                           │   - sample output_image_ (scene)                │
     │                         │                           │   - overlays / grid / pivots / vignette         │
     │                         │                           │  → swapchain image (color attachment)           │
     │                         │                           │                       │                        │
     │                         │                           │                       │  RmlUi draw lists:     │
     │                         │                           │                       │  - text / scissor      │
     │                         │                           │                       │  - blends onto         │
     │                         │                           │                       │    swapchain image     │
     │                         │                           │                       │                        │
     │                         │  vkQueueSubmit (graphics) + vkQueuePresentKHR ────┴───────────────────────►│
```

## Frame ring & synchronization

`VulkanContext::kFramesInFlight = 2` — double-buffered. All per-frame state is sized to this.

Per-frame timeline:

1. CPU side at frame start: `vkWaitForFences(in_flight_[currentFrameSlot()])` — guarantees the prior use of slot N is done before reuse
2. CUDA work for this frame is enqueued (Vulkan→CUDA wait happens implicitly via the CPU-side fence wait — when the CPU reaches this point, slot N's prior Vulkan reads are done, so CUDA can safely overwrite the slot)
3. After CUDA's `cudaMemcpyAsync` packs the inputs, CUDA signals a per-ring-slot **timeline semaphore** (Vulkan-exportable, imported into CUDA) on the same stream. The Vulkan submit registers `addFrameTimelineWait(timeline, value)` so the compute submit blocks on that signal at `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`. **No CPU stall** — the prior `cudaStreamSynchronize` is gone (see `vksplat_viewport_renderer.cpp::uploadInputs`).
4. Vulkan submit consumes the slot's CUDA-imported buffers, signals `in_flight_[slot]` on completion
5. `vkQueuePresentKHR` blocks on the render-finished semaphore

Three rings to be aware of, all keyed by `currentFrameSlot() % framesInFlight()`:

| Ring | Owner | Files |
|---|---|---|
| Vulkan command pool / cmd buffer / semaphores / fence | `VulkanContext` | `vulkan_context.{cpp,hpp}` |
| Per-frame viewport-pass resources (overlay buffers, descriptors) | `VulkanViewportPass::FrameResources` | `vulkan_viewport_pass.cpp` |
| CUDA-imported splat input buffers (xyz/rot/scales+opacs/sh) | `VksplatViewportRenderer::cuda_inputs_` | `vksplat_viewport_renderer.{cpp,hpp}` |

Rule of thumb: anything written by CUDA and read by Vulkan within a frame must be ring-buffered to the same depth as `framesInFlight()`, or the cross-API ordering breaks under multi-frame-in-flight.

## Hard requirements at device init

`VulkanContext::createDevice()` returns failure (refusing to start) if the device lacks any of:

- `VK_KHR_external_memory` + platform variant (`VK_KHR_external_memory_win32` / `VK_KHR_external_memory_fd`)
- `VK_KHR_external_semaphore` + platform variant
- Vulkan 1.3 mandatory features: `synchronization2`, `dynamicRendering`
- Vulkan 1.2 mandatory features: `timelineSemaphore`, `bufferDeviceAddress`

There is no CPU-staging fallback. CUDA + Vulkan interop is the only supported path.

After init, `VulkanContext` exposes opportunistic feature flags via `has*()` accessors:

- `hasDescriptorIndexing()` — Vulkan 1.2 core; `descriptorIndexing` + `runtimeDescriptorArray` etc.
- `hasPushDescriptor()` — `VK_KHR_push_descriptor`
- `hasShaderObject()` — `VK_EXT_shader_object`
- `hasExtendedDynamicState3()` — `VK_EXT_extended_dynamic_state3`
- `hasCooperativeMatrix()` — `VK_KHR_cooperative_matrix`
- `hasHostImageCopy()` — `VK_EXT_host_image_copy`

Each gates a Phase 3/4 modernization path. All six are present on current NVIDIA desktop drivers.

## Device-identity check

Multi-GPU systems (e.g. NVIDIA dGPU + Intel iGPU) can silently expose a different GPU to CUDA than to Vulkan. `WindowManager::init` calls `lfs::rendering::setExpectedVulkanDeviceUuid(vulkan_context_->deviceUUID())` after Vulkan init. The first CUDA/Vulkan interop import call (`CudaVulkanInterop::init` or `CudaVulkanBufferInterop::init`) lazily verifies `cudaGetDeviceProperties().uuid` matches; mismatch is a hard failure with a clear log naming both UUIDs.

## Image transitions

All viewport-pass image transitions go through `VulkanContext::imageBarriers()` — a `VulkanImageBarrierTracker` that:

- Stores `(layout, last_stage, last_access)` per `VkImage`
- On `transitionImage(cmd, image, aspect, new_layout)` emits a `VkImageMemoryBarrier2` with sync2 stage/access derived from the layout transition
- Skips the barrier if `state.layout == new_layout`

Direct `vkCmdPipelineBarrier` (sync1) is no longer used in the viewport / vksplat composition paths — the only remaining inline barrier is `vkCmdPipelineBarrier2` with `VkBufferMemoryBarrier2` for the pixel-state buffer in the splat→compose handoff (`vksplat_viewport_renderer.cpp::composePixelState`).

External images created via `VulkanContext::createExternalImage` must be registered with the tracker after creation and `forgetImage`'d before destruction; on swapchain reset the tracker is cleared, so a stale entry silently restarts at `VK_IMAGE_LAYOUT_UNDEFINED` (safe because that's "discard contents").

## File map

| Concern | File |
|---|---|
| Vulkan device + swapchain + frame ring | `src/visualizer/window/vulkan_context.{cpp,hpp}` |
| Image transition tracker (sync2) | `src/visualizer/window/vulkan_image_barrier_tracker.{cpp,hpp}` |
| CUDA↔Vulkan interop primitives | `src/rendering/cuda_vulkan_interop.{cpp,hpp,cu}` |
| vksplat compute rasterizer (forward only) | `src/rendering/rasterizer/vksplat_fwd/src/{gs_pipeline,gs_renderer,buffer,perf_timer}.cpp` |
| vksplat shader sources (Slang + GLSL) | `src/rendering/rasterizer/vksplat_fwd/shader/src/{slang,radix_sort}/` |
| vksplat shader build rules | `src/rendering/rasterizer/vksplat_fwd/CMakeLists.txt` (slangc + glslang) |
| Splat input packer (CPU + GPU) | `src/visualizer/rendering/vksplat_input_packer.{cpp,hpp}` |
| Per-frame ring + plug-in for vksplat | `src/visualizer/rendering/vksplat_viewport_renderer.{cpp,hpp}` |
| Viewport pass (scene compose + overlays) | `src/visualizer/rendering/passes/vulkan_viewport_pass.cpp` |
| Scene-image upload (external interop only) | `src/visualizer/rendering/passes/vulkan_scene_image_uploader.cpp` |
| RmlUi vk backend (project-vendored from RmlUi 6.2) | `src/visualizer/gui/rmlui/rmlui_vk_backend.cpp` |

## Adding a new render pass

1. If the pass needs frame-local state, allocate it inside `VulkanViewportPass::FrameResources` and grow the ring to `framesInFlight()`.
2. If the pass writes to a new external image, allocate via `VulkanContext::createExternalImage()` and register with `context.imageBarriers().registerImage(...)` immediately.
3. Wherever you change image layouts, call `context.imageBarriers().transitionImage(cmd, image, aspect, new_layout)` — never call `vkCmdPipelineBarrier`/`vkCmdPipelineBarrier2` inline for image transitions.
4. For buffer barriers within a pass, use `vkCmdPipelineBarrier2` with `VkBufferMemoryBarrier2` and explicit stage/access masks; the tracker doesn't manage buffers.
5. If the pass adds CUDA work, ring-buffer any cross-API state to `framesInFlight()` depth — see `VksplatViewportRenderer::cuda_inputs_` for the pattern.

## What got removed

The codebase used to support CPU-staging fallbacks for both the scene image and vksplat inputs. Those paths are gone. Failure to acquire external interop now hard-fails at `VulkanContext::createDevice()` rather than silently dropping to a slower path. There is no `--no-interop` CLI flag, no `LFS_NO_VK_CUDA_INTEROP` env var, no `LFS_VULKAN_NO_INTEROP_FALLBACK` build option.

vksplat_fwd no longer carries its standalone Python-module init path (own `VkInstance`/`VkPhysicalDevice`/`VkDevice` creation): `initializeExternal` is the only entry point and the visualizer always supplies the Vulkan + VMA handles.
