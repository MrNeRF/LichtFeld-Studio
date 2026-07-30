# Vulkan Viewer Migration

This migration replaces the viewer's OpenGL integration with Vulkan in stages. The viewer, RmlUi, Python UI texture path, thumbnails, split-view UI cache, CUDA texture upload path, app-facing Gaussian/point-cloud raster engine, scene guide overlays, custom transform gizmos, HDRI environment backgrounds, mesh compositor, and mesh-to-splat conversion are Vulkan/tensor-backed. The default build no longer resolves or links OpenGL or glad. Dear ImGui / ImPlot are fully removed; RmlUi is the only GUI stack (see PR #1395 and `docs/imgui_inventory.md`).

## Dependency Baseline

Vulkan is resolved through vcpkg so the Windows and portable builds use the same dependency source:

- `vulkan` provides `Vulkan::Vulkan` and the loader.
- `vulkan-memory-allocator` is available for image and buffer lifetime management.
- `volk` is available if the renderer switches to generated dispatch tables.
- RmlUi is the only GUI stack (`rmlui` with `svg`); there is no ImGui/ImPlot vcpkg feature or CMake link. Viewport immediate drawing and pie-menu icons use `ScreenOverlayRenderer` / `IconCache` rather than a legacy immediate-mode UI backend.

`ENABLE_VULKAN_VIEWER` controls whether CMake resolves and links the Vulkan viewer dependencies. The application probes the Vulkan loader at startup so portable builds report missing loader/runtime problems early in logs.
The legacy OpenGL/glad renderer target has been replaced by `lfs_rendering_tensor`; the `lfs_rendering` CMake target name is now an interface compatibility target that forwards to the tensor renderer. Root CMake no longer resolves or links OpenGL or glad for the app, viewer, Python module, MCP server, tests, or benchmarks.

## Completed Viewer Migration

1. Replace `SDL_WINDOW_OPENGL` and `SDL_GLContext` in `WindowManager` with `SDL_WINDOW_VULKAN`, `SDL_Vulkan_GetInstanceExtensions`, and `SDL_Vulkan_CreateSurface`.
2. Introduce a viewer-owned Vulkan context: instance, surface, physical device selection, device, queues, swapchain, command pools, frame synchronization, and resize handling.
3. Replace viewer-owned UI textures with backend-neutral Vulkan image/image-view/sampler ownership.
4. Replace CUDA-OpenGL UI texture interop with Vulkan external memory import where available, with a staging fallback.
5. Replace the UI backend with the Vulkan renderer initialized from the viewer Vulkan context.
6. Replace the copied RmlUi GL3 backend with the Vulkan backend using the viewer command buffer and swapchain.
7. Remove `ENABLE_CUDA_GL_INTEROP` and the retired CUDA-GL framebuffer/texture path.
8. Move the app, visualizer, Python module, and MCP server off `lfs_rendering`; verified Debug binaries do not link `libGL`, `libGLU`, `libOpenGL`, GLX/EGL, or glad.
9. Route selection screen-position and hovered-Gaussian queries through the tensor raster engine, so editor selection remains available without initializing the legacy renderer.
10. Restore Vulkan viewport parity paths for point-cloud mode, raw point-cloud scenes, independent split view, PLY comparison, and GT comparison through tensor rendering/composition instead of GL textures.
11. Add tensor-backed frame handles for legacy GPU-frame call sites and a software mesh compositor for the Vulkan viewport/video-export path, so mesh overlays no longer require app-side OpenGL linkage.
12. Restore HDRI environment backgrounds in tensor composition for viewport and video export.
13. Restore grid, coordinate axes, pivot, viewport vignette, crop/depth/ellipsoid guides, and camera frustum/image guides in the Vulkan overlay path, with raster-only camera frustum picking.
14. Replace the GL Mesh2Splat converter with a CPU/tensor surface sampler so mesh-to-splat conversion no longer requires an OpenGL context.
15. Delete the orphaned legacy renderer source files, copied GLSL shader tree, and renderer-private font assets from the default source tree.
16. Remove dead public renderer entry points for direct mesh texture rendering, GPU-frame presentation, and GL-style zero-copy depth texture handoff.
17. Keep transform, crop-box, ellipsoid, sequencer-keyframe, and viewport-orientation manipulation on the project-owned custom gizmo modules; remove the renderer viewport-gizmo draw/hit-test fallback so there is no alternate gizmo route.
18. Remove Dear ImGui / ImPlot entirely (PR #1395). Python plugin panels use the Rml im-mode bridge; viewport immediate drawing uses `ScreenOverlayRenderer`; Zep uses `ZepDisplay_Rml`.

## Remaining Code Migration

1. ~~Replace the remaining legacy immediate-mode UI draw-list/panel transport used by custom gizmos and native overlays with the target Rml/Vulkan-native overlay layer.~~ **Done (PR #1395):** gizmo/pie-menu drawing and viewport-overlay Python `draw_*` / `draw_window_*` calls go through `ScreenOverlayRenderer`; panels go through RmlUi / `RmlImModeLayout`.
2. Replace tensor/software compatibility renderers with true Vulkan pipelines where needed for performance and shader-identical output, especially high-volume mesh/PBR work and mesh-to-splat conversion.
3. Retire remaining compatibility names such as `GpuFrame`/`TextureHandle` once all callers use Vulkan image handles or tensor-only transfer points directly.
4. Convert any retained GLSL shaders outside the deleted legacy renderer tree to SPIR-V at build time and install the compiled shader assets in portable builds.
