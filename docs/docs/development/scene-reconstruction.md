---
title: Scene reconstruction
---

# Scene reconstruction

Scene reconstruction is a viewport-only presentation stage. It changes the
internal resolution used to draw the current scene and reconstructs that image
at the viewport resolution. It never changes training tensors, model precision,
or data stored in a `.licht` project or exported splat file.

## Stable runtime contract

The scene reconstruction registry owns stable backend and preset identifiers.
The built-in registry exposes:

| Backend ID | UI label | Presets | Temporal inputs |
| --- | --- | --- | --- |
| `native` | Off | `native` (1.0) | None |
| `spatial` | Spatial | `quality` (0.75), `balanced` (0.67), `performance` (0.50) | None |
| `temporal` | Temporal | `quality` (0.75), `balanced` (0.67), `performance` (0.50) | Depth, motion, jitter and per-view color/depth history |

The renderer's existing `render_scale` remains the base scene scale. A selected
backend's input multiplier is applied independently, so reconstruction does not
rewrite the base control. Native presentation ignores the multiplier.

The requested backend, effective backend, fallback state, and runtime readiness
are distinct. The built-in spatial Vulkan pipeline is created lazily on first
use. If creation fails, the frame is presented through the native path and the
transition is logged; the saved request is not silently rewritten.

## Spatial path

Spatial reconstruction reuses the existing scene image and descriptor. The
viewport pass selects a fullscreen fragment pipeline that performs bounded
five-tap sharpening while sampling the reduced-resolution image. There is no
extra intermediate image, queue submission, CUDA conversion, history buffer,
motion-vector producer, jitter, or CPU fallback.

Split view uses the same sampling rule inside its existing composite shader.
Its content rectangle is transformed from the renderer coordinate extent to
the framebuffer extent before compositing, so letterboxing and the split
divider remain aligned at reduced internal resolutions.

## Temporal path

Temporal reconstruction owns independent history for the main viewport and
both supported split-view panels. It derives motion from the current and
previous camera projections plus the VkSplat depth image, rejects disoccluded
history with current and previous depth, and resolves into a full-resolution
Vulkan image before presentation. Startup and explicit backend transitions may
produce one native warm-up frame while the first paired color/depth input is
established; invalid contracts and pipeline failures remain observable errors.

The temporal path is available for the regular and training viewports,
including orthographic projection, Independent Dual split view, and PLY
comparison. Orthographic frames explicitly declare that no perspective jitter
was applied while retaining motion and depth history. Ground-truth comparisons
deliberately preserve their reference image. Equirectangular projection and
appearance-corrected readback currently remain native because their projection
or ownership contracts are not equivalent to the Vulkan temporal path. A split
result is presented only when both panel resolves succeed; otherwise both
panels fall back together.

## Persistence and safe mode

The selected backend and the last valid preset for each backend are user-global
preferences in `config/preferences.json`; they are not project state. Invalid
backend or preset identifiers fall back to the registry defaults. Safe mode
starts native presentation, disables the two Preferences reconstruction
selects, and neither reads nor writes the preference file; Python and plugins
can still change the live setting for that session.

Vendor backends use the same registry and effective-state contract while
keeping their SDK resources and synchronization lifecycles isolated from the
built-in Native, Spatial, and Temporal paths.
