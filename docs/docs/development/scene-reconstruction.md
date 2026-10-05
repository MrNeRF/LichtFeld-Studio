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
| `metalfx_spatial` | Apple MetalFX Spatial (native Metal) | `quality` (2/3), `balanced` (1/1.7), `performance` (0.50) | None |
| `metalfx_temporal` | Apple MetalFX Temporal (native Metal) | `quality` (2/3), `balanced` (1/1.7), `performance` (0.50) | Depth, motion and jitter; history is owned by the MetalFX feature |
| `amd-fsr3` | AMD FSR 3.1 (optional) | `quality` (2/3), `balanced` (1/1.7), `performance` (0.50) | Depth, motion and jitter; history is owned by the FidelityFX feature |

The renderer's existing `render_scale` remains the base scene scale. A selected
backend's input multiplier is applied independently, so reconstruction does not
rewrite the base control. Native presentation ignores the multiplier.
For optional provider modules, the table records the catalog's bootstrap
values only. Once the vendor runtime is initialized, its optimal-settings query
selects the exact render extent for the current output size and preset; that
result is cached until one of those inputs changes.

The requested backend, effective backend, fallback state, and runtime readiness
are distinct. The built-in spatial Vulkan pipeline is created lazily on first
use. If creation fails, the frame is presented through the native path and the
transition is logged; the saved request is not silently rewritten.
Mode-ineligible temporal requests report `unsupported_mode`, while runtime
failures continue to report `runtime_unavailable`.

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

Temporal reconstruction owns independent history for each 3D view and its
comparison panels. It derives motion from the current and
previous camera projections plus the VkSplat depth image, rejects disoccluded
history with current and previous depth, and resolves into a full-resolution
Vulkan image before presentation. Startup and explicit backend transitions may
produce one native warm-up frame while the first paired color/depth input is
established; invalid contracts and pipeline failures remain observable errors.

Projection jitter is converted to the render image's pixel convention before
resolve. Current color and motion are sampled on the jittered render grid.
Motion vectors are jitter-free: they address the previous stable output
coordinates, and the previous jitter is added only to look up history depth
on the previous jittered render grid. The
eight-frame warm-up uses uniform sample accumulation capped by the selected
preset's history weight. Deterministic synthetic regressions compare PSNR and
SSIM against a high-resolution reference and exercise moving-history
reprojection to detect blur and ghosting regressions.

The temporal path is available for the regular and training viewports,
including orthographic projection and PLY comparison. Orthographic frames
explicitly declare that no perspective jitter was applied while retaining
motion and depth history. Ground-truth comparisons
deliberately preserve their reference image. Equirectangular projection and
appearance-corrected readback currently remain native because their projection
or ownership contracts are not equivalent to the Vulkan temporal path. A split
result is presented only when both panel resolves succeed; otherwise both
panels fall back together.

When the selected backend leaves Temporal, the viewport first retires submitted
frames and then releases the per-view color and depth history allocations. The
immutable compute-pipeline state remains available for a later Temporal
selection, avoiding persistent history VRAM without paying full pipeline
creation cost on every backend switch.

## Native Metal window presentation

The native compositor produces a final RGBA8 tensor. `MetalGraphicsContext`
presents it with a fullscreen-triangle render pass: the fragment shader reads
the tensor buffer and writes to the `CAMetalLayer` drawable as a color attachment.
The layer therefore keeps `framebufferOnly = YES`; tensor compute stays in buffers.
Apple specifies that
[framebuffer-only textures](https://developer.apple.com/documentation/metal/mtltexture/isframebufferonly)
can only be render-pass attachments and cannot be bound as texture arguments to
compute, blit, or render encoders. A compute-based final conversion would require
[CAMetalLayer.framebufferOnly](https://developer.apple.com/documentation/quartzcore/cametallayer/framebufferonly)
to be disabled. The render-pass presentation avoids that incompatible usage.

Validate native window presentation with Metal API Validation and the displayed
window. The internal window-capture API reads the composited tensor before the
drawable conversion: a correct capture does not prove that screen presentation
is correct. The incompatible framebuffer-only compute path can show a solid
magenta window despite a correct internal capture.

## Native Metal reconstruction

Apple builds with Metal graphics expose `metalfx_spatial` and
`metalfx_temporal` only when the system GPU supports the corresponding MetalFX
scaler. These are framework-backed built-ins, not Vulkan provider modules.
They do not require the FidelityFX SDK, CUDA, MoltenVK, or Vulkan headers.
Vulkan graphics builds retain FSR through their existing provider ABI; they do
not advertise MetalFX. Preferences and Python/MCP use the same registry and
requested/effective/fallback contract for both graphics APIs.

The existing `spatial` and `temporal` modes execute their common Slang kernels
on Metal tensors. They retain the original presets and reconstruction math.
Temporal reconstruction supports perspective and orthographic Gaussian views
and PLY comparison, with independent main/left/right histories for each
viewport owner. Calibrated/cropped panels derive motion from crop-local
intrinsics. Ground-truth comparison, panorama, appearance correction, and
point-cloud temporal requests report `unsupported_mode` and remain native.
MetalFX Spatial also reconstructs regular Gaussian and point-cloud views and
PLY panels. GT comparison keeps its existing full-resolution reference/display
path; no MetalFX work is applied to the reference image.

MetalFX uses RGBA16Float color/output textures, R32Float non-reversed raster
depth, and RG32Float current-to-previous motion in top-left render pixels.
The shared motion kernel uses unjittered camera matrices. Raster jitter is
passed separately in render pixels; macOS 26 does not need the newer
jittered-motion descriptor option. View-space depth is converted on the GPU.
MetalFX writes opaque alpha, so output conversion restores scene coverage by
bilinearly sampling the current color alpha at jitter-corrected coordinates.
Coverage is not reconstructed by MetalFX's internal temporal history.

Texture conversion and scaler execution are ordered through
`MetalTensorReader::submitWrites`. Input snapshots, output storage and the
feature are retained through command completion; resize and release do not
wait on the CPU. Tensor consumers wait through the existing GPU timeline.
A feature is recreated when extents change. Camera cuts, scene/backend/preset
changes, projection/crop changes and explicit resets invalidate only the
corresponding temporal history. Ineligible modes release history. Encoding
failures retain the last complete frame and then fall back to native; changing
the backend or preset permits a retry. Asynchronous GPU write failures use the
existing sticky tensor failure contract.

This remains a viewport stage. The existing offline video preflight contract
does not execute FSR or MetalFX in the export worker.

GPU regression target: `metal_scene_upscaler_contracts`. It covers native
producer/consumer ordering, alpha, independent panels/owners, crop calibration,
jitter, reset reasons, resize, destruction with pending work and invalid inputs.
The same target includes shared tracker/coordinator/registry contracts and
basic Metal spatial/temporal checks. Existing `tensor_rasterizer_contracts`
provide CPU-oracle and image-convergence coverage for the basic kernels.

## Optional reconstruction providers

All optional modules, including AMD FSR 3.1, use the same generic discovery and
loading mechanism. Multiple providers can coexist in different subdirectories
of `<executable-directory>/scene_upscalers/` or `<library-directory>/scene_upscalers/`.
The executable-relative folder has priority when both roots contain the same
subdirectory. Discovery runs once at startup; restart after changing installed
modules. The application never searches arbitrary system library paths.

Each provider folder contains its platform module and three text metadata files:

- `provider-id.txt`: stable lowercase identifier, up to 128 letters, digits,
  underscores or hyphens. Built-in IDs are reserved. The ID must match the ABI.
- `provider-module.txt`: module stem without a path or extension, up to 128
  ASCII letters, digits, underscores or hyphens. Platform prefixes and extensions
  are applied by the host.
- `provider-presets.txt`: three finite input scales in quality, balanced,
  performance order, using decimal points; each must be greater than zero and
  at most one. Runtime optimal-settings queries still determine exact extents.

The UI reads the provider's `display_name` from the validated C ABI; it does not
infer a name from the folder or translate an application-owned vendor label.
Python and MCP catalog records expose the same `display_name`. Preferences
retain the provider's stable string ID, independent of discovery order.
Invalid metadata and duplicate IDs in different folders are excluded. Missing
or incompatible modules are omitted from the available catalog. Safe mode reads
metadata but does not execute provider modules. An unavailable requested backend
presents Native and retains its remembered preference for reinstallation.

AMD FSR 3.1 remains optional at build time. Its existing build/install rules now
stage these metadata files beside its module in `scene_upscalers/amd/`; when it
is not built and no compatible installed module exists, it is absent from the
catalog. Existing module installations require the new metadata files.

The host validates the complete versioned ABI and queries Vulkan extensions
before device creation. Runtime initialization and feature allocation remain
lazy. Depth, motion, jitter, reset reasons and independent view identities keep
the shared contracts. Both split panels resolve successfully or both present
Native; an evaluation failure latches the existing transactional fallback.

Separately built providers carry their own installation instructions and
licenses. Primary portable staging rejects extra provider directories before
installation; use a clean staging prefix. FSR is the explicitly packaged
optional provider. Locally installed modules belong to the user's installation.

The focused loader tests compile real fixture modules and the production loader:

```sh
cmake -S tests/scene_upscaler_loader -B build-loader-tests -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/gtest
cmake --build build-loader-tests
ctest --test-dir build-loader-tests --output-on-failure
```

A Vulkan SDK and GoogleTest are required. These tests do not evaluate GPU images.

## Plugin host

Every optional vendor backend goes through the same host code; only the plugin
module is vendor specific. `rendering/scene_upscaler_plugin.cpp` lists each
plugin's backend, ABI identifier, directory under `scene_upscalers/`, module
stem and cache folder, and loads it through the C ABI described above.
`VulkanScenePluginPipeline` records motion, plugin depth and evaluation for
whichever plugin is selected, and the viewport, render manager and Vulkan
bootstrap iterate the plugin list instead of naming vendors. Adding a backend
means adding a plugin module, one row in that list and one registry descriptor.

Vendor differences are declared by the plugin, not by the host:

- `capabilities` advertises `DYNAMIC_VIEW_IDS` (one feature per independent
  3D view) and `REQUIRES_PERSPECTIVE` (orthographic views stay native and
  report `unsupported_mode`).
- `optimal_settings` returns the exact render extent for an output size and
  preset and, through the optional `jitter_phase_count` tail, the jitter
  sequence length the vendor expects. The host wraps its Halton sequence and
  sizes the static-scene settle burst to that length; 0 keeps the default.
- Every evaluation carries the frame's near plane, far plane and vertical field
  of view in an optional tail, for plugins that linearize depth themselves.

The host always supplies jitter-free motion vectors
(`motion_vectors_include_jitter = 0`) and the exact applied jitter, so plugins
translate that one convention into their SDK's. When a frame is presented
again without a newly published color/depth generation, the pipeline reuses the
last resolved output instead of feeding the same sample into vendor history
twice.

## Optional AMD FSR 3.1 plugin

The AMD FSR 3.1 plugin builds `lfs_scene_upscaler_amd_fsr3` under
`scene_upscalers/amd` (`.dll`, `.so`, or `.dylib` on macOS, where it runs on
MoltenVK). Only that module links the FidelityFX SDK. It consumes the same LDR
`RGBA8` color, `R32_SFLOAT` raster depth and `RG16F` motion as external reconstruction plugins, uses a
neutral pre-exposure and view-space scale, and disables the optional RCAS
sharpening pass, which hardens splat edges. Reactive and transparency masks
remain absent until the renderer can publish semantically correct material
signals. Up to eight views may hold an FSR feature at once.

Ordinary developer builds leave the plugin disabled. Use the
[AMD FidelityFX SDK v1.1.4 release](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK/releases/tag/v1.1.4)
and enable it explicitly:

```sh
git clone --branch v1.1.4 https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK external/fidelityfx-sdk
cmake -S . -B build -DLFS_ENABLE_AMD_FSR3=ON -DLFS_AMD_FSR3_ROOT=external/fidelityfx-sdk
```

When prebuilt libraries are absent, `LFS_AMD_FSR3_BUILD_SDK=ON` builds the FSR
upscaler and Vulkan backend from an isolated copy of the SDK. Windows uses the
SDK's own project. Linux and macOS use `src/scene_upscalers/amd_fsr3/posix`,
which drives the SDK shader compiler with vcpkg's glslang (built with its
SPIR-V optimizer) and applies small SDK v1.1.4 compatibility patches. On macOS,
build one architecture at a time. The SDK license is staged next to the plugin
and installed with the package licenses.

## Persistence and safe mode

The selected backend and the last valid preset for each backend are user-global
preferences in `config/preferences.json`; they are not project state. Invalid
backend or preset identifiers fall back to the registry defaults. Safe mode
starts native presentation, disables the two Preferences reconstruction
selects, and neither reads nor writes the preference file; Python and plugins
can still change the live setting for that session.

## Integration benchmark

`tools/benchmark_scene_reconstruction.py` compares every registered backend and
backend-specific preset through the production viewport path. It discovers the
catalog at runtime, so newly registered backends participate without a
hard-coded benchmark list.

Before running it, start LichtFeld Studio, load a representative scene, stop or
pause training and any animation, size the live viewport to the resolution being
measured, and enable the MCP server. Then run:

```sh
uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --frames 128 --quality-frames 8 --warmup-frames 8 --performance-rounds 4
```

The script has a complete command reference and examples:

```sh
uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --help
```

The script runs two globally isolated phases over deterministic camera orbits.
The complete performance phase for every case finishes before the first PNG
capture is requested. Performance samples are divided across deterministic
rounds whose backend order is cyclically rotated and alternately reversed. This
reduces bias from fixed case order, thermal drift, and the preceding backend.
`--frames` is the total measured frame count per case across all rounds; adding
rounds does not multiply that measured count. Every case is warmed independently
in each round.

The performance phase waits for production viewport frames without capturing
them. Its median, 10% trimmed mean, standard deviation, range, and p95 latency
therefore exclude viewport readback, PNG encoding, Base64 conversion, and HTTP
response transfer. The reported viewport FPS is derived from the in-application
camera-to-post-render median latency. This scope still covers GUI scheduling and
command recording and is not an isolated GPU timestamp.

The quality pass performs a smaller number of full-resolution captures. For
each pose it waits until `render.reconstruction.status` reports
`convergence_remaining == 0`, then calls `render.capture` with `presented:
true` to crop the reconstructed viewport from the presented window. The
`presented` flag is opt-in (default false) so other `render.capture` callers
keep the internal raster. This prevents the benchmark from reading a
pre-reconstruction raster input or an intermediate convergence frame. It reports aggregate PSNR
and global RGB SSIM against matching Native frames, and reports PNG capture
latency separately rather than presenting it as renderer FPS. `--frames`
controls the performance sample count and `--quality-frames` controls the sparse
image-comparison count.

Results are written to `build/scene-reconstruction-benchmark.json` by default.
Schema version 4 records the global phase boundary, round order, raw performance
samples, per-round summaries, aggregate statistics, the presented-viewport
capture source, convergence policy, phase-specific effective backend state, and
explicit fallback information. The original camera, backend, and preset are
restored when the run completes or fails. Progress and per-phase ETA are printed
while the benchmark runs. The JSON report is replaced atomically after every
completed case, so partial measurements remain available if a later case or
state restoration fails. The deterministic CPU quality regressions in the test
suite remain the smaller algorithm-level guard; this benchmark validates the
actual renderer with the scene selected by the developer.

The benchmark also performs a preflight before changing the camera or renderer.
It requires a visible, non-empty Gaussian scene and records the active projection,
raster backend, appearance-correction state, and split-view mode. A
Temporal-incompatible viewport fails immediately with an explicit reason. If a
backend still falls back after warm-up, its diagnostic is retained in the partial
report and the run fails instead of silently publishing a complete benchmark
with that backend skipped.

### Command reference

| Option | Meaning |
| --- | --- |
| `--url URL` | MCP HTTP endpoint. Default: `http://127.0.0.1:45677/mcp`. |
| `--width PIXELS` | Expected live viewport width. `0` discovers it from `camera.get`. Width and height must both be automatic or both explicit. |
| `--height PIXELS` | Expected live viewport height. `0` discovers it from `camera.get`. |
| `--frames COUNT` | Total measured performance frames per case, divided across the performance rounds. Default: `24`. |
| `--quality-frames COUNT` | Sparse full-resolution PNG comparison frames per case. Default: `8`. |
| `--warmup-frames COUNT` | Untimed warm-up frames for each case in each performance round, and for each quality case. Default: `8`. |
| `--performance-rounds COUNT` | Rotated/reversed performance rounds used to reduce order bias. Cannot exceed `--frames`. Default: `3`. |
| `--orbit-degrees DEGREES` | Total span of the deterministic camera orbit. Default: `8`. |
| `--backends IDS` | Optional comma-separated backend IDs, for example `native,spatial,temporal`. Native is always included as the quality reference. |
| `--timeout SECONDS` | Timeout for each MCP HTTP request and for convergence at each quality pose. Default: `60`. |
| `--output PATH` | Atomic JSON report path. Default: `build/scene-reconstruction-benchmark.json`. |

Useful invocations:

```sh
# Short production-path smoke run
uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --frames 16 --quality-frames 4 --warmup-frames 4 --performance-rounds 2

# Longer comparison with more protection against ordering drift
uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --frames 128 --quality-frames 8 --warmup-frames 8 --performance-rounds 4

# Restrict the catalog while retaining the Native quality reference
uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --backends native,spatial,temporal

# Select a different MCP endpoint and report path
uv run --no-project --with numpy --with pillow python tools/benchmark_scene_reconstruction.py --url http://127.0.0.1:45677/mcp --output build/reconstruction.json
```
