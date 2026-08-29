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
| `nvidia-dlss` | NVIDIA DLSS (optional) | `quality` (2/3), `balanced` (0.58), `performance` (0.50) | Depth, motion and jitter; history is owned by the NGX feature |

The renderer's existing `render_scale` remains the base scene scale. A selected
backend's input multiplier is applied independently, so reconstruction does not
rewrite the base control. Native presentation ignores the multiplier.
For NVIDIA DLSS, the table records the catalog's bootstrap values only. Once
NGX is initialized, its optimal-settings query selects the exact render extent
for the current output size and preset; that result is cached until one of
those inputs changes.

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
including orthographic projection, Independent Dual split view, and PLY
comparison. Orthographic frames explicitly declare that no perspective jitter
was applied while retaining motion and depth history. Ground-truth comparisons
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

## Optional NVIDIA DLSS plugin

NVIDIA DLSS is isolated behind the versioned
`scene_upscaler_plugin_api.h` C ABI. The main executable and
`lfs_visualizer` do not link against the NVIDIA SDK. A build that enables DLSS
produces `lfs_scene_upscaler_nvidia_dlss.dll` on Windows or
`liblfs_scene_upscaler_nvidia_dlss.so` on Linux and places the NVIDIA runtime
beside it under `scene_upscalers/nvidia`. The backend is registered only when
that external plugin is present and its ABI and identifier validate.

The host opens the plugin during Vulkan bootstrap because NGX must declare its
required Vulkan instance and device extensions before those objects are
created. NGX runtime initialization, capability queries, per-view feature
creation and full-resolution output allocations remain lazy and occur only
after the user selects NVIDIA DLSS. Safe mode does not load optional
scene-reconstruction plugins. Dynamic libraries are opened from exact
application-relative plugin paths; arbitrary system search-path discovery is
not used.

If the LichtFeld-owned plugin module is absent, the optional backend is omitted
from the runtime catalog and Native remains selected. If the plugin is valid but
the separately staged NVIDIA runtime is missing or unsupported, the request is
retained, presentation falls back atomically to Native, and diagnostics report
both the NGX failure and the effective Native fallback. The failure is latched
instead of being retried every frame; selecting Native and then NVIDIA DLSS is
the explicit retry action after correcting the installation.

DLSS consumes the same reviewed temporal frame contract as the built-in
Temporal backend: unjittered current-to-previous pixel motion, the exact jitter
applied to the rendered color image, scene/camera/backend reset reasons and
independent main/left/right view identity. Switching between Temporal and DLSS
changes the history key. A failed DLSS initialization or evaluation is latched
to native presentation instead of being retried every frame; selecting Native
before selecting DLSS again is the explicit retry action. Split output is
transactional, so both panels resolve through DLSS or both remain native.

VkSplat publishes positive linear view depth, while NGX expects raster depth.
An isolated compute pass therefore converts each valid render subregion to a
non-inverted `R32_SFLOAT` Vulkan depth image (`near = 0`, `far = 1`) before
evaluation. Color remains perceptually encoded LDR `RGBA8`; motion remains
low-resolution `RG16F`, current-to-previous, top-left pixel motion without
jitter. The exact applied projection jitter is supplied separately in render
pixel units. Output extents below NGX's 32-by-32 minimum use a temporary native
presentation and do not latch a backend failure, so resizing back restores the
requested DLSS path without a manual retry.

Ordinary developer builds leave the plugin disabled. Obtain an SDK checkout
from the official [NVIDIA/DLSS repository](https://github.com/NVIDIA/DLSS),
including its Git LFS objects, and enable the plugin explicitly:

```sh
git clone https://github.com/NVIDIA/DLSS external/nvidia-dlss-sdk
cmake -S . -B build -DLFS_ENABLE_NVIDIA_DLSS=ON -DLFS_NVIDIA_DLSS_ROOT=/path/to/NVIDIA-DLSS-SDK
```

Both Debug and Release SDK runtimes are supported in ordinary opt-in developer
builds. Portable configurations force `LFS_ENABLE_NVIDIA_DLSS=ON` and therefore
require the SDK at `external/nvidia-dlss-sdk` unless `LFS_NVIDIA_DLSS_ROOT` is
set explicitly.
CMake never downloads the SDK or accepts its license for a developer build.
Anyone redistributing a portable package must satisfy NVIDIA's SDK and runtime
redistribution terms.
The Windows nightly portable workflow is the controlled exception to local SDK
discovery: it checks out a pinned `NVIDIA/DLSS` revision with Git LFS before
configuration, then stages the external LichtFeld plugin and matching vendor
runtime in the package. Missing portable SDK artifacts are fatal rather than
silently producing a package without the advertised backend.

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
