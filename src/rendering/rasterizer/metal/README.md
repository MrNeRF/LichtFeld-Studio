<!-- SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
     SPDX-License-Identifier: GPL-3.0-or-later -->

# Native Metal viewer backend

The native Metal rasterizer consumes resident LichtFeld tensors, including Q16
SH storage. On macOS it can be selected explicitly under Preferences; Automatic
continues to use Vulkan. The desktop UI, grid and gizmos still use the existing
Vulkan compositor, sampling shared Metal color/depth textures on the same GPU.
No full-frame CPU readback occurs during interactive native presentation.

Startup logs distinguish the requested viewer API, selected tensor backend and
active Vulkan compositor before any scene opens. Runtime logs announce actual
viewer routes and fallbacks after successful submission; tensor preference
changes explicitly require restart. Finder drag-and-drop uses SDL's native Cocoa
destination and the existing single import path, including hover/cancel feedback.
`MacNativeDragDropContracts` exercises that event bridge without a GPU in macOS CI;
actual Finder interaction remains a manual UI check.

The shared desktop status bar shows compact Renderer and Tensors badges beside
FPS on all platforms. Renderer telemetry comes from published frame metadata,
including mixed split frames and software point-cloud panels, rather than from
the saved preference. The tensor badge reports the current process backend.
Compact R and T role markers sit to the right of FPS and keep the same
appearance on hover. Localized tooltips expand the role names and explain
independent routing and restart semantics. With no scene output, R reports
the active Vulkan desktop compositor and explains this in the tooltip. CPU visualizer
contracts cover metadata propagation/reset and status-bar sizing without a GPU.

## Implemented and GPU-tested

`SplatPreprocessor` encodes a GPU projection into the caller's command buffer.
It consumes existing Metal buffers/slices, including their offsets. No upload,
CPU sort, readback, queue wait or full-scene SH decompression occurs here.
Callers can precompile cached specializations with `prepare` before interaction.

- Canonical float32, swizzled float32, swizzled IEEE float16, and **integer Q16**
  SH-rest layouts. Active degree and resident layout degree are separate.
- Q16: 32-row cell swizzle, three uint16 cells per coefficient without float4
  padding, one float2 bound per 256 primitives. Decode uses the existing codec's
  fused multiply-add. RAD's signed-byte/page-frame pool is a **different layout**
  and is not accepted as Q16.
- SH0 specialization does not bind/read SH rest or bounds; points specialization
  does not bind/read Gaussian scales or rotations.
- Affine object transforms, positive-view-Z perspective projection, normalized
  source quaternion, covariance projection, contribution bounds, near/far and
  deleted-mask rejection. Screen positions/depth use the requested render extent.
- Caller-owned projection output and input lifetimes. Metal retains resources
  of normal command buffers; producers must be synchronized before encoding.
  Input mutation or scratch reuse before completion is not allowed. No borrowed
  raw tensor pointer is converted into a host copy here.
- Per-object transforms, visibility and active SH limit without rewriting source
  positions; invalid object indices are culled. Orthographic projection, Gaussian,
  point and disc specializations, and mip covariance/opacity compensation.
- Native 3DGUT uses seven unscented-transform samples for projection and retained
  inverse-Gaussian geometry for per-pixel 3D ray alpha/depth evaluation.
  Perspective, orthographic and equirectangular, affine/Q16/SH, mip, crop, selection and center
  markers share the existing tensor/filter contracts; interactive work performs
  no CPU geometry copies, sorting or per-frame readback.

`TileRasterizer` consumes the resident projected splats:

- GPU tile counts and hierarchical uint64 exclusive scan. SIMD reductions use
  16-bit limbs because MSL 2.4 does not provide ulong SIMD reductions.
- Bounded instance generation, stable GPU radix sort by tile and full float32
  positive radial distance squared, tile ranges, front-to-back composition and transmittance exit.
  The radix sort reuses the algorithm in `training/kernels/metal/fast_raster.metal`,
  without depending on the trainer. It only sorts the tile-ID bytes actually used.
- Color/alpha, weighted depth, first-contributor depth/ID and median depth.
  Equal-depth ties preserve source order. Median depth uses 1e10 for no crossing.
- Explicit frame reservations with retained GPU lifetimes. Encoding performs no
  per-frame storage allocation, CPU sort or instance-count readback.
- Typed instance-capacity overflow with the complete required count. An overflow
  frame contains background only; the caller must retry or report it, not publish
  it as a successfully rendered scene.

Frame owners must submit encoded command buffers and retain reservations until
all downstream consumers finish. `busy()` covers the raster command, not an
unrelated queue subsequently sampling its textures. Do not reuse a reservation
while a compositor still owns its output. A discarded/uncommitted submission must
discard its reservation too.

`rendering/viewer_backend.hpp` separates requested and effective GPU API from the
tensor backend and the 3DGS/3DGUT algorithm. `UserPreferences` persists the request
under `viewer_backend`, defaults missing/invalid values to automatic, and does not
rewrite the saved request when a backend is unavailable. The resolver reports a
fallback reason. Desktop routing and the preferences UI are connected.

The native desktop adapter supports Studio 3DGS and 3DGUT color/depth, perspective and
orthographic and equirectangular views, resident float32/half geometry, Q16/half/float SH storage,
node transforms/visibility/SH limits, deletion, crop boxes and ellipsoids,
screen depth windows, dimming, committed/preview selection, brushes, node flash,
rings and center markers. SH evaluation removes object scale from its direction
in the same way as Vulkan. A single object can omit per-primitive object indices.
Point-cloud display uses a native hardware point pipeline and depth testing,
avoiding Gaussian tile expansion and sorting; its synchronous completion matches
the current desktop point-renderer contract.

Equirectangular 3DGUT keeps full-camera dimensions/origin when exporting tiles,
unwraps sigma points across longitude, admits both view-Z hemispheres by radial
distance, and wraps physical pixel support before tile conversion. Odd widths
do not wrap at the padded GPU-grid period. Camera rays are computed once per
live pixel and reused across Gaussian batches. The spherical UT accumulates
relative offsets to avoid cancellation from its large negative central weight.
The viewer's fixed raster near threshold is derived from the existing Vulkan
shader configuration; desktop projection near/far planes do not incorrectly
cull native Gaussian input. Separate close-range parity cases cover this.

Main, split-left, split-right and preview outputs have independent reservations.
Expected-depth captures normalize only valid forward contributor weights within
the requested far range. Invalid GUT rays retain their visible opacity but cannot
contaminate that depth average. The valid weight reuses a depth texture channel
only in expected capture; normal first-contributor and median channels retain
their contracts, without an extra texture/allocation. Analytic GPU cases cover
mixed valid/invalid contributors, far cutoff and failed reservation validation.
Native median and normalized alpha-weighted depth capture, deterministic export,
synchronous reads and asynchronous color/depth tickets use the actual native
output. High-resolution export dilation and covariance caps remain calibrated
to source viewport pixels. Temporal jitter is excluded from scene-refinement
invalidation so reconstruction can finish converging. Offline captures check completion and retry overflow before publication.
Abandoned tickets release their destination pointer while retaining GPU staging.
Native ticket identities remain unique across renderer reset and recreation.
Resource release waits for both native producers and graphics consumers. A
failed encode with no submitted command cannot expose uninitialized GPU status.

New reservations are admitted conservatively against 80% of the device's
recommended working set, accounting for allocations already on the shared
device. This guards large growth before allocation; it is not an eviction or
adaptive-quality policy and does not measure driver memory exactly.

Unsupported requests, including the standard portal profile and LOD/RAD
traversal/paging, retain the existing Vulkan path. Selection queries,
the desktop UI, grid, gizmos and final composition also remain on Vulkan. This
backend is not yet a fully independent Metal desktop presentation/editor stack.
Automatic continues to use Vulkan, and no global Vulkan shader is modified.

## Verification and remaining native work

GPU contracts cover projection, stable sorting/composition, tensor producer
ordering, interop, native hardware point coverage, resize/reuse, failed-encode
recovery, adapter routing, four output slots, asynchronous ticket delivery and
abandonment, median/expected depth capture and resource release. CPU contracts
cover backend selection and working-set admission, including 64-bit overflow.
An independent analytic ray reference exercises native 3DGUT color, first,
weighted and median depth, IDs and reservation validation. The macOS-only parity
suite also compares 3DGUT SH0/Q16 and its mip, orthographic, depth, export-scale,
affine, selection, crop and marker variants against Vulkan.

The panorama tests include an independent double-precision projected-center
oracle, spherical alpha/depth/ID rays, rear/longitude seam coverage at odd widths,
and full-image versus native subregion equivalence. The macOS parity fixtures
include spherical mip/depth/export, transforms, selection, crop/window and markers.
Panorama subregion parity uses a crop of the full Vulkan reference camera: its
legacy subregion path wraps the local grid and can clip a seam. That reference
path is not modified here. Subregion timing ratios are omitted because the two
render extents differ. Hard marker core/edge transitions can differ at individual
pixels from subpixel FP32 projection rounding. Those exact flat-color boundary
pairs are reported separately and bounded to 0.1% of pixels; full-image RMS and
the stable-color max gate remain enforced. This is not pixel identity.

The deterministic Vulkan comparisons cover SH0 and SH3 Q16, mip, orthographic,
depth, crop/ellipsoid/window, committed/preview selection, center markers, flash
and affine transforms. Color gates are max 4/255 and RMS 1/255. Depth separately
reports full-image error, same-coverage error and coverage disagreement; its
gate allows at most 0.1% disagreement, max 4/255 and RMS 1/255 on shared coverage.
FP32 native and FP16 reference blending can cross the 50% median boundary on
different pixels. A passing depth gate does not mean pixel-identical depth.

A local macOS real-scene check imported a 1,179,648-splat SH0 PLY, rendered it
with confirmed native routing, exercised selection and whole-node translation
without point selection, exported SPZ, and activated spatial and temporal
reconstruction, including convergence to zero remaining temporal samples. The
fixed-view color comparisons had PSNR 65.54-65.56 dB. Three repeated native
captures were identical. Repeated Vulkan captures varied at one pixel out of
456,320 (maximum 6/255); comparisons against its stable captures had maximum
error 2/255. The isolated reference variation remains undiagnosed. This is
close visual parity, not pixel identity or a guarantee for every scene,
camera or editor workflow. Synthetic maximum-error gates remain unchanged.

The native desktop boundary returns structured `lfs::Result`/`lfs::Status`
errors, preserving existing typed causes and classifying invalid arguments,
missing tickets, empty output and memory admission failures. Explicit adapters
retain the existing Vulkan-facing string contracts. Backend routing logs are
emitted on successful frames and route changes, including Automatic and fallback;
tensor selection is logged after startup preflight. Preferences reject unavailable
CUDA/Metal choices with a localized dialog and preserve the previous settings.

Remaining native work includes 3DGUT/equirectangular and portal profiles,
LOD/RAD admission, page layouts and leases, selection-query kernels, pressure
eviction and adaptive reservation, and representative sustained performance
across large scenes. RAD's signed-byte/page-frame layout must not be decoded as
SplatData Q16. iOS can reuse projection/raster/scene contracts but needs direct
Metal presentation and its own device/simulator verification.

The existing tensor Metal backend's OS/feature requirements do not automatically
become this viewer's requirements. This module currently compiles MSL 2.4 and
does not depend on the tensor backend's Metal 4 submission machinery.

## Reproduce the native GPU contracts

No third-party dependency build is needed:

```sh
cmake -S tests/metal_viewer -B build-macos-metal-viewer-contracts -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-macos-metal-viewer-contracts --parallel 4
ctest --test-dir build-macos-metal-viewer-contracts --output-on-failure
```

The test uses real Metal execution, not a software mock. It compares all four
SH layouts and active degrees 0–3 against analytic +Z SH values, including
31/32/33 and 255/256/257/511 row counts; verifies absent SH0 buffers, the point
path, affine object motion without source mutation, deletion, empty scenes,
clipping invalidating old outputs and rejection of malformed buffer/frame inputs.
It skips explicitly (77) if no Metal device exists. Physical iOS and the complete
desktop application are not validated by this target.

`MetalViewerRasterContracts` compares GPU pixels with an independent stable CPU
reference for color, alpha, weighted/median/first depth and picking. It covers
empty/reused frames, overflow recovery, partial tiles, equal-depth ties, and
255/256/257 and 2047/2048/2049/4097 splats in Gaussian/point/disc modes.
`ViewerBackendSelectionContracts` exercises requested/default/availability/frame
capability combinations independently of GPU hardware.

The optional `LFS_TEST_METAL_VULKAN_INTEROP=ON` test links an existing Vulkan SDK;
provide `Vulkan_INCLUDE_DIR`, `Vulkan_LIBRARY` and an existing macOS ICD through
`VK_DRIVER_FILES`. It tests native texture import and Metal-to-Vulkan GPU timeline
ordering with an exact readback comparison. It does not test a descriptor-based
desktop draw or authorize rebuilding/installing MoltenVK. The texture is imported
already backed by Metal memory, as specified by `VK_EXT_metal_objects`.

## macOS Vulkan/Metal benchmark

The macOS desktop test build adds `mac_viewer_backend_benchmark` by default
(`LFS_BUILD_MAC_VIEWER_BENCHMARK=OFF` disables it). It is not built on Windows,
Linux, iOS, or in the standalone raster contracts without the desktop adapter.

```sh
VK_DRIVER_FILES=/path/to/MoltenVK_icd.json ./build-macos-release/tests/metal_viewer/mac_viewer_backend_benchmark \
  --count 100000 --width 1280 --height 720 --warmup 12 --samples 40 \
  --output build-macos-release/mac-viewer-benchmark.json
ctest --test-dir build-macos-release -R '^MacViewerBackendBenchmarkSmoke$' --output-on-failure
```

The deterministic fixture runs SH0 and SH3 with production Q16 on the same
resident model, camera, extent and display settings. AB/BA pairs alternate APIs;
warmup excludes shader compilation and initial reservations. Each measured frame
waits for its actual GPU completion. Native overflow/error output is rejected.
CPU image readback and complete desktop UI/composition are excluded. These are
serial completed-frame wall latencies, not GPU kernel timestamps or pipelined
viewer FPS. Safe mode isolates the process from saved user preferences.

The JSON contains raw samples, median/p95, Vulkan/Metal median ratio, image
MAE/RMSE/PSNR/max error, device/OS/compiler, validation environment, and combined
process peak RSS. A ratio above one means Metal was faster for that case only.
Disable GPU validation for performance runs; retain it for correctness checks.
Synthetic scenes do not establish representative real-scene performance or
visual parity. Inspect local image errors as well as aggregate PSNR.

The Mac CI runs a small smoke case and the parity fixtures, and uploads
all JSON reports without a speed gate.
It explicitly skips on hosts lacking the resident tensor Metal backend (currently
macOS 26/Metal 4); the raster-only contracts still run on supported older hosts.
Windows and Linux CI continue to run only the CPU backend-selection contract.
