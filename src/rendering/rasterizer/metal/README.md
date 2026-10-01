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
- RAD pool storage has a separate `RadSigned8` specialization: signed-byte
  swizzled SH with per-page/per-degree maxima, padded half SH0/log scale,
  half quaternion/opacity and float32 xyz. Its page size is supplied explicitly
  by the pool owner and validated against the 32-row SH cell width. It is never
  interpreted as Q16 or expanded into a whole-scene float32 SH array. Independent
  analytic contracts exercise degree/band/sign/cell/page boundaries, non-prefix
  physical cuts, half padding and Gaussian/point/disc/GUT modes. A real tensor
  uploader contract quantizes production Q16 and SH0 sources into the pool,
  submits the native consumer without a host producer wait, and mutates the same
  pool afterward to verify generation ordering.
- Desktop RAD paging uses the existing `LodPageCache` and `LodUploadEngine`.
  Native 2048-node pages remain distinct from larger RAD file blocks. Packed
  attributes, signed SH and quantized bounds/links are decoded on the tensor GPU
  queue; mappings are published only after completion. Each native frame owns
  its mapping/age snapshot, and uploads wait for already-submitted consumers
  before reusing pool pages. Logical IDs drive masks, object transforms and
  picking even when they exceed physical pool capacity. Initial CPU previews
  migrate only their resident prefix; captures wait for the pinned root.
  Working-set admission, priority eviction, page fade and saturated-pool sleep
  preserve the shared pager contracts. The macOS `MetalViewerRadPagerContracts`
  generates a real RAD and exercises eviction, metadata, generation replacement,
  first-frame coverage, capture, larger file blocks and stable-view convergence.
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
  Radix histograms use uint32 storage and SIMD scan after the exact uint64
  admission check: their total cannot exceed the uint32 instance capacity.
  This halves histogram storage without narrowing overflow reporting or keys.
- Bounded instance generation, stable GPU radix sort by tile and full float32
  positive radial distance squared, tile ranges, front-to-back composition and transmittance exit.
  The radix sort reuses the algorithm in `training/kernels/metal/fast_raster.metal`,
  without depending on the trainer. It only sorts the tile-ID bytes actually used.
- Four independent 8x8 blend groups share each stable 16x16 bin list. Each
  group stops on its own pixel saturation; ordinary GS candidates are compacted
  stably against the subtile support with a rounding margin. Ordinary GS
  rings/markers, panoramas and portal GUT retain the complete parent list.
- Non-portal pinhole/orthographic GUT stably rejects a Gaussian only when its
  complete 3D alpha-support sphere is outside a subtile ray-frustum plane.
  The affine covariance Frobenius bound includes anisotropy and shear, with
  floating-point margins. Spark uses its actual density cutoff. Ill-conditioned
  transforms disable this optimization. Metadata reuses an otherwise unused
  inverse-row component, with no added buffers or per-pixel copies. GPU tests
  compare exact color/alpha, all depth channels and source IDs against disabled
  culling in eight camera/density/affine cases, including partial subtiles.
- Ordinary alpha support is cached per Gaussian and rejected before exponentials.
  FP16 portal/marker footprints retain their own support equations. Shared GUT
  batches cache inverse geometry and camera-origin transforms for pinhole and
  spherical rays; orthographic origins remain pixel-dependent.
- Blend pipelines specialize mode and active feature flags, eliminating inactive
  branches and register pressure. The synchronized cache creates a variant before
  acquiring a frame reservation, so a compile failure cannot strand its busy flag.
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

The native desktop adapter supports Studio and standard portal 3DGS/3DGUT color/depth, perspective and
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

The standard portal profile embeds the same MIT-licensed compact codecs and
display-color functions used by Vulkan, without modifying those sources. It
quantizes displayed scales, rotation, opacity and radiance, uses the unclamped
pinhole Jacobian, 0.075-pixel GS dilation, normalized Gaussian tails and portal
billboard limits. The GUT path retains its 0.3 dilation, no mip compensation,
raw ray geometry and explicit clamped billboard fragment bound. Tone operators
are applied per Gaussian before blending, with no second tone pass. Portal GS
uses the reference's macro-relative FP16 footprint; native accumulation remains
FP32. An independent analytic alpha contract and eighteen macOS-only comparisons
cover SH0/Q16, mip, orthographic, depth, export, close range, GUT/panoramas, ACES,
affine transforms, selection, crop and markers. Existing parity limits are retained.

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

Resident LOD cuts use compact draw reservations while directly addressing the
original SplatData source, including Q16 cell swizzle and per-block bounds. Physical
and logical indices remain distinct through projection, object transforms, masks
and picking. Paged cuts carry an explicit logical scene extent independently
of their smaller physical pool. Selection/preview and deletion masks may cover
only a resident logical prefix: bounds are checked before reading, unseen IDs
remain unselected, and RAD deletion addresses logical IDs. Independent GPU
contracts cover IDs beyond the physical pool, short masks, picking and invalid
logical IDs under Metal shader validation. Optional transition weights and level colors use cut-space metadata.
Invalid source/logical IDs are culled before reads; empty cuts clear prior coverage.
Buffers are retained/reused per output slot, with working-set checks on growth,
and cut changes participate in refinement invalidation. No full-attribute gather,
CPU sort or SH expansion is needed. Twelve macOS comparisons cover sparse reverse
cuts, logical masks, weights and level colors in Studio/portal/GUT.

The existing Vulkan GUT fragment gather has no LOD-indirection bindings and reads
compact slots as source IDs. Its sparse-cut reference in this suite is therefore
a full source-layout scene with unselected nodes hidden by zero opacity, weights
folded into opacity, and debug/mask state mapped to source IDs. This is explicitly
reported and no timing ratio is produced for that fixture. Vulkan is unchanged.

Resident and paged Spark-encoded 3DGS/3DGUT opacity is decoded from the compact 1..2 range
to the 1..5 density kernel, with transition weights and non-mip compensation.
Density and cutoff are computed once per Gaussian/tile into existing shared
slots, with no extra allocation or per-pixel density exponential. Portal tone
remains per Gaussian while its compact probability codecs/tails are disabled
for Spark. Eight real-reference comparisons and analytic activation/alpha
oracles exercise this separately from the ordinary sigmoid path.
Spark projection support uses the density cutoff, including the covariance cap,
so high-density tails survive tile binning. Native 3DGUT retains weighted source
opacity for 3D ray evaluation rather than projected mip compensation. The
standalone `MetalViewerSparkGutContracts` checks more than two million pixels
against independent double-precision 3D ray/density equations in perspective,
orthographic and equirectangular views, with mip, output scale and LOD weights.
The test exposed truncated high-density tails before the support correction.
RAD capture contracts also exercise the complete native Spark GUT route.
These cases are not compared blindly against the current Vulkan sparse-GUT
reference: its fragment source indexing and ordinary sigmoid GUT opacity do
not implement the logical-cut/Spark-density contract. Vulkan remains unchanged.

Resident ordered hierarchies now select their cut entirely on Metal GPU,
including root reachability, threshold transitions, page fade, missing-page
interest and up to sixteen budget retries. GPU indirect retries stop once the
cut fits, with a final root cut for pathological size ordering. Projection reads
the live GPU count, clears unused slots and rejects an overflowed cut without
reading unwritten private indices. Source tensors/Q16 remain in their original
layout; no CPU cut traversal or attribute repack occurs per frame. Residency
checks operate once per covered page, preserving partial-page bounds while
avoiding repeated checks per child. Shared counters are read only for deferred
diagnostics after completion, never to drive projection. Per-output reservations
reject in-flight reuse; metadata is cached for four independent models.

Independent GPU contracts cover coarse/fine/non-monotone cuts, transition
complements, missing/partial pages, fade, budget retries, lifetimes and dynamic
GPU count-to-projection reuse. Twelve real Metal/Vulkan comparisons cover SH0/Q16,
budget, mip, orthographic, depth, export, portal/tone, Spark, selection and transforms.
GPU tests and comparison binaries are restricted to macOS; CPU policy tests
remain available to Windows/Linux CI without a GPU.

The desktop UI, grid, gizmos and final composition still use Vulkan. This
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
with confirmed native routing, exercised exact 3DGS mask queries and positive
GS/GUT ring picking, and applied whole-node translation/rotation/scale with undo.
SPZ export/reimport and spatial/temporal reconstruction succeeded, including
convergence to zero remaining temporal samples. Native repeated captures were
identical. The three GUT color comparisons differed by at most 1/255 after the
saturation correction below. The first 3DGS Vulkan frame uses its legacy warmup
chain and differs at 28 pixels by more than 4/255; later frames use its macro
chain, with isolated reference variation of up to 6/255 and RMS 0.000529. Later
repeats reproduced that variation between Vulkan captures of the same scene;
the corresponding native captures remained identical. These are measured scene-specific results,
not pixel identity or a guarantee for every scene, camera or editor workflow.
Synthetic maximum-error gates remain unchanged.

The desktop GUT adapter explicitly matches the legacy Vulkan saturating-color
rule: if the next transmittance is below 1e-4, that splat's color and alpha update
are omitted. Median and expected-depth numerator/weight still include it, as in
the reference. GS macro composition and native analytic/Spark rendering retain
their include-last-contributor equations. Independent GPU assertions test both
color modes and unchanged expected depth. Two macOS comparison fixtures use
nearly opaque overlapping Gaussians: the GUT fixture fails before this correction
with a 10/255 maximum color error and passes afterward. No Vulkan shader changes.

The native desktop boundary returns structured `lfs::Result`/`lfs::Status`
errors, preserving existing typed causes and classifying invalid arguments,
missing tickets, empty output and memory admission failures. Explicit adapters
retain the existing Vulkan-facing string contracts. Backend routing logs are
emitted on successful frames and route changes, including Automatic and fallback;
tensor selection is logged after startup preflight. Preferences reject unavailable
CUDA/Metal choices with a localized dialog and preserve the previous settings.

Representative sustained performance across multiple large scenes remains under evaluation; projection, resident GPU traversal,
RAD page decoding/admission/eviction and adaptive frame reservations are native. RAD's signed-byte/page-frame layout must not be decoded as
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
It also compares a 131073-splat stable sort with the CPU oracle, exercises three
histogram scan levels and reuses that reservation for empty and smaller frames.
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

`--input /path/to/scene.ply` (or another shared-loader splat format) measures a
real resident model. Loading, optional Q16 codec preparation and camera fitting
precede warmup and measurements. Both adapters use the same model, node basis,
camera, size and settings. Without `--camera`, GPU bounds fit a perspective camera.
`--camera camera.json` accepts the camera pose from a saved MCP `camera_get`
response or a raw object with `eye`, `rotation_matrix` and `fov_degrees`. Reports
retain its actual pose/focal length, input extent, loader and SH storage. Synthetic
hierarchy/geometry/overlay fixtures are rejected with real inputs; they must not
be mistaken for the imported file's hierarchy. The macOS CI generates a small
binary PLY through the shared writer and verifies the loader, SH3/Q16 and camera-fit
path without downloading assets.

```sh
VK_DRIVER_FILES=/path/to/MoltenVK_icd.json ./build-macos-release/tests/metal_viewer/mac_viewer_backend_benchmark \
  --input /path/to/scene.ply --camera /path/to/camera.json \
  --width 600 --height 668 --warmup 12 --samples 40 --output build-macos-release/real-viewer-benchmark.json
```

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


Native editor kernels can submit explicit mutable tensor outputs through
`MetalTensorReader::submitWrites`. Producer readiness, native completion and
storage retirement use the same GPU events as read-only rendering. Later tensor
reads/writes/host access observe the native result without an intermediate CPU
copy. A failed encode discards the command; a failed submitted write quarantines
the tensor context before its wait is released, preventing partial output from
becoming valid data. Reader contracts cover sliced outputs, immediate dependent
tensor operations, concurrent source mutation and encode failure recovery.


### Native editor selection

Brush, rectangle, polygon and ring queries now run on Metal and return a native
Bool tensor directly to the shared editor. Polygon coverage uses one GPU pass
per clipped screen region, caches the first 2048 vertices and handles larger
gestures from device memory. Ring picking resolves nearest depth and source-ID
ties on the GPU; only an explicitly requested picked ID requires a scalar wait.
Object visibility, affine transforms, soft deletion, GUT/panorama projection,
mip compensation and packed IEEE half geometry follow the editor query contract.
Queries retain the shared viewport's projection coordinates; raster export tiles
have a separate camera/origin contract. RAD queries cover the resident preview
prefix exposed by SplatData, as the current shared editor does; this does not
enable editing every nonresident leaf of an out-of-core file.

`MetalViewerSelectionContracts` checks independent geometry and polygon coverage,
including a 4099-vertex gesture and deterministic ring ties.
`MetalViewerSelectionViewportContracts` validates the native desktop adapter,
immediate tensor consumers and deletion/undo with Metal validation enabled.
`MacViewerParity_selectionQuery` compares 160 FP32/FP16, camera, affine, visibility,
shape and mip cases against Vulkan. The Vulkan selection binding requires expanded
Float32 geometry, so FP16 comparison uses the same decoded values in the reference.
The parity comparison runs without Metal's shader validator because the existing
MoltenVK polygon pipeline exceeds its threadgroup-memory validation limit; native
query contracts keep full API and shader validation. No Vulkan shaders are changed.


Selection UT means avoid subtracting large weighted image coordinates. An
orthographic projection preserves the source mean exactly; pinhole +/- sigma
pairs use the equivalent rational correction, and spherical means accumulate
unwrapped relative offsets. A targeted double-precision geometry oracle checks
eight subpixel splats at gesture boundaries; the previous shader fails this
regression and the stable formulation passes. A real 1.18-million-splat scene
matched 3DGS selection indices exactly. Seven GUT brush IDs differed from the
legacy Vulkan arithmetic within 0.002 pixels of the boundary; independent double
projection confirmed the native decisions in every case. This is semantic and
numerical parity, not a promise of bit-identical floating-point boundary decisions.

Local paired tests on an Apple M4 Pro illustrate the effect of the blend changes,
without establishing a universal speed advantage. On the 1.18M SH0 PLY at 600x668,
native 3DGS fell from 31.5 ms to roughly 11 ms, and GUT from 62.2 ms to 31.5 ms.
Vulkan varied around 8.7-12.9 ms for GS and 26.6-27.2 ms for GUT. The native GS
image remained byte-identical through the optimizations; repeated reference GS
runs varied at isolated pixels (up to 6/255) without Vulkan source changes. Those
full errors remain reported: a real-input strict parity run can reject them, and
its 4/255 gate is not weakened. GUT stayed within 1/255. The synthetic 100k
SH0/Q16 tests measured about 2.3x for GS and 1.4x for GUT in these runs. All figures
are serial completed-frame latency, exclude desktop composition and must be
retested on other scenes/devices. The later conservative GUT support-sphere
culling reduced the same real-scene native median to 15.3 ms versus Vulkan's
26.7 ms (1.75x in that paired run), while retaining max error 1/255. These are
scene-specific measurements, not a universal or CI speed gate.
