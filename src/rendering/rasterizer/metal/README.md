<!-- SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
     SPDX-License-Identifier: GPL-3.0-or-later -->

# Native Metal viewer backend

The native Metal rasterizer consumes resident LichtFeld tensors, including Q16
SH storage. On macOS it can be selected explicitly under Preferences; Automatic
continues to use Vulkan. The desktop UI, grid and gizmos still use the existing
Vulkan compositor, sampling shared Metal color/depth textures on the same GPU.
No full-frame CPU readback occurs during interactive native presentation.

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

`TileRasterizer` consumes the resident projected splats:

- GPU tile counts and hierarchical uint64 exclusive scan. SIMD reductions use
  16-bit limbs because MSL 2.4 does not provide ulong SIMD reductions.
- Bounded instance generation, stable GPU radix sort by tile and full float32
  positive depth, tile ranges, front-to-back composition and transmittance exit.
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
fallback reason. Desktop routing and the preferences UI are connected. Native rendering supports
Studio 3DGS color/depth, perspective/orthographic views, Gaussian/point/disc
projection, node transforms and visibility, resident float32/half geometry and
Q16/half/float SH storage. Unsupported requests (including 3DGUT, LOD/RAD,
crop/clipping and editor selection overlays) retain the existing Vulkan path.
Preview and deterministic export also retain Vulkan. This is an incremental
native raster integration, not a complete replacement of the desktop renderer.

The GPU contracts cover projection, stable sorting/composition, tensor producer
ordering, interop and native viewport readback/resize. Full workflow and visual
parity, memory-budget behavior and representative performance remain validation
requirements. Passing contracts alone is not evidence of a speed advantage.

## Remaining integration, in order

1. **Capacity policy:** connect typed overflow to the host retry/budget policy;
   evaluate compaction and bounded depth waves on representative large scenes.
2. **Frame contract:** native color, linear depth/alpha, completion and resource
   leases; independent output slots for main/split/preview/export. Shared events
   and bounded buffers across in-flight frames; no reuse while a consumer owns
   the frame. Keep rendering separate from desktop Vulkan composition. A temporary
   GPU interop presenter may integrate it with that compositor, but must be named
   accurately: it is not yet a fully native Metal UI/presentation stack.
3. **Scene adapter:** consume existing ViewportRenderRequest, model transforms,
   node visibility, SH active degree, selection/deleted masks and original IDs.
   Preserve Q16, tensor storage generations and producer completion. Do not
   reintroduce the iOS flat CPU scene or per-frame canonical SH materialization.
4. **Desktop integration:** explicit backend selection with capability reporting;
   never silently ignore unsupported filters, modes or projections. Vulkan remains
   the unchanged default until reference comparisons and workflow gates pass.
5. **Parity:** Studio/profile behavior, points/discs, depth, orthographic camera,
   antialiasing/mip compensation, selection/gizmo/grid depth, crop/clipping,
   resolution changes, split views and export. RAD/LOD admission, paging and
   eviction must respect their own quantized layout and resource ownership.
6. **Performance:** representative SH0/SH3 Q16 scenes at multiple sizes/resolutions,
   camera entering closed scenes, sustained navigation, RSS/GPU buffer peaks,
   per-stage GPU timings, queue depth and p50/p95 frame times against Vulkan.
   Avoid claiming "faster" from a shader-only test or a simulator.
7. **iOS reuse:** share the backend and scene/edit contracts. UIKit owns controls
   and input mapping only. Object transforms do not require point selection.

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
