# Native Metal scene rendering

## Architecture

macOS uses the Metal scene renderer through `SceneRenderer` and
`PointSceneRenderer`. Windows and Linux use their Vulkan implementations.
The desktop UI, grid, gizmos, temporal effects and upscalers use the existing
Vulkan compositor. Native color and depth textures are imported on the same
GPU; interactive presentation does not read the full frame back to the CPU.
Scene rasterization has no per-frame backend fallback or viewer preference.
Scene results carry opaque compositor handles; `vulkan_scene_output.hpp` converts
them at the presentation boundary without changing ownership or completion.

`SplatPreprocessor` projects resident geometry and evaluates SH on the GPU.
`TileRasterizer` builds tile intersections, sorts stable full-width radial keys,
and composes color, first-contributor, median and expected depth.
`LodSelector` chooses resident hierarchy cuts. `MetalRadPager` uses the shared
page cache and upload engine for out-of-core RAD scenes.
`MetalViewportRenderer` owns per-target frame reservations, texture interop,
selection queries and readback tickets. Tensor access uses the public
`MetalTensorReader` contract to retain storage and order producers and consumers.
Output recycling waits for native producers, readbacks and compositor consumers;
closed views retire independently.

Dense Gaussian blending can split long lists into parallel summaries. These
use separate scratch admitted from completed counts within the device working
set. Growing frames use the complete serial path until that scratch fits;
parallel summaries never alias sorting buffers.

## Storage layout

- Geometry: float32 xyz, with float32 or padded IEEE half scale, rotation and
  opacity. Attribute semantics and affine transforms match the shared scene.
- SH0: float32 or padded IEEE half. Active SH degree and resident storage degree
  are independent.
- SH-rest: canonical/swizzled float32, IEEE half, or integer Q16. Q16 uses
  32-row cell swizzling, three uint16 cells per coefficient and a float2 bound
  per 256 sources. Rendering decodes resident data without expanding the scene.
- RAD pools: signed-byte swizzled SH with per-page/per-degree maxima, padded
  half SH0/log scale/quaternion/opacity, float32 xyz and quantized page bounds.
  RAD signed-byte storage is distinct from Q16. The supplied page size must
  respect the 32-row cell width.
- LOD/RAD cuts carry physical source indices and logical IDs separately.
  Selection, deletion and object masks use logical IDs. Page generations
  prevent stale cuts from observing reused storage.

## Run the tests

Build the normal macOS project with tests enabled, then run:

```sh
cmake --build build-macos-release
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  ctest --test-dir build-macos-release -L metal --output-on-failure
ctest --test-dir build-macos-release -L parity --output-on-failure
```

Independent projection, raster, selection, LOD, Spark and frame-budget contracts
can also be built without the desktop application:

```sh
cmake -S tests/metal_viewer -B build-metal-contracts
cmake --build build-metal-contracts
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 \
  ctest --test-dir build-metal-contracts --output-on-failure
```

Tests that need Metal 4 skip explicitly when it is unavailable. Set
`LFS_METAL_TEST_REQUIRE_DEVICE=ON` to require a supported GPU. Desktop parity
contracts additionally need the test-only Vulkan reference renderer and the
shared loaders. Run application stress checks with an isolated `LFS_HOME`:
training, several viewports, repeated window captures, resizing and close/reopen.
GPU contracts do not replace those application checks.
