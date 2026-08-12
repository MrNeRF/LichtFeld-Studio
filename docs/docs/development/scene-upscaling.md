---
sidebar_position: 7
---

# Scene upscaling and temporal rendering

LichtFeld Studio separates scene rendering from interface scaling. The scene is
rendered at a backend-owned input extent and reconstructed into the full
viewport extent; RmlUI and the rest of the interface remain at normal
resolution. **Native is the normal LichtFeld Studio renderer**, not an
upscaling implementation. It is the behavioral and performance baseline and
must retain its existing behavior with zero optional-upscaler cost.

## Available backends

| Backend | Availability | Inputs and behavior |
| --- | --- | --- |
| Native | Always | The normal LichtFeld Studio renderer at full resolution, without an upscaling adapter, history, motion, or optional SDK cost. |
| Spatial | Always | Limited internal spatial prototype used to validate resolution scaling, adapter lifetime, split view, and fallback contracts before external SDK integration. It is not intended as a quality-oriented production upscaler. |
| Temporal | Always | Limited experimental temporal prototype used to validate per-view history, depth, motion, jitter, and reset contracts before external SDK integration. It is not intended to match production temporal upscalers. |
| NVIDIA DLSS | Optional build | DLSS Super Resolution through a user-provided NGX SDK. Frame Generation is not enabled. |
| AMD FSR 3.1 | Optional build | FidelityFX FSR 3.1 upscaling through a user-provided SDK. Frame Interpolation is not enabled. |

Preferences lists only backends registered by the current build and available
on the active Vulkan device. Scale and quality values are owned independently:
switching backend does not overwrite the base renderer's scale or another
backend's choices. Native is fixed at 100 percent.

The selected backend is requested, probed, and resolved explicitly. Safe mode,
a missing runtime, an unsupported device or graphics API, a failed probe, or a
recording failure causes an explicit fallback to Native. Unavailable optional
features must not allocate adapters or input resources.

## Build configuration

Both external integrations support 64-bit Windows and Linux Vulkan builds.
`LFS_ENABLE_NVIDIA_DLSS` and `LFS_ENABLE_AMD_FSR3` both default to `OFF`; a
fresh build therefore contains neither vendor SDK. CMake cache values survive
later configure runs, so a build directory previously configured with either
option enabled remains enabled until that option is explicitly set back to
`OFF` or the cache is recreated.

### NVIDIA DLSS

Obtain the SDK from the
[NVIDIA DLSS developer page](https://developer.nvidia.com/rtx/dlss/get-started),
unpack it locally, and configure:

```sh
cmake -S . -B build \
  -DLFS_ENABLE_NVIDIA_DLSS=ON \
  -DLFS_NVIDIA_DLSS_ROOT=/path/to/DLSS
```

`LFS_NVIDIA_DLSS_PROJECT_ID` defaults to LichtFeld Studio's stable public NGX
project identifier. It is not a credential or a per-user setting. Override it
only for a separately registered application identity.

CMake validates the NGX and Vulkan headers and platform libraries. On Windows
it links the compatible static import library and stages `nvngx_dlss.dll`
beside opted-in targets. On Linux it links `libnvsdk_ngx.a` and stages the
newest matching release `libnvidia-ngx-dlss.so.*`. Missing files stop
configuration with the official download link instead of silently disabling a
requested integration.

### AMD FidelityFX FSR 3.1

Obtain FidelityFX SDK 1.1.4 from the
[AMD FidelityFX SDK page](https://gpuopen.com/amd-fidelityfx-sdk-1/) and
configure:

```sh
cmake -S . -B build \
  -DLFS_ENABLE_AMD_FSR3=ON \
  -DLFS_AMD_FSR3_ROOT=/path/to/FidelityFX-SDK-1.1.4
```

`LFS_AMD_FSR3_BUILD_SDK=ON` is a subordinate Windows convenience option: it is
consulted only after `LFS_ENABLE_AMD_FSR3=ON` and cannot enable FSR by itself.
When active, CMake copies the user-provided SDK into the build tree and builds
only the Vulkan FSR 3.1 upscaler and backend libraries; it never modifies the
source SDK. Set `LFS_AMD_FSR3_LIBRARY_DIR` to use existing libraries instead.

The upstream 1.1.4 standalone generator is Visual Studio-oriented. Linux
builders provide Linux-built `ffx_fsr3upscaler` and `ffx_backend_vk` libraries
through `LFS_AMD_FSR3_LIBRARY_DIR`. The integration is static, so no FidelityFX
runtime is staged beside the application.

Both SDKs can be enabled in the same build. `BUILD_TESTS=ON` can be included in
the same configure command; it does not enable either SDK by itself.

## Rendering contracts

The registry describes each backend's requirements before creating resources.
Adapters are constructed lazily only after selection and a successful probe.
Each viewport, including both sides of split view, owns independent adapter,
temporal-frame, and history state.

### Resolution

- The output extent is the full viewport extent.
- The render extent is derived from the selected backend's scale and rounded to
  valid pixel dimensions.
- Base render scale and backend scales are separate persisted values.
- A resize, split-view change, backend change, or scale change updates extents
  without stretching an old rendered texture.
- Native requires no reconstruction resources and remains fixed at full scale.

### Depth

Unavailable depth is a valid zero-cost state. Requested depth declares
dimensions, tensor or Vulkan-image storage, linear-view or Vulkan-NDC encoding,
near and far planes, projection type, and vertical origin. Available depth must
match the render extent and have finite, ordered clipping planes.

### Motion vectors

Unavailable motion is also a valid zero-cost state. Requested motion declares
storage, dimensions, encoding, direction, jitter inclusion, and vertical
origin. The canonical direction is current-to-previous pixel displacement;
normalised-UV and NDC inputs are converted explicitly.

### Temporal history and jitter

Temporal history is per view and advances only after a valid frame is
committed. It is invalidated on first use, explicit camera cuts, incompatible
extents or projection, scene changes, backend changes, quality changes, and
failed resolves. Color history is mandatory when requested; depth history is
allocated only when requested. Jitter is resolution-aware and applied only to
the scene projection supplied to the temporal path.

## Persistence and reset behavior

The atomic `preferences.json` store owns:

- `scene_render_scale` for the base renderer;
- `scene_upscaler` for the selected backend;
- `scene_upscaler_scales` for independent per-backend scales;
- `scene_upscaler_qualities` for independent per-backend quality presets;
- `scene_temporal_quality` as the compatibility value for older preferences.

Invalid or unknown values use built-in defaults. Safe mode neither reads nor
writes these preferences and resolves optional backends to Native. Section and
global resets restore the same defaults through the preference owner; no
upscaler state belongs in `layout.json`.

## Diagnostics

The performance HUD exposes grouped counters under `viewer.resolution`,
`viewer.depth`, `viewer.motion`, `viewer.temporal`, and `viewer.upscaler`.
Ordinary console output reports adapter activation, the effective backend,
fallbacks, warnings, and errors. Run with `--verbose` or `--log-level debug` to
also see SDK queries, recommended scales, context dimensions, adapter teardown,
and temporal reset masks.

## Contract tests

The focused suites cover registry and fallback behavior, independent persisted
settings, render extents, depth and motion encoding, reprojection, frame and
history ownership, temporal planning and resolve, adapter lifetime, and
optional SDK contracts. With tests enabled, run:

```sh
build/tests/lichtfeld_tests --gtest_filter="SceneUpscalerRegistry.*:SceneDepthContract.*:SceneMotionContract.*:SceneMotionReprojection.*:TemporalFrameTracker.*:SceneTemporalPlan.*:SceneHistoryContract.*:SceneHistoryTracker.*:SceneTemporalCoordinator.*:SceneTemporalResolve.*:VulkanSceneUpscalerController.*:AmdFsr3Contract.*:ThemePreferencesContract.Scene*" --gtest_color=yes
```

On Windows use `build\tests\lichtfeld_tests.exe` and ensure the build directory
and `build\vcpkg_installed\x64-windows\bin` are on `PATH`. Optional SDK builds
should also be exercised with their backend selected, in split view, and in
safe mode to verify the explicit Native fallback.
