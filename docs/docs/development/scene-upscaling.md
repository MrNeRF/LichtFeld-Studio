---
sidebar_position: 7
---

# Scene upscaling and temporal rendering

LichtFeld Studio separates scene rendering from interface scaling. RmlUI and
the rest of the interface remain at the normal output resolution. The scene
render extent starts from the existing base renderer scale; when an upscaler is
selected, its input multiplier is applied on top of that base scale and the
result is reconstructed into the full viewport extent. **Native is the normal
LichtFeld Studio renderer**, not an upscaling implementation. It preserves the
existing base-scale behavior and is the zero-optional-upscaler-cost baseline.

## Available backends

| Backend | Availability | Inputs and behavior |
| --- | --- | --- |
| Native | Always | The normal LichtFeld Studio renderer using the existing base render scale, without an upscaling adapter, history, motion, or optional SDK cost. |
| Spatial | Always | Limited internal spatial prototype used to validate resolution scaling, adapter lifetime, split view, and fallback contracts before external SDK integration. It is not intended as a quality-oriented production upscaler. |
| Temporal | Always | Limited experimental temporal prototype used to validate per-view history, depth, motion, jitter, and reset contracts before external SDK integration. It is not intended to match production temporal upscalers. |
| NVIDIA DLSS | Optional build | DLSS Super Resolution through a user-provided NGX SDK. Frame Generation is not enabled. |
| AMD FSR 3.1 | Optional build | FidelityFX FSR 3.1 upscaling through a user-provided SDK. Frame Interpolation is not enabled. |

Preferences lists only backends registered by the current build and available
on the active Vulkan device. The base renderer scale and each backend's input
multiplier are persisted independently, so switching backend does not overwrite
either the base scale or another backend's choices. Native ignores backend
multipliers and therefore uses the base renderer scale unchanged. The internal
Spatial and Temporal prototypes expose their multiplier directly; optional
vendor backends currently derive it from the selected quality preset and SDK
recommendations.

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
it links the SDK's NGX import library and stages `nvngx_dlss.dll`
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

## Adding an optional backend

New production backends should use the optional registry and the common Vulkan
adapter contract rather than adding another value to the internal
`SceneUpscalerBackend` enum. That enum is reserved for Native and the two
in-tree validation prototypes. A registered backend is identified by a stable,
lowercase string ID that is also used by preferences, diagnostics, fallback
status, and the Python UI catalog.

### 1. Declare requirements, do not create them manually

Register an `OptionalSceneUpscalerDescriptor` with the smallest accurate
`SceneUpscalerRequirements` set:

```cpp
optionalSceneUpscalerRegistry().registerAdapter(
    {
        .id = "vendor-backend",
        .label_key = "preferences.scene_upscaler_vendor_backend",
        .requirements = {
            .depth = true,
            .motion_vectors = true,
            .jitter = true,
            .history = true,
            .reactive_mask = false,
            .exposure = true,
        },
    },
    &makeVendorBackendAdapter);
```

These flags are the resource contract, not descriptive metadata. The temporal
planner, viewport pass, input validator, and controller use them to request and
validate only the required work:

| Requirement | Common contract |
| --- | --- |
| `depth` | Render-extent depth with declared storage, encoding, clipping planes, projection, and vertical origin. |
| `motion_vectors` | Current-to-previous displacement with explicit units, jitter inclusion, and vertical origin. |
| `jitter` | Resolution-aware current and previous jitter applied only to the scene projection. |
| `history` | Per-view color history, plus depth history only when depth is also requested. |
| `reactive_mask` | Availability validated by the common input contract; a backend must remain unavailable until the renderer supplies a real mask. |
| `exposure` | Finite positive pre-exposure supplied through the dispatch contract. |

Leaving a flag `false` is a zero-cost guarantee: the backend must not depend on
that input, and LichtFeld must not allocate or generate it for that backend.
Declaring an input that is not yet produced makes selection fail safely rather
than allowing an implicit dummy resource.

### 2. Implement the Vulkan adapter lifecycle

Derive from `VulkanSceneUpscalerAdapter` and implement:

- `probe()` for platform, safe-mode, runtime, device, and Vulkan capability
  checks without allocating persistent rendering resources;
- `initialize()` for device-level SDK state after a successful probe;
- `record()` to validate `VulkanSceneUpscalerDispatch`, record commands into
  the supplied command buffer, and preserve the declared layouts;
- `output()` to return a full-output-extent image and monotonically meaningful
  generation;
- `reset()` to invalidate the specified view for every supplied temporal reset
  reason;
- `shutdown()` as idempotent cleanup for partial initialization, selection
  changes, failures, and application shutdown.

Do not cache one global frame history. `TemporalViewId::Main`, `SplitLeft`, and
`SplitRight` are independent, and split views can have different extents.
Creation should be lazy; unselected backends must have no runtime or VRAM cost.
An adapter failure is reported to `VulkanSceneUpscalerController`, which owns
cleanup, retry boundaries, status counters, and Native fallback.

### 3. Add opt-in build discovery

Create a guarded CMake setup module following `SetupNvidiaDlss.cmake` or
`SetupAmdFsr3.cmake`:

- use `LFS_ENABLE_<BACKEND>=OFF` by default;
- require a user-provided SDK root and link/runtime files;
- never download or redistribute a restricted SDK;
- validate supported 64-bit Windows and Linux layouts explicitly;
- expose an internal `<BACKEND>_AVAILABLE` result only after complete
  validation;
- add sources, libraries, and a private `LFS_HAS_<BACKEND>` definition only
  when available;
- stage user-provided runtime libraries beside every opted-in executable when
  dynamic loading requires it;
- include the official acquisition URL in actionable configuration failures.

Register the adapter after Vulkan instance/device initialization, when the
probe context has real API, vendor, and device IDs. If an SDK needs Vulkan
instance or device extensions before device creation, expose a bootstrap query
like the DLSS integration and disable the backend cleanly if any required
extension is absent.

### 4. Expose settings without hard-coded availability

Add the localized label key to every locale and let
`availableSceneUpscalerCatalog()` drive the Preferences list. Do not add an
unconditional Python list entry for an optional backend. The UI persists its
stable ID, per-backend input multiplier, and per-backend quality preset through
the existing `scene_upscaler`, `scene_upscaler_scales`, and
`scene_upscaler_qualities` owners. These values do not replace the separate
base renderer scale. Invalid or unavailable saved IDs resolve to Native without
rewriting unrelated backend settings.

If the SDK publishes recommended scales or backend-specific quality modes,
expose them as catalog metadata. Backend-specific controls must appear only
when that backend is selectable; they must not mutate the base render scale.
All new visible labels, status text, and validation errors require locale keys.

### 5. Preserve logging and diagnostics policy

At `info`, report only effective-backend deactivation/activation, explicit
fallbacks, and actionable failures. SDK discovery, extension lists, context
dimensions, resource recreation, and teardown belong at `debug` and are
enabled with `--verbose`. Add HUD gauges under `viewer.upscaler` only when they
help verify ownership, fallback, or resource cost; counters must not force
continuous rendering when the scene is idle.

### 6. Required tests

A backend integration is incomplete without tests for:

- registration uniqueness, stable ID, label key, and exact requirements;
- safe mode, not-compiled, missing-runtime, unsupported-device/API, and probe
  failure reasons;
- lazy construction, adapter reuse, per-view reset, idempotent shutdown, and
  destruction after record/output failure;
- invalid or incomplete dispatch rejection for every declared input;
- valid output extent/layout/generation and split-view independence;
- explicit Native fallback without repeated retry or allocation each frame;
- independent scale/quality preference round trips and malformed-value
  fallback;
- CMake OFF-by-default behavior and missing-SDK diagnostics;
- real Windows and Linux runtime smoke tests when the SDK license permits CI or
  developer-machine execution.

Start from the `SceneUpscalerRegistry`, `VulkanSceneUpscalerController`, input
contract, temporal coordinator, split-view, and preference suites. SDK-specific
GPU tests should be filtered separately so the CPU-only contract suite remains
usable on CI runners without the optional runtime.

### Resolution

- The output extent is the full viewport extent.
- The Native render extent is derived from the base renderer scale.
- With an upscaler selected, the effective input scale is the base renderer
  scale multiplied by that backend's input multiplier, clamped to the supported
  range and rounded to valid pixel dimensions.
- Base render scale and per-backend multipliers are separate persisted values.
- A resize, split-view change, backend change, or scale change updates extents
  without stretching an old rendered texture.
- Native requires no reconstruction resources and ignores the backend
  multiplier; it does not force the existing base renderer scale to 100 percent.

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
- `scene_upscaler_scales` for independent per-backend input multipliers;
- `scene_upscaler_qualities` for independent per-backend quality presets;
- `scene_temporal_quality` as the compatibility value for older preferences.

Invalid or unknown values use built-in defaults. Safe mode neither reads nor
writes these preferences and resolves optional backends to Native. Section and
global resets restore the same defaults through the preference owner; no
upscaler state belongs in `layout.json`.

## Diagnostics

The performance HUD exposes grouped counters under `viewer.resolution`,
`viewer.depth`, `viewer.motion`, `viewer.temporal`, and `viewer.upscaler`.
Ordinary console output reports effective-backend activation and deactivation,
fallbacks, warnings, and errors. Run with `--verbose` or `--log-level debug` to
also see SDK queries, recommended scales, adapter initialization and teardown,
context dimensions, and temporal reset masks.

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
