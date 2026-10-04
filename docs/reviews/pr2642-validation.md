# Native macOS viewer integration validation

Runtime source: `d9bd2a680ea92169ac1c2fe0ef9c3b6ae99b81c3`.
These measurements cover an M4 Pro with 24 GiB RAM, macOS 27.2 beta,
Xcode 27.2 beta 2, SDK 26.5 and a macOS 26.0 deployment target.
They do not substitute for the reported M5 Max reproduction or other platforms.

## Garden snapshots during training

The public MipNeRF360 Garden scene contains 5,000,000 Gaussians in four views.
Each process opens a fresh copy of the saved project with an isolated `LFS_HOME`.
No memory-limit overrides, MoltenVK argument-buffer overrides or validation
layers are enabled for timing. The original project is hash-checked before and
after the runs. Training parameters, topology limits and quality are unchanged.

The application's regression tracker selects contiguous windows of 100 steady
steps before and after each snapshot. Refinement boundaries reset the candidate
post-snapshot window; iteration counts alone do not establish a valid window.
FSR is warmed before capture, and its pre-window must start after the renderer
switch. Temporary power assertions hold the system/display awake; system power
logs confirm full wake with no sleep transition during the final process:

| Rendering | Pre mean, ms | Post mean, ms | Change | Existing <=10% gate |
|---|---:|---:|---:|---|
| Native, save 1 | 135.660 | 139.380 | +2.742% | PASS |
| Native, save 2 | 138.518 | 139.705 | +0.857% | PASS |
| FSR Quality, save 3 | 130.223 | 130.546 | +0.248% | PASS |

Earlier runs retained historical observations but used an uncontrolled DarkWake
power state. One also overlapped the FSR switch by a single step in its pre-window.
They are superseded by this full-wake process for performance claims. Passing
the existing gate does not prove zero slowdown or long-run performance.

The final process completes three saves during active training, retains 5M Gaussians,
passes snapshot consistency and host-RSS gates, and reopens committed generation 8.
The saved optimizer advances twenty further steps with finite loss.
Those sampled losses are not a quality comparison between training runs.

Pause and cold-path gates remain unmet. Pause samples are
125.099, 113.060 and 136.873 ms against a calibrated 35.894 ms gate.
Initial synchronization accounts for 48.518, 77.064 and 74.210 ms respectively;
serialization/issue accounts for 75.491, 34.843 and 61.372 ms.
The limit remains `checkpoint_bytes / measured_pinned_D2H_bytes_per_second * 1.12`;
it is not replaced with a slower-device allowance. Capture correctness, final
device fences, complete byte coverage and immutable-reader lifetime are retained.
An end-to-end project-write speedup is not established by capture-time gains.

## Metal and FSR lifecycle

On the same runtime source, the full-wake process completes forty four-view FSR Quality
captures with camera movement, retire three view targets, recreate four views,
and complete ten further captures. Per-view reconstruction status reports FSR
Quality with no fallback. An intermediate one-view capture also verifies the
retired-target state. Representative images are visually checked.
The process also completes ten captures while training is active. The app exits
normally after saved-optimizer restoration and a final clean project reopen.
The timing comparison itself excludes these capture/lifecycle operations.

## Project storage and discarded runs

In one matched full-wake pair, three paused saves per process have total-write
medians of 3.262 s on external USB storage and 2.034 s on the internal SSD.
Only project location changes; executable, dataset and shaders remain external.
This preliminary pair is not the planned three-pair campaign. Cross-volume file
copy versus APFS cloning and cache effects also limit capture-pause conclusions;
no cold-disk or cache-flush condition is claimed.

Two interrupted trials are excluded. A long system sleep coincides with a GPU
wait quarantine in one trial. In another, kernel logs confirm disappearance of
the external medium, EIO and an unclean-unmount remount; the app reports pwrite
EIO followed by SIGBUS. These incidents are not normal-operation timing evidence.
The original project's SHA-256 is unchanged after the disconnect, Git connectivity
passes, and the integration working tree remains clean. Further storage comparison
requires a stable connection; no filesystem repair or power-policy override is used.

## Automated coverage

The matching runtime source has a completed Release build, including the
excluded visualizer test target, and the following recorded results:

- CTest: 259/259.
- Portable Metal suite with API and shader validation: 496 pass, 117 skip.
- Viewer/shared/profile contracts with validation: 19/19.
- Vulkan snapshot, step-tracker and queue contracts with validation: 22/22.
- Selected Python suite: 3228 pass, 212 subtests, one skip, 67 deselected.
- Backend neutrality, error census, I/O discipline, locales and whitespace pass.
- App and first-party dylib deployment commands use macOS 26 or newer.

The 117 Metal-suite skips comprise 52 unavailable backend/CUDA cases, 61 missing
CUDA oracle/golden/timing cases, one Vulkan-process-only case, one unsupported
CUDA-family loss-curve case, one oversized dispatch-limit fixture and one
MoltenVK-specific allocation-lifetime fixture. Bare gsplat parity skips also
require CUDA, as confirmed from their source guards. They are not counted as passes.
Five disabled cases are timing benchmarks: million-row Vulkan gsplat, fused SSIM
on Metal/Vulkan, and fused Adam on Metal/Vulkan. Python GPU, slow and integration
selections are outside the reported Python run.

## Remaining acceptance evidence

- M5 Max first frame with default MoltenVK configuration, fresh app profile,
  and the dataset/PLY inputs from the review.
- Garden pause/cold-path targets and repeatable total-save-time improvement.
- Independent builds/tests of the prepared dependency stack before scope separation.
- Hosted CI, stable Xcode 26, physical macOS 26, CUDA and non-Mac production Vulkan.
- LOD CPU-download follow-up and corresponding issue linkage.

The LOD fallback remains accepted for this viewer change. Production Vulkan
shader precision is unchanged; the isolated Mac test reference must not be used
to claim a production-native speedup from its own increased precision cost.
