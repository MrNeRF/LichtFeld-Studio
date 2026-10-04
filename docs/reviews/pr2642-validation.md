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
switch. Two independent processes produced the following native results; the
second also provides a complete FSR window:

| Process | Rendering | Pre mean, ms | Post mean, ms | Change | Existing <=10% gate |
|---|---|---:|---:|---:|---|
| A | Native, save 1 | 136.408 | 138.040 | +1.196% | PASS |
| A | Native, save 2 | 138.040 | 139.945 | +1.379% | PASS |
| B | Native, save 1 | 135.797 | 137.135 | +0.986% | PASS |
| B | Native, save 2 | 137.135 | 136.260 | -0.638% | PASS |
| B | FSR Quality, save 3 | 130.642 | 128.392 | -1.722% | PASS |

Process A's FSR pre-window overlaps the renderer switch by one step and is
excluded from the FSR comparison. Passing the existing gate does not prove
zero slowdown, nor does this short-run sampling establish long-run performance.

Each process completes three saves during active training, retains 5M Gaussians,
passes snapshot consistency and host-RSS gates, and reopens committed generation 8.
The saved optimizer advances at least twenty further steps with finite loss.
Those sampled losses are not a quality comparison between training runs.

Pause and cold-path gates remain unmet. In process B, pause samples are
152.843, 117.050 and 143.893 ms against a calibrated 36.156 ms gate.
The limit remains `checkpoint_bytes / measured_pinned_D2H_bytes_per_second * 1.12`;
it is not replaced with a slower-device allowance. Capture correctness, final
device fences, complete byte coverage and immutable-reader lifetime are retained.
An end-to-end project-write speedup is not established by capture-time gains.

## Metal and FSR lifecycle

On the same runtime source, both processes complete forty four-view FSR Quality
captures with camera movement, retire three view targets, recreate four views,
and complete ten further captures. Per-view reconstruction status reports FSR
Quality with no fallback. Representative images are visually checked.
Each process also completes ten captures while training is active. The app exits
normally after saved-optimizer restoration and a final clean project reopen.
The timing comparison itself excludes these capture/lifecycle operations.

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
