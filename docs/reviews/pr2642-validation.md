# Native macOS viewer integration validation

Historical integrated runtime source: `d9bd2a680ea92169ac1c2fe0ef9c3b6ae99b81c3`.
The sections below retain their original source attribution. Current primary-PR
results after scope cleanup are in the final section; historical passing counts
do not imply the full validation suite passes after restoring upstream kernels/locks.
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
- Selected Python suite in the fresh isolated run: 3228 pass, 211 subtests, one skip, 67 deselected.
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
- Viewer tail latency: current Raccoon front SH0 p95 13.010 versus historical
  6.161 ms; Lizard front SH0 6.036 versus 5.213 ms. Cause remains open.
- Actual scope separation using the prepared cumulative dependencies; earlier nine ref binaries are not individually GPU-tested.
- Hosted CI, stable Xcode 26, physical macOS 26, CUDA and non-Mac production Vulkan.
- LOD CPU-download follow-up and corresponding issue linkage.

The LOD fallback remains accepted for this viewer change. Production Vulkan
shader precision is unchanged; the isolated Mac test reference must not be used
to claim a production-native speedup from its own increased precision cost.

## Isolated cumulative-scope validation

The isolated current runtime/test source `5a3e9183a` matches integration `9b9c03709` except `docs/reviews/pr2642-validation.md`. Production code is unchanged from `d9bd2a680`. Fresh validation: CTest 259/259; Metal 496 pass, 117 skip, 5 disabled; viewer 19/19; Vulkan snapshot/queue/step 22/22; Python 3228 pass, 211 subtests, 1 skip, 67 deselected.

The shared-Metal-timeline readiness fixture fails 99/100 repetitions before its setup correction and passes 100/100 afterward with API/shader validation. The production queue implementation and all completion/data assertions are unchanged. The bounded-command-frame scope additionally passes 66 contracts in each of three repetitions.

Twelve cumulative refs build successfully; CPU checks pass at each ref. Functional GPU/Python coverage is recorded at refs 10/11/12. The earlier nine ref binaries are not claimed individually GPU-tested; actual online scope separation remains pending.

The first Release build is clean. Subsequent refs use the same isolated checkout/build path with CMake regeneration and Ninja dependency checks. The early CPU-labelled checks are SDL smoke coverage; they are not full CPU or GPU suites. Excluded drag/drop and visualizer targets are explicitly built where present.

## Current viewer benchmark and tail diagnostics

The immutable isolated source `5a3e9183a` completes three repetitions of all
25 original PR configurations. Production runtime is identical to `d9bd2a680`.
The benchmark binary SHA-256 is
`1fc9c90b21665986271c02d72afc30976c35cf11c733697134b59b1110bada53`.
Each repetition uses 100 warmup pairs and 40 samples per backend/configuration;
120 samples are aggregated. Scene, camera, resolution, mode, SH storage, renderer
identity and isolated preferences match the original JSON metadata. All 75 image
comparisons pass unchanged numerical gates. No peaks or unfavorable rows are removed.

The ordinary timing campaign runs in full system wake from 12:04:32 to 12:07:20
on 2026-10-04, without concurrent compilation, training or GPU tests. Default
MoltenVK is used without configuration overrides; API/shader validation and
profiling are off. Inputs are read-only and output files are internal. Seven Metal
medians improve and eighteen increase by 0.06–2.11%. Raccoon front SH0 p95 is
13.010 versus original 6.161 ms (+111.2%); Lizard front SH0 is 6.036 versus
5.213 ms (+15.8%). These unfavorable results remain acceptance boundaries.
Historical/current measurements without controlled thermals or alternating
baseline binaries do not isolate a code regression. The Mac Vulkan reference
remains test-only; its higher-precision cost is not counted as a native gain.

A separate read-only diagnostic uses the same immutable source and binary,
eight invocations across both front cameras, two repetitions and profiling
off/on. SH0 and SH3 each collect 400 samples/backend/invocation. All sixteen
image comparisons pass. Full wake is verified from 12:18:13 to 12:19:59.
Ordinary timing and profiling samples are kept separate. The second profiled
Raccoon SH0 run has Metal median/p95 10.357/15.671 ms and Vulkan
10.398/15.115 ms. Metal GPU command time is 9.303/14.274 ms; multiple GPU
stages slow down. Similar intermittent GPU-stage increases appear in Lizard
SH3. This demonstrates a GPU-execution component rather than filesystem/host
wait alone, but does not identify its cause or rule out code regressions.
No production fix is claimed from the profiling run, and it does not replace
the unfavorable 25-case measurements. Driver, competing GPU load and GPU
frequencies are not independently controlled.

## Primary PR after scope cleanup

The primary branch removes the four out-of-scope areas requested in re-review: training dispatch/kernel resizes, the MoltenVK buffer-creation lock, the retained screen-session correction and unrelated Python fixture rework. Only those 25 files change in the four removal commits; viewer and snapshot sources are unchanged. Separate refs preserve the proposals. The deployment CI assertion additionally requires exactly macOS 26.0, verified on the app and eight first-party dylibs.

Current runtime `a4c7bf552`: Release application/default targets and excluded visualizer/drag-drop targets build successfully. CTest **259/259**; ordinary portable Metal **496 passed, 117 skipped, 5 disabled**; viewer/shared/profile with API and shader validation **19/19**; Vulkan snapshot/queue/step with validation **22/22**. Neutrality, error census, I/O discipline, locales and whitespace pass. App and eight first-party dylibs declare macOS **26.0** in their load commands. CI now checks that exact deployment baseline.

The full portable suite with API/shader validation aborts, so it is not reported as passing. With a canonical temporary directory, Python has **3200 passed, 28 failed, 211 subtests passed, one skipped, 67 deselected**. Skipped/disabled cases are exclusions, not successes; CUDA and Python GPU/slow/integration selections are outside these counts.

The two validation aborts also reproduce on a separately built, pristine `dev` (`dee7d8925`) with the same Xcode/SDK/driver: `TensorVulkanReduce.ReadbacksStayOrderedBehindTheirProducersAcrossThreads` aborts on a nil MoltenVK resource in **3/3 fresh processes on each revision**; `Backends/PortableLosses.PhotometricPathsMatchReference/Metal` exceeds the shader-validation threadgroup limit (**36,448 > 32,768 bytes**) on both revisions. The removed lock/kernel changes had addressed these local failures; their fixes are preserved for separate review rather than retained in the viewer diff.

The same 28 Python failures reproduce using pristine `dev` Python source/tests with the same compiled bindings and environment. This is a Python-only control, not a complete native `dev` Python build. They concern legacy file-credential expectations in bug-report/gallery/portal fixtures. The first Python attempts additionally have 13 `/var` versus `/private/var` temporary-path failures; using canonical `TMPDIR` removes those without modifying the restored fixtures or hiding Keychain. No failing test is converted to a skip.

Garden 5M, four views, fresh app profile, default MoltenVK, no memory/argument-buffer/validation overrides for timing. The project copy and executable are on the internal SSD; the original project and dataset inputs are read-only. The power log confirms uninterrupted full wake from **13:08:57 to 13:12:42 on 2026-10-04**.

| Active save | Pre 100-step mean, ms | Post 100-step mean, ms | Change | Existing <=10% gate |
|---|---:|---:|---:|---|
| native / 1 | 139.811 | 142.665 | +2.041% | PASS |
| native / 2 | 141.780 | 140.555 | -0.864% | PASS |
| FSR quality / 3 | 128.365 | 129.294 | +0.724% | PASS |

Three saves during active training pass snapshot consistency and real RSS gates. Generation 8 reopens and its restored optimizer advances 21 steps with 5M Gaussians and finite loss. Fifty paused FSR Quality captures plus a one-view probe, target retirement/recreation, and ten captures during training complete without fallback. Representative images are inspected; the app closes normally and the original SHA-256 is unchanged. Sampled loss is not a cross-run quality comparison.

Pause samples are **147.710, 254.332 and 193.359 ms**, against the unchanged calibrated formula yielding **33.605 ms** in this run. Initial sync is **67.305, 74.749 and 75.278 ms**; serialization/issue is **78.914, 178.338 and 116.971 ms**. Pause/cold gates remain **FAIL**. Total-save times on this internal project copy are not a controlled code A/B or a completed storage comparison.

Three complete repetitions of all **25 original PR configurations**, 100 warmup pairs and 40 samples/backend/configuration/repetition (**120 aggregated samples**), with matched scenes/cameras/modes/storage and all **75 image comparisons passing unchanged numerical gates**. Power logs confirm full wake **13:13:17–13:16:04**; profiling and validation are off and other builds/training/GPU tests have ended.

9 Metal medians improve relative to the historical PR table; 16 increase, with a maximum **+2.10%**. Worst aggregate p95 increase is **+6.34%** (User PLY Depth). The large earlier Raccoon front SH0 spike is not reproduced in these three current series; its earlier GPU-stage slowdown remains unexplained. These historical/current measurements do not isolate a code effect or establish zero regression. Every raw sample and unfavorable row is retained.

## Integration with dev

The branch incorporates `dev` through `b96cc1a37`. The tested Release runtime is `0e2028ae3`. It retains upstream frame-demand scheduling, idle scratch release, node modifier evaluation, responsive training preparation, codec/build fixes and separate UI/view FPS. Native Metal scene factories and the existing snapshot ownership contracts remain in place. The 20 restored out-of-scope files match this base; training-kernel/dispatch resizes, the MoltenVK creation lock, screen-session correction and unrelated Python fixture changes remain excluded.

Integration fixes schedule renderer refinements through frame demand, use the production localization adapter in the narrow status-bar fixture, and prevent unchanged render-setting bindings from publishing scene/settings edits. The latter preserves conversion, validation, fresh-setting rebasing and notifications for real edits. Six echo cases fail before that guard; the final runtime passes all **8** publication/idle/live-edit tests. Upstream's conditional node-gizmo overlay update also contributes to avoiding needless redraw; the recovery is not attributed to the binding guard alone.

## Current validation

M4 Pro, 24 GiB; macOS 27.2 beta, Xcode 27.2 beta 2, SDK 26.5; Release, trainer enabled, CUDA disabled, FSR enabled. The app/default targets and excluded visualizer target build. The app and eight first-party dylibs declare exactly macOS **26.0**. CI requires macOS 26+, keeps the offline Metal kernel compilation, and probes hosted Metal 4 support: native Metal GPU coverage is explicitly not executed when that capability is absent. Shared/CPU contracts remain enabled.

- CTest: **262/262**.
- Ordinary portable Metal: **894 passed, 131 skipped, 5 disabled**.
- Viewer/shared/profile with API and shader validation: **19/19**.
- Vulkan snapshot/step/queue/project contracts with validation: **22/22**.
- Render-setting publication and idle/live-edit integration: **8/8**.
- Python ordinary selection: **3307 passed, 8 failed, 10 skipped, 88 deselected; 213 subtests passed**.
- Neutrality, error census, I/O discipline, locales, tool tests (50), CI shell syntax and changed-source whitespace checks pass.

The full portable suite with API/shader validation **aborts**. Concurrent Vulkan readback produces a nil MoltenVK resource in **3/3 fresh processes on each revision**; the Metal photometric test requests **36,448 > 32,768 bytes** of threadgroup memory on each revision. Controls use separately compiled pristine `dev` at `cd85ea4ad`; the affected sources are unchanged through `b96cc1a37`. These failures are not counted as passes. All **8** Python failures reproduce using pristine `b96cc1a37` Python source/tests with the same compiled primary bindings and environment. This isolates Python source/fixtures, not a complete native build of that control. No failing test or numerical gate is weakened.

## Garden snapshots and training

Fresh profile, Garden **5M**, **four views**, default MoltenVK; project copies and executable on the internal SSD, source project/dataset read-only. Timing runs have no validation/profiling/memory overrides or concurrent builds/GPU tests.

| Active save | Pre 100-step mean, ms | Post 100-step mean, ms | Change | Existing <=10% gate |
|---|---:|---:|---:|---|
| native / 1 | 139.507 | 142.752 | +2.326% | PASS |
| native / 2 | 143.731 | 141.691 | -1.419% | PASS |
| FSR quality / 3 | 131.676 | 131.204 | -0.358% | PASS |

All three saves pass consistency and real host-RSS gates. Generation 8 reopens and its optimizer advances 20 steps with 5M Gaussians and finite loss. Ten captures during training, fifty paused FSR Quality captures, a one-view probe and target retirement/recreation complete without fallback. The app closes normally; the original SHA-256 `2f664b9124524e279a3a12c6232b03b8d473034a8ad291c7edd9c850c66f58b0` is unchanged. Sampled loss is not a cross-run photographic-quality comparison.

The earlier integrated runtime `a41ef8625` reproduced a slowdown from the exact pre-merge control's **138.153/139.271 ms** to **365.334/366.118 ms**. The table above records the final revision after both redraw guards; the unfavorable intermediate results remain in the local raw record.

| Snapshot | Pause, ms | Cold path, ms | Calibrated gate, ms | Pause / cold | Real RSS |
|---|---:|---:|---:|---|---|
| 1 | 142.840 | 143.144 | 36.166 | FAIL / FAIL | PASS |
| 2 | 353.683 | 353.683 | 36.166 | FAIL / FAIL | PASS |
| 3 | 349.401 | 349.401 | 36.166 | FAIL / FAIL | PASS |

Pause/cold acceptance and repeatable end-to-end save-time improvement remain open where the table fails. These thresholds are unchanged.

## Viewer benchmark

Three repetitions of all **25 original PR configurations**, 100 warmup pairs and 40 samples/backend/configuration per repetition (**120 aggregated samples**), matched scene/camera/mode/storage, default MoltenVK, profiling/validation off. Benchmark SHA-256: `70618838abe0cf196452fc0fc127aa69e1ecd037a5e28a24067ef7d8dd38aefd`. All **75** image comparisons pass the unchanged gates. 11 Metal medians improve versus the historical PR table; 14 increase. Maximum median increase is **+2.15%** (Raccoon SPZ — wide, SH3); worst p95 increase is **+34.86%** (Raccoon SPZ — wide, SH3). Unfavorable samples and earlier intermittent spikes are retained.

The metric includes encoding, submission and GPU completion; it excludes loading, warmup, CPU image comparison, desktop composition and frame pipelining. Vulkan is the isolated Mac test reference; its additional precision cost is not a native speedup.

| Scene / mode | SH | Metal median / p95, ms | Vulkan reference median / p95, ms | Metal median vs original PR | Max RGB error /255 |
|---|---:|---:|---:|---:|---:|
| User PLY — Gaussian | 0 | 6.786 / 6.851 | 5.963 / 6.051 | +0.1% | 2.00 |
| Bicycle SOG — Gaussian | 0 | 6.750 / 6.817 | 5.554 / 5.666 | -0.1% | 1.00 |
| Bicycle SOG — Gaussian | 3 | 7.265 / 8.471 | 6.311 / 9.135 | -8.8% | 2.00 |
| Lizard SPZ — front | 0 | 5.156 / 5.574 | 4.523 / 4.970 | -0.1% | 2.00 |
| Lizard SPZ — front | 3 | 5.511 / 5.587 | 5.005 / 5.119 | -0.0% | 2.00 |
| Lizard SPZ — back | 0 | 4.138 / 4.321 | 4.347 / 4.445 | -13.5% | 2.00 |
| Lizard SPZ — back | 3 | 4.510 / 4.568 | 4.830 / 4.965 | -12.0% | 2.00 |
| Lizard SPZ — wide | 0 | 5.910 / 5.972 | 5.491 / 5.579 | +0.5% | 1.00 |
| Lizard SPZ — wide | 3 | 6.295 / 6.368 | 6.047 / 6.173 | +1.1% | 1.00 |
| Raccoon SPZ — front | 0 | 6.106 / 6.184 | 5.205 / 5.300 | +0.1% | 2.00 |
| Raccoon SPZ — front | 3 | 6.564 / 6.628 | 5.826 / 5.942 | +0.6% | 4.00 |
| Raccoon SPZ — back | 0 | 3.544 / 3.620 | 3.519 / 3.631 | -18.4% | 2.00 |
| Raccoon SPZ — back | 3 | 3.872 / 3.943 | 3.973 / 4.075 | -41.8% | 3.00 |
| Raccoon SPZ — wide | 0 | 5.615 / 5.696 | 6.084 / 6.252 | +1.4% | 2.00 |
| Raccoon SPZ — wide | 3 | 6.104 / 8.132 | 6.797 / 10.879 | +2.1% | 2.00 |
| Apartment SOG — Gaussian | 0 | 4.479 / 4.550 | 4.613 / 4.706 | -0.6% | 1.00 |
| User PLY — transparent | 0 | 7.118 / 7.211 | 9.418 / 9.513 | -14.0% | 136.00 |
| User PLY — 3DGUT | 0 | 12.234 / 12.957 | 24.472 / 29.704 | +0.9% | 1.00 |
| User PLY — Depth | 0 | 7.030 / 7.180 | 11.845 / 12.194 | +1.5% | 2.00 |
| Flowers PLY — Gaussian | 0 | 3.285 / 3.366 | 4.258 / 4.336 | -0.8% | 2.00 |
| Flowers PLY — Gaussian | 3 | 3.584 / 3.685 | 4.707 / 4.817 | +0.3% | 2.00 |
| Flowers PLY — Depth | 0 | 3.394 / 3.524 | 9.483 / 9.849 | +0.3% | 1.00 |
| Flowers PLY — Depth | 3 | 3.685 / 3.791 | 9.927 / 10.335 | +1.6% | 1.00 |
| Flowers PLY — 3DGUT | 0 | 10.780 / 11.114 | 15.244 / 17.775 | +0.6% | 1.00 |
| Flowers PLY — 3DGUT | 3 | 11.216 / 11.563 | 15.932 / 18.382 | +1.6% | 1.00 |

The transparent User PLY whole-frame RGB maximum occurs at negligible alpha; valid-alpha and composited comparisons retain their independent existing gates. A passing gate does not mean bit-identical output.

A bounded alternating control reruns exact pre-merge `265dddf2e` and current binaries in **pre/current/current/pre** order on the two previously unstable back cameras, 100 warmup pairs and 400 samples/backend per invocation. All 16 image comparisons pass. The following aggregates contain 800 Metal samples per revision/case:

| Scene | SH | Pre-merge median / p95, ms | Current median / p95, ms | Median change | p95 change |
|---|---:|---:|---:|---:|---:|
| Raccoon SPZ — back | 0 | 4.059 / 5.209 | 3.551 / 3.638 | -12.52% | -30.16% |
| Raccoon SPZ — back | 3 | 3.896 / 3.975 | 4.858 / 12.471 | +24.71% | +213.77% |
| Lizard SPZ — back | 0 | 4.187 / 11.330 | 4.144 / 4.603 | -1.03% | -59.37% |
| Lizard SPZ — back | 3 | 6.192 / 11.426 | 4.522 / 4.593 | -26.97% | -59.80% |

Thermals and GPU frequencies are not independently controlled. Historical/current measurements and this bounded control do not establish zero regressions or a universal speedup.

## Remaining verification

- M5 Max first frame using a fresh profile and default MoltenVK configuration.
- Garden calibrated pause/cold targets and repeatable total-save-time improvement.
- Intermittent viewer tail latency and its cause.
- Hosted CI, stable Xcode 26, physical macOS 26, CUDA and non-Mac production Vulkan.
- The accepted cold-path LOD CPU metadata fallback remains; replacing full means/scales downloads and the requested tracking issue remain follow-up work.
