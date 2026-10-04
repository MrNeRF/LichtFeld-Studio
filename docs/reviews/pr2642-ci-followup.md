# CI after PR #2642

Base: merge `718ea0c7f31777d23dbe5849ffafe90dc8575d0f`. Tested code: `f60dafc932aaa5750ff8503b96a049f06c63912d`. Subsequent changes to this record do not alter the tested runtime.

## Observed failures

The [macOS PR run](https://github.com/MrNeRF/LichtFeld-Studio/actions/runs/37207158756/job/111450630964) compiles the application/tests and passes the deployment check, then fails two snapshot fixtures:

- `MeasuresResidentCpuStateDuringCapture`: preparation requires 4,295,033,467 bytes with 3,108,044,800 available.
- `Q16Sh3ChunkedCaptureMatchesHostSerializeBitIdentical`: preparation requires 4,322,719,864 bytes with 3,243,982,848 available.

Both stop at `prepared.has_value()`, before their RSS or byte-equality assertions. The production admission check correctly retains a 4 GiB reserve. The fixtures accidentally depended on ambient free memory; other capture fixtures already provide deterministic available-memory inputs. With an explicit 3 GiB available/16 GiB total input, the unmodified test binary reproduces both failures locally.

The failed step also suppresses all subsequent depth, CPU/shared/native viewer checks and the Metal capability summary through GitHub Actions' implicit success condition.

The [Ubuntu Debug PR job](https://github.com/MrNeRF/LichtFeld-Studio/actions/runs/37207158753/job/111450631009) builds successfully, then reports eight posterize GPU-parameter failures in the CPU Python selection. Each reports `GPU backend 'CUDA' is unavailable`. Five new device parametrizations lack the existing `pytest.mark.gpu` parameter marker. The empty-geometry GPU variant happens to complete without allocating CUDA tensors, but belongs to the same GPU selection.

## Corrections

1. Give only the two successful-capture fixtures scoped available-memory inputs, restored after each test. Actual RSS sampling and the byte-exact, UUID, lifetime and mutation assertions remain unchanged. Dedicated admission tests retain their explicit low-memory rejection/relaxed-save/shortfall checks. Production memory policy is unchanged.
2. Apply the existing GPU parameter marker to the five posterize parametrizations. CPU variants remain selected; all nine newly marked GPU variants remain available to GPU runs, together with the pre-existing animated-input GPU case.
3. Give independent macOS verification steps explicit `!cancelled()` and successful-build conditions. Failed tests still fail the job; there is no `continue-on-error`. Native checks additionally retain the Metal 4 capability condition. The capability notice runs after test failures when the probe succeeded, and does not report absent hardware after a failed probe.
4. Build the excluded visualizer test target and select all non-GPU CPU contracts, excluding the separately exercised `lichtfeld_tests` and `VulkanDepthContracts`. This expands the current Mac CPU selection from four viewer/profile entries to twelve application/viewer entries, covering drag/drop, visualizer layout, temporal reconstruction, SDL coordinates, formats, frame demand and codec exports too.

No production source, numerical tolerance, memory acceptance threshold or GPU assertion changes.

## Local verification

M4 Pro, 24 GiB; macOS 27.2 beta, Xcode 27.2 beta 2, SDK 26.5, Release, trainer on, CUDA off, FSR on. CMake regeneration, application/default targets, and excluded visualizer/drag-drop targets build. App and eight first-party dylibs declare macOS 26.0.

| Check | Result |
|---|---|
| Unmodified snapshot fixtures, explicit 3 GiB available | Both reproduce ResourceExhausted |
| Snapshot/config/serialization selection, Vulkan, same 3 GiB input | 11 passed, no skips |
| Snapshot/config/serialization selection, Metal, same 3 GiB input | 11 passed, no skips |
| Expanded CPU CTest selection, SDL dummy driver | 12/12 passed |
| Full portable Vulkan/VideoToolbox suite | 895 passed, 130 skipped, 5 disabled |
| Vulkan HiGS depth | 1/1 CTest passed |
| Shared viewer contracts | 1/1 CTest passed |
| Python node CPU selection | 25 passed, 10 deselected |
| Python node GPU selection | 10 passed, 25 deselected |
| actionlint 1.7.12 | Passed |
| bash syntax, all workflow run blocks | 17 passed |
| Changed-source whitespace | Passed |

Skipped and disabled cases are not counted as passes. GTest XML, command/source/binary manifests and complete logs are retained in the local verification record.

## Hosted coverage boundaries

The failed hosted Mac run uses macOS 26.6.2 and Xcode 26.6, and explicitly reports `Metal device: Apple Paravirtual device`, `metal4=false`. Fixing execution order can restore CPU/Vulkan coverage and its capability summary; it cannot provide native Metal 4 GPU coverage on that device. Physical Metal 4 Mac validation remains required. Local M4 results do not establish hosted VM or M5 behavior.

The merge's macOS push run was cancelled after newer upstream pushes. The candidate workflow has not been run on GitHub. Windows and Ubuntu Release jobs were still active at the inspection snapshot; no successful conclusion is inferred for unfinished jobs. The branch stays based on the merge under examination rather than incorporating later production changes.
