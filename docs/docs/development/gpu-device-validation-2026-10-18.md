# GPU device selection: validation on 18 October 2026

Local branch `feat/gpu-backend-preferences`, based on `dev` commit
`4ef50cbbc7f649cfed1d76d1d1770a3bff62d27f`. Validation was initially local;
publication for review was subsequently authorized.
The implementation and reproduction checklist are in [GPU device preferences](gpu-device-preferences.md).

The branch subsequently merges `upstream/dev` commit
`d23669db9de3d7cfc0b3517fb439428d65fb743d` through merge commit
`acb933295f2d02c03953230d095a0df7f165ed97`, without conflicts.
The user subsequently authorized compilation and runtime tests. Windows, Linux/WSL
and all three macOS build configurations were rebuilt after this merge. The controlled
2,000-step measurements below remain the historical pre-merge comparisons; final
functional checks are recorded separately and are not replacement performance benchmarks.

## Scope and hardware

Windows validation uses a GeForce RTX 5060 Ti and integrated Radeon 890M with a
4 GB BIOS reservation. The driver also exposes a shared system-memory heap;
that heap is labelled as shared in Preferences. macOS validation uses an Apple
M4 Pro with 24 GB unified memory. Linux validation uses Ubuntu 24.04 under WSL;
it cannot establish native Radeon Linux performance. There is only one physical
CUDA adapter, so multiple-CUDA-device policy is tested synthetically, not with
two physical NVIDIA cards. The full LibTorch-based CUDA test suite is unavailable in this local Windows setup; CUDA validation instead includes portable operator/transfer tests and complete training runs.

CUDA Auto retains the existing default-device behavior. Missing or incompatible
saved requests fall back for the current run and remain saved. Backend/device
changes take effect after restart; restoring the active configuration clears
the pending indication.

## Controlled training comparisons

All runs use the same initial checkpoint, SHA-256
`4273c94a3c29daffc84c490651510d91ac71ba9cfb9eba8aa3abc0fced3f63f7`,
with 19,555 initial Gaussians at iteration 1. The dataset contains 82 images,
71 training and 11 held out. Settings: maximum image dimension 1920, SH0,
MCMC, one-million Gaussian cap, CPU image cache and 2,000 iterations.
Windows controls and candidates use the same Release build with `LFS_FAST_COMPILE=ON`, CUDA 12.8 and SM 120. Mac controls and candidates use the same Release settings. Per-image evaluation metadata confirms 1440×1920 pixels. Comparison PNGs are
2884×2016 because they contain two images, a separator and a header.

The timing window is steps 201–1900. The fixed-size window is steps 700–1900,
all at one million Gaussians. These are `train_step` wall times; data-loader wait
is separate. Final evaluation is deliberately excluded. Evaluation uses CUDA
for every Windows comparison and Metal for every Mac comparison. Private trace
instrumentation is removed from the source after each run.

### Times and quality

| Platform/backend | Run | Step mean (ms) | At 1M (ms) | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|---:|
| Windows CUDA | control | 47.189 | 50.071 | 18.146027 | 0.426403 | 0.602287 |
| Windows CUDA | candidate | 46.932 | 49.687 | 18.189753 | 0.426636 | 0.602192 |
| Windows Vulkan NVIDIA | control 1 | 124.614 | 133.081 | 18.358570 | 0.444131 | 0.573281 |
| Windows Vulkan NVIDIA | control 2 | 139.440 | 147.841 | 18.373449 | 0.444272 | 0.572824 |
| Windows Vulkan NVIDIA | candidate 1 | 103.118 | 110.491 | 18.302874 | 0.443372 | 0.572941 |
| Windows Vulkan NVIDIA | candidate 2 | 124.368 | 132.374 | 18.336113 | 0.443877 | 0.573313 |
| Windows Vulkan Radeon | candidate | 229.977 | 247.000 | 18.356977 | 0.443490 | 0.572773 |
| macOS Metal | control 1 | 73.680 | 79.517 | 18.378172 | 0.443720 | 0.573187 |
| macOS Metal | control 2 | 73.455 | 79.234 | 18.352442 | 0.443435 | 0.573988 |
| macOS Metal | control 3 | 73.964 | 79.789 | 18.320723 | 0.442785 | 0.574575 |
| macOS Metal | candidate 1 | 74.285 | 80.072 | 18.335232 | 0.443635 | 0.573355 |
| macOS Metal | candidate 2 | 73.376 | 79.077 | 18.313410 | 0.443408 | 0.573963 |
| macOS Vulkan | control 1 | 435.779 | 470.127 | 18.390209 | 0.443962 | 0.573070 |
| macOS Vulkan | control 2 | 437.292 | 471.184 | 18.339127 | 0.443322 | 0.573353 |
| macOS Vulkan | candidate | 436.453 | 471.267 | 18.323524 | 0.443512 | 0.573430 |

Training is not bitwise reproducible: repeated controls also differ. Metal
control PSNR spans 18.321–18.378 dB. Candidate Metal results are 18.313–18.335 dB,
with small, mixed changes in SSIM and LPIPS. The paired NVIDIA Vulkan candidate
PSNR averages are 0.037–0.056 dB below their controls; SSIM changes are below
0.001 and LPIPS changes below 0.0005. These numerical differences are retained
here, rather than described as identical results.

Visual inspection used the first, middle and last held-out viewpoints
(`IMG_5368`, `IMG_5400`, `IMG_5448`), including amplified difference views.
Statue contours, face structure and the base remain coherent; differences are
most visible in foliage and fine detail. No new holes, stripes or structural
failure were seen in these views. Ground-truth pixels match exactly within
every inspected comparison. This is evidence from these runs, not a guarantee
of equivalence for every dataset or driver.

### Memory and complete process time

GPU counters differ by backend: CUDA derives usage from device total/free,
Vulkan uses VMA device-local heap budget usage, and Metal uses backend allocation
statistics. Compare each backend to its own control. GPU values below are the
largest step-boundary samples before final evaluation. RSS is sampled once per
second across the training process and can miss transient peaks. On unified
memory, RSS and GPU usage must not be added as disjoint allocations. Complete
process time includes startup, loading, checkpoint saving and final evaluation.

| Platform/backend | Run | Sampled GPU (MiB) | Sampled RSS (MiB) | Complete process (s) |
|---|---|---:|---:|---:|
| Windows CUDA | control | 3165.4 | 884.7 | 123.1 |
| Windows CUDA | candidate | 3261.4 | 863.0 | 112.1 |
| Windows Vulkan NVIDIA | control 1 | 1608.0 | 1554.2 | 304.2 |
| Windows Vulkan NVIDIA | control 2 | 1654.5 | 1496.4 | 314.2 |
| Windows Vulkan NVIDIA | candidate 1 | 1547.0 | 1139.4 | 257.2 |
| Windows Vulkan NVIDIA | candidate 2 | 1532.7 | 1188.8 | 288.2 |
| Windows Vulkan Radeon | candidate | 1506.4 | 2068.1 | 640.5 |
| macOS Metal | control 1 | 1085.3 | 1200.8 | 213.4 |
| macOS Metal | control 2 | 996.3 | 1340.1 | 211.4 |
| macOS Metal | control 3 | 1002.3 | 1340.3 | 212.5 |
| macOS Metal | candidate 1 | 1097.3 | 1173.7 | 212.5 |
| macOS Metal | candidate 2 | 1085.3 | 1339.5 | 210.5 |
| macOS Vulkan | control 1 | 1626.6 | 1119.0 | 1011.1 |
| macOS Vulkan | control 2 | 1629.5 | 1118.8 | 1013.2 |
| macOS Vulkan | candidate | 1639.2 | 1121.2 | 1011.9 |

### Sampled device phases at one million Gaussians

These are 97 timestamp samples from steps 700–1900 in the first candidate runs.
Host intervals and device timestamps overlap asynchronously and must not be
added together. Fused parameter updates can be included in backward; the final
optimizer column is not the cost of every Adam operation.

| Backend | Pre-forward (ms) | Forward (ms) | Loss (ms) | Backward (ms) | Final optimizer phase (ms) |
|---|---:|---:|---:|---:|---:|
| cuda | 1.611 | 11.915 | 1.476 | 35.026 | 0.007 |
| vulkan-nvidia | 1.520 | 15.687 | 2.459 | 91.116 | 0.017 |
| vulkan-amd | 3.376 | 106.362 | 11.339 | 123.566 | 0.514 |

## Historical numerical and integration checks

These counts describe the earlier validation phase. Post-merge final counts and
additional operator coverage are recorded in the final-results table below.

- Windows policy: 13 passed. CUDA/Vulkan source and consumer transfer checks
  pass on both physical adapters, including float32, float16, packed uint8,
  offset views, immutable snapshots and reverse timeline waits.
- Windows portable training/queue harness: per adapter, 30 passed, 15 skipped
  and 3 disabled. RGB, depth and normal gradients pass finite differences;
  nonzero backward also matches CUDA. Metal-only cases and two unavailable
  Vulkan sort-capacity hooks are skipped.
- Windows complete UI suite: 352 passed, 5 skipped, 2 disabled. The pending
  restart action is checked at 1280, 960, 640, 480 and 320 pixels, including
  restoration of the active settings.
- macOS: both tensor backends pass the 1,413-case suite in both graphics builds,
  with no failures and five disabled cases per suite. Unavailable backend cases
  are skipped. The Vulkan graphics build's UI suite lists 409 cases with zero
  failures and two disabled. After the final preference guard, Metal-only UI has
  399 passed, 3 skipped and 2 disabled; Metal graphics with both tensor backends
  has 398 passed, 4 skipped and 2 disabled; Vulkan graphics has 399 passed,
  8 skipped and 2 disabled. The Metal-only tensor contracts have 76 passed and
  14 skipped; training contracts have 39 passed, 38 skipped and 2 disabled.
  Its general suite has 748 passed, 416 skipped and 4 disabled. All 25 additional viewer contracts pass, including
  Metal/Vulkan interop and Vulkan-tensor viewer paths.
- Python preferences/localization/gate tests: 221 passed, one skipped. Backend
  neutrality, I/O discipline, CUDA logging and error-debt gates pass without
  changing debt baselines.

The final audit confirms that all 110 tensor SPIR-V binaries match the control
byte for byte. Of 68 training SPIR-V outputs, 62 match; six changed outputs are
raw/optimized Adam and the two pair-sort variants. The pair-sort correction counts a partially occupied subgroup with ceiling
division: a 32-worker group on wave64 has one active subgroup, not zero. At
wave32 the count stays one. Backward wave32 selection is restricted to compatible
AMD devices, with a logged fallback; other vendors retain their selection.

## Independent-device viewer checks

The same one-million-Gaussian checkpoint was opened with CUDA/NVIDIA tensors
and NVIDIA or Radeon viewing, then with Radeon Vulkan tensors and Radeon or
NVIDIA viewing. All four configurations load and render correctly. Offscreen
captures do not change the primary R/T hardware status.

With the same viewer GPU, changing the tensor GPU produces byte-identical
640×480 captures. Between NVIDIA and Radeon viewing, mean absolute channel
difference is 0.03124 on the 0–255 scale, with a maximum isolated difference of
eight. This directly verifies the foreign-device transfer independently of
the variability of training.

Concurrent viewing/training also completes a 2,040-step CUDA run with the Radeon
viewer. Radeon training with the NVIDIA viewer is stopped through the job API
at iteration 658, after reaching one million Gaussians by iteration 600; it
finishes without a GPU error and retains the actual R/T selections. Intermediate
evaluation is disabled for that functional stress test. An earlier harness run
timed out during an inherited intermediate evaluation; it is not included in
the performance or quality comparisons.

The Windows PE audit scans 94 binaries and follows the 64-binary eager dependency
closure of the application and Python extension. There are no eager CUDA/NVIDIA
driver imports in that closure; the application-owned nvcuda imports are delayed.

## Linux and remaining hardware coverage

The Linux Release combined CUDA+Vulkan application builds successfully with GCC
14 and CUDA 12.8 (the final build uses the native SM120 target; the earlier build used SM86/PTX86). An incremental final build
also succeeds. The complete ELF dependency audit follows 90 files: no unresolved
libraries and no eager `libcuda` or `libnvidia-ml` dependencies.

For the process-only driver-absence test, an `LD_PRELOAD` interceptor rejects
`dlopen("libcuda.so.1")`; system driver files are untouched. Application help
exits successfully. The fresh one-iteration smoke selects Vulkan, logs
`GPU fallback: requested=CUDA/Auto, effective=Vulkan, reason=no usable CUDA GPU`,
and saves its final project. Its adapter is Mesa 25.2.8 llvmpipe (LLVM 20.1.2),
a CPU Vulkan implementation. Before the final correction, a longer three-iteration smoke **failed** with
SIGSEGV in a driver execution thread while the application waits for
`Tensor::append_gather` during MCMC growth. It also fails with a fixed Gaussian
cap and with the Khronos validation layer enabled; no validation message is
reported before the crash. This does not establish whether the cause is a driver
or application shader defect from that trace alone. The later causal investigation and compiled correction below supersede this crash result.
The one-iteration result is only loader/fallback/save evidence, not proof of
stable Linux Vulkan training. The Linux CUDA control completes three iterations
with Gaussian growth and saves successfully. Native Radeon Linux training still
requires validation on a machine exposing the actual hardware driver.

All 13 Linux selection-policy tests pass. Both device-transfer tests are skipped
because this WSL setup exposes no Vulkan adapter compatible with the rendering
requirements; they are not counted as passed. The corresponding physical-device
checks pass on Windows. Native Radeon Linux execution and two physical CUDA
cards remain outside the available hardware coverage.

### Follow-up isolation without rebuilding

The available WSL environment does not expose a Vulkan hardware adapter for the
Radeon or NVIDIA GPU. Its llvmpipe software driver is not a substitute for native
Linux GPU validation. The software-driver crash therefore must not be presented
as evidence that Radeon/NVIDIA Vulkan training fails on Linux hardware, or used
alone to classify the branch as a hardware-backend regression.

Further process-isolated runs reuse the existing pre-merge executable only:

- Explicit Vulkan selection without `LD_PRELOAD` still reproduces the crash.
  The artificial CUDA-driver interception is not necessary to trigger it.
- A three-step configuration with normal refinement intervals (`start_refine=500`,
  `refine_every=100`, no automatic step scaling) completes and saves successfully.
- The same configuration, changing only `start_refine=0` and `refine_every=1`,
  crashes with SIGSEGV. This narrows reproduction to work triggered by refinement;
  it does not identify the exact failing dispatch within asynchronous execution.
- `tensor_masking_ops.cpp`, `vk_ops_index.cpp`, `index.slang`, `vk_recorder.cpp`,
  `vk_memory.cpp`, `mcmc.cpp` and `adam_optimizer.cpp` have no branch diff against
  the newly merged dev. That is evidence about patch scope, not proof that the
  failure is a driver bug or that dev itself passes this reproduction.

The main-thread backtrace waits in `append_gather` via Adam/MCMC; a queued earlier
operation could also be responsible. Absence of a validation message does not
rule out shader address errors. At this diagnostic stage no kernel change or software-device
blacklist was added. The subsequent isolated operation and compiled corrections
are described below.
Native Linux hardware testing remains separate and outstanding.

Evidence: `linux-explicit-vulkan-stack.log`, `linux-explicit-no-refine.log`,
`linux-explicit-with-refine.log`, and the paired `linux-*-refine-config.json` files
in the local validation artifact directory.

### Causal isolation of the llvmpipe crash

A subsequent investigation uses only the same pre-merge executable and GDB.
There is no application/shader compilation, binary patch on disk, or driver
installation/change. The earlier main-thread wait in `append_gather` was not the
failing operation: Adam's previously queued joint-moment zero encoding is involved.

The detailed trace shows the first parameter gather completing, followed by
extension of its optimizer state, then the next gather waiting when the worker
faults. The failing generated CPU instruction is
`vbroadcastss (%rsi,%r8,8), %ymm0`, with `rsi=0` and `r8=80`: it attempts to read
address `0x280`. This matches an optional float2 bounds read for block 80.

`encode_zero` initializes `Params` to zero and sets `operation=2`; it supplies
valid packed moment and moment-bounds buffers, but no value-bounds buffer.
`loadOldBounds` in `adam.slang` also loads optional value bounds, although its
value is not consumed by operation 2. Disassembly of the existing `adam.spv`
shows a nonzero test and `OpBranchConditional` guarding that optional load.
Thus the SPIR-V contains the null guard; the observed software execution still
performs a null-based read. This strongly implicates lowering/execution of this
conditional optional read in the installed llvmpipe/LLVM path. It is not proof
that every Mesa version or native GPU driver has the same behavior.

Controlled debugger experiment, limited to `operation=2` with null value bounds:

| Process | Debugger intervention | Result |
|---|---|---|
| Negative control | Identical breakpoints; reads fields but changes nothing | SIGSEGV after first observed zero-encode launch |
| Diagnostic address substitution | Sets only `Params.value_bounds=Params.bounds` before parameter upload | Normal exit, three completed steps, 15 substituted zero-encode launches; 19,555 to 22,635 Gaussians |

The diagnostic pointer references a buffer already present in the operation's
read/write tracking. A float2 read at block b fits inside that buffer's float4
bounds allocation. The optional value read is unused by this operation; no
optimizer formula, precision, sample count or Gaussian cap is changed. This is
causal diagnostic evidence, not a production regression/quality certification.
GDB itself can return zero after stopping on SIGSEGV, so the negative-control
result is classified from the inferior's signal, not the shell's GDB exit code.

At the time of causal diagnosis, the Vulkan Adam adapter and shader are unchanged
against merged dev; the stored
control/candidate SPIR-V audit also lists Adam among the unchanged binaries.
The reproduction exposes the pre-existing optional-read code path on llvmpipe;
the targeted follow-up adapter correction is described below.
Interactions with the broader
runtime still require a matched dev reproduction before claiming a full
baseline-level exclusion of regressions.

A narrowly scoped candidate is to provide an already valid bounds address for
the unused value-bounds field in Vulkan `encode_zero` only, with a comment
explaining why the address must be safe even though the value is unused. That
leaves CUDA, native Metal and other Adam operations untouched. It was initially
kept as a diagnostic candidate; the subsequent source correction is recorded
below. Compiled execution and numerical/Metal+Vulkan regression checks remain
necessary when builds resume. No software-GPU
blacklist, changed hardware selection or global driver override is introduced.

Evidence: `linux-refine-debug-detail.log`, `linux-gather-addresses.log`,
`adam-existing.spvasm`, `linux-adam-safe-address-diagnostic.log`,
`linux-adam-readonly-control.log`, and their `.gdb` / `.sh` harnesses under
`.local/gpu-backend-preferences`.

### Final compiled corrections and numerical contract

The initial host-only `encode_zero` pointer substitution was superseded: the
expanded software suite also reproduced the optional read in FP32 SH Adam.
The final correction is in the shared Vulkan shader helper `loadOldBounds`.
It selects a valid physical address before loading: real value bounds use
`block * 8`, while the absent-buffer fallback uses the current group's moment
bounds at `block * 16`. Reading its own block avoids racing another group's
in-place bounds update. The logical absent value remains zero and the original
`valueBounds` flag retains its codec-selection meaning. There is no additional
allocation or change to optimizer arithmetic. CUDA and native Metal kernels
are unchanged by this correction; Mac Vulkan uses the corrected helper.

The software suite then isolated the same speculative-null-load behavior in
radix sort's optional indirect-control read. The direct path now supplies its
nonempty, immutable source payload as a safe physical read address and retains
its original element/partition counts. The indirect path retains its control
buffer and computed counts. Both pair-sort variants use the same protection.
Every stage declares the extra source-payload read to the recorder, preserving
the dependency on the preceding pass and avoiding reliance on implicit ordering.
GDB evidence is retained in `linux-morton-crash-diagnostic.log`.

The portable CPU oracle now covers the exact growth geometry (19,555 to 20,532
primitives), comparing all packed bytes and bounds for row/swizzled layouts,
8/16-bit codecs, partial block 80, untouched rows and padding. Existing 700-row
fixtures keep their inputs. An additional FP32 SH test covers skipped rows.

Expanded physical-GPU tests exposed three quantization codes exactly at FP32
half-rounding boundaries. The two permutation assertions now use the already
established one-code/one-FP32-ULP boundary contract from this test file, with
independent CUDA-to-CPU and Vulkan-to-CPU comparisons. All nonboundary codes,
padding and bounds retain strict checks. This changes test assertions, not
GPU math. Diagnostic scaled values were 14309.5, 37461.5 and 35179.5; evidence
is retained in `portable-boundary-diagnostic-amd.log`.

### Post-merge dependencies and smoke-test discipline

The merged TIFF allocation-failure regression initially failed against the
machine's older unpatched shared vcpkg TIFF installation. Private dependencies
were built with dev's existing TIFF overlay and linked/staged for validation;
no shared vcpkg installation was changed. Windows and macOS standalone codec
CTest suites each pass all five registered tests with that dependency.
Linux application builds use the same patched TIFF library; public TIFF headers
were verified byte-identical to the existing prefix. The optional standalone
Linux EXR fixture additionally requires `openexr[cpp]`, which is absent from this
Core-only production prefix; its configuration failure is an environment
limitation, not a passing Linux codec-suite result.

Fresh 200-step training checks use max width 1920, SH0 and a one-million cap,
and require both completion and a saved checkpoint. They exercise CUDA and
both physical Windows Vulkan adapters, plus native Metal and Mac Vulkan.
A first nominal Mac Vulkan smoke profile lacked `schema_version=2` and was
migrated to Auto/Metal; that run is excluded from Vulkan evidence. Corrected
profiles explicitly verify the actual backend in the log. These short-run
wall times are functional evidence, not controlled iteration benchmarks.

### Final results after rebuilding

| Platform/configuration | Completed | Skipped | Disabled | Failures |
|---|---:|---:|---:|---:|
| Windows portable operators, NVIDIA | 50 | 32 | 3 | 0 |
| Windows portable operators, Radeon | 50 | 32 | 3 | 0 |
| Windows complete UI | 352 | 5 | 2 | 0 |
| Mac Metal graphics, Metal tensors | 1235 | 177 | 5 | 0 |
| Mac Metal graphics, Vulkan tensors | 1236 | 176 | 5 | 0 |
| Mac Vulkan graphics, Metal tensors | 1234 | 176 | 5 | 0 |
| Mac Vulkan graphics, Vulkan tensors | 1235 | 175 | 5 | 0 |
| Mac native Metal-only general suite | 750 | 418 | 4 | 0 |
| Mac Metal graphics UI | 398 | 4 | 2 | 0 |
| Mac Vulkan graphics UI | 404 | 3 | 2 | 0 |
| Mac Metal-only UI | 399 | 3 | 2 | 0 |

All 13 selection-policy tests pass on Windows, Linux and each Mac build.
The additional Mac Adam/SH/Morton CPU-oracle runs pass 32/32 in each combined
build, and 16 with 16 unavailable Vulkan cases skipped in Metal-only.
Windows CUDA byte-contract tests and both adapters' cross-device transfers pass;
Linux CUDA byte-contract tests also pass. The final WSL ELF audit follows 90
files with no missing dependencies or eager CUDA/NVIDIA driver imports. Windows
PE audit follows the eager dependency closure without eager driver imports.

Final 200-step Windows functional runs complete and save one-million-Gaussian
checkpoints with CUDA, NVIDIA Vulkan and Radeon Vulkan. Mac Vulkan's final
verified run also completes and saves.
The previously verified native Metal run completes and saves as well.
CUDA shutdown also emits the nvImageCodec `cuDevicePrimaryCtxRelease failed: 4`
message already present in the immutable baseline training/evaluation logs; it
is not counted as a newly introduced warning. These
are not directly comparable timings or new quality benchmarks.

The exact three-step, refine-every-step reproduction also completes and saves
22,635 Gaussians after rebuilding.

The rebuilt WSL application now completes the three-step driver-absence training
smoke, saving 21,558 Gaussians. Morton sorting and all Adam regressions also
complete without crashes. The original strict portable suite reports 14 passed,
20 unavailable cases skipped and **three numerical failures**: Q16 decode,
range decode and joint permutation. The oracle explicitly uses `std::fma` to
match CUDA; llvmpipe instead executes these pre-existing shader expressions
without contraction. A separate, artifact-only diagnostic changes only the
oracle decode to a volatile multiply followed by addition: **17 passed,
20 skipped, zero failures**, including the previously failing joint bounds/code.
The repository's strict oracle remains unchanged. The relevant SH/joint-Morton
SPIR-V modules are byte-identical to the control. This isolates the mismatch to
fused/nonfused FP32 results in unchanged codec paths, rather than treating the
original failures as passes or loosening production precision. Native Linux
GPU behavior still requires physical-driver validation.

Actual MCP integration with an isolated profile hides CUDA and requests a
missing saved rendering adapter. It verifies Vulkan fallback, preserved saved
requests, CLI `GPU fallback:` reasons, and the effective GPU names. Changing the
tensor adapter leaves the actual GPU unchanged, sets `restart_required=true`,
and displays the amber **Restart needed** action in the main status bar.
Restoring the active configuration clears it and logs `restart required=false`.
Evidence: `runtime/final-fallback-restart-4`. Consolidated final results and
source hashes are retained in `final-physical-gpu-validation-summary.json`.

The same saved one-million-Gaussian project renders successfully in all four
cross-device viewer combinations. Offscreen images are byte-identical when the
rendering adapter is held constant while the tensor adapter changes. Across
rendering adapters, mean absolute channel difference is 0.03124/255 and maximum
8/255. Window captures were visually inspected.

The final runtime source files were hash-checked against the Mac copy;
`final-runtime-source-manifest.json` records the matching contents. Final source
gates report zero backend-neutrality, I/O-discipline, CUDA-format or error-debt
violations, without baseline changes. Historical 2,000-step quality/timing data
remain above; they predate these final safe-address guards. No absolute absence
of regressions, full Linux codec-suite pass, native Radeon Linux pass, or
physical multi-CUDA-card validation is claimed.

## Evidence locations

Detailed logs, profiles, CSV traces, JSON metrics and contact sheets are under
`.local/gpu-backend-preferences` in the original workspace. Relevant groups are
`training-windows-*`, `mac-results`, `quality-comparison`, `runtime/large-final-*`
and `runtime/large-training-*`. No validation-layer/VUID-clean claim is made;
the Khronos validation layer is not installed on the Windows test machine.

### Device-menu follow-up

The normal profile still requested Radeon UUID `00000000c70000000000000000000000`
at startup, while current Vulkan enumeration reports `00000000c80000000000000000000000`.
The old missing-selection option was mistaken for a second Radeon. Menus now
contain only currently enumerated devices and Auto. A missing saved identity
maps to Auto for display, while its saved preference and CLI fallback remain.
Names continue to come from Vulkan; real distinct adapters sharing a name are
not merged. Eight Python preference tests pass, including the exact c700/c800
case and multi-adapter identities. The plugin-copy build target was updated;
an isolated live MCP run confirms both role menus contain exactly Auto, one
`AMD Radeon(TM) 890M Graphics` and one `NVIDIA GeForce RTX 5060 Ti`.
Evidence: `runtime/gpu-inventory-no-phantom/inventory.json`.

The four gsplat/Fast forward and backward shader sources are identical to the
previous `fix/vulkan-amd-validation` branch. The current dispatch retains the
AMD-compatible backward wave32 policy. Prepared-image CPU caching and the
full-resolution-source limit are present. This does not establish that every
historical experiment was retained, or that a new optimization will be faster;
forward/backward substage measurements remain the next justified performance
investigation. No training kernel is changed by this menu correction.

## Stable sort performance follow-up

Two independent controls and two candidates resume the same 1M-Gaussian CUDA
checkpoint from step 200 to 400, with max width 1920 (actual tensor 1440 x 1920),
SH0, MCMC, CPU image cache, Fast/3DGS rasterization and evaluation disabled.
The optional ballot sort uses core subgroup operations on eligible AMD devices;
existing generic and NVIDIA-match SPIR-V modules are byte-identical to controls.
The candidate preserves stable integer key/value ordering and capacity tails.

| Configuration | Steady step, run 1 / run 2 (ms) | Whole process, run 1 / run 2 (s) |
| --- | --- | --- |
| Generic stable sort | 173.088 / 174.050 | 60.812 / 58.656 |
| Ballot stable sort | 149.763 / 148.069 | 55.484 / 51.594 |

The mean step decreases 14.2%; mean whole-process time decreases 10.4%.
These are 200-step collector means, not the historical 2,000-step comparison.
Loader waits are separately reported (96.6-108.0 ms/step across these runs),
and asynchronous phase spans must not be summed to reconstruct total time.
Peak backend allocations are about 1.66-1.67 billion bytes in both paths.
An isolated 10M-key test decreases sorting GPU time from 75.027 to 14.027 ms;
that factor is not a whole-training speedup.

Forcing forward wave32 in addition to ballot sort yielded 149.013 / 144.015 ms,
versus 149.763 / 148.069 ms without it. This small, variable increment is not
retained. Incorrect early probes (the gsplat-stage control and the candidate
before its host selector was wired) are excluded. Definitive runs occurred
after the unrelated local build finished. Profiling hooks were restored out
of production sources. Evidence: `amd-forward-optimization/whole-step-comparison.json`.
These timing results alone do not establish final trained-model quality parity.

## Forward/backward research and next experiments

Research checked on 18 October 2026; candidate ideas below are not claims of
implemented or measured improvements. External implementations are design
references only. No external kernel is copied.

### Existing mechanisms confirmed in this branch

`fast_forward.slang::project` already counts ellipse/tile overlaps using
`tileSpan` scanlines. `rasterSubgroup` already uses lane broadcasts, conservative
8x4 subtile tests and early transmittance termination. `fast_backward.slang`
already reuses final transmittance/last contributor and reduces gradients in
logical 32-lane groups before atomics. `dispatch.cpp` already requests wave32
for eligible AMD raster-backward pipelines. `atomicFloat` already has separate
native FP32-atomic and CAS fallback modules. Merely enabling any of those again
is not a new optimization. SH storage already has a subgroup-aligned swizzle.

AMD recommends balancing register pressure, LDS and occupancy, profiling wave
size choices and avoiding redundant submissions/barriers. LDS layout matters:
its RDNA banks are 32-bit wide, and structure-of-arrays layouts can help.
These recommendations motivate experiments, not guaranteed gains on this APU.
Sources: [RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/),
[occupancy explanation](https://gpuopen.com/learn/occupancy-explained/),
[wave-size guidance](https://gpuopen.com/presentations/2023/RDNA3_Beyond-the-current-gen-v4.pdf).
Khronos requires checking supported operations/stages and subgroup-size-control
properties: [Vulkan subgroups](https://docs.vulkan.org/guide/latest/subgroups.html).

### Prioritized experiments preserving FP32 formulas

| Priority | Experiment and concrete code location | Invariants and evidence required |
| --- | --- | --- |
| 1 | Forward-only 128/64-thread variants instead of 256: `fast_forward.slang::rasterSubgroup`, `dispatch.cpp`, indirect raster arguments in `fast_vulkan.cpp`. Divide a 256-pixel tile among independent groups while retaining each logical warp's pixel ranks. | Same per-pixel Gaussian order, threshold comparisons and FP32 expressions. Keep scan/projection workgroup contracts unchanged. Compare image/alpha/depth/normal/transmittance/last-contributor bits on fixed inputs, odd sizes, empty tiles, early termination and partial batches. More groups may cost more than they save. No measured gain yet. |
| 2 | Coalesced projected-data access: `fast_common.slangh::Projected`, `projectedAt`, producer and workspace accounting. Current record is 18 words (72 bytes); adjacent lanes fetching different Gaussians use that stride. Prototype a full-FP32 field-major or AoSoA scratch layout. | Copy bits without new packing/quantization; preserve original indices and stable order. Include projection writes and backward reads in the timing, not only raster time. Check unused depth/normal fields in compiler output before assuming they consume bandwidth. Size/padding and 64-bit address bounds require tests. |
| 3 | Backward live-state specialization: `fast_backward.slang` has two pixel states and 16 accumulators per lane, plus 8 KiB reduction scratch and maxLast. Inspect actual compiler resource usage for RGB-only versus depth/normal/densification variants; shorten variable lifetimes or split optional fields only where retained. | Preserve each field's XOR reduction tree, clamp location, native atomic behavior and all enabled auxiliary gradients. Do not replace the tree with unordered WaveActiveSum. Compiler may already eliminate unused fields; inspect generated code first. Do not blanket-pad `reduction[field][lane]`, whose lanes are already contiguous. |
| 4 | Stage readback and recorder stalls: deferred forward performs a status readback after sort/ranges, before raster. Profile the CPU wait separately; inspect resource dependencies and avoid redundant read-to-read barriers only where proven. | Preserve fault reporting, overflow replay and no partial output publication. GPU validation cannot simply be removed. Recorder already batches commands, so a blanket batching rewrite is unjustified. Cross-device viewer transfers and queue lifetimes must remain covered. |
| 5 | Projection SH/layout and uniform metadata reads: `fast_projection_backward.slangh`, `fast_common.slangh`. Consider read-only, wave-uniform loads and remaining scattered FP32 accesses. | Only broadcast genuinely uniform data; retain SH basis order and quantized storage decode exactly. SH0 timings cannot justify an SH3 optimization; test all degrees and existing storage modes separately. |

The Radeon 890M uses shared physical RAM. This is not the discrete-card PCIe
case in external benchmarks. Current `vk_memory.cpp::ensure_staging` already
prefers coherent, host-visible device-local mapped staging with a fallback.
Investigate measured loader/transfer waits without changing CPU image-cache
semantics or assuming the integrated GPU has the same transfer bottleneck as
a discrete Radeon. Pipelining uploads must preserve image order and prepared
image contents, concurrency and memory budgets.

### Larger alternatives requiring a separate numerical decision

VkSplat demonstrates two raster-backward organizations (per-Gaussian and a
shared-memory formulation), adaptive selection and fused projection/optimizer.
Its [paper](https://arxiv.org/html/2605.00219v1) reports quality consistency,
not bitwise identity to our rasterizer. The public
[per-splat kernel](https://github.com/harry7557558/vksplat/blob/main/vksplat/slang/alphablend_shader_bwd_per_splat.slang)
stages pixel state and accumulates per-splat gradients; the
[tensor kernel](https://github.com/harry7557558/vksplat/blob/main/vksplat/slang/alphablend_shader_bwd_tensor.slang)
uses another shared-data organization. Either can change reduction order,
transmittance replay and atomic scheduling. Do not port these as a supposedly
exact replacement. First quantify our backward substages and preserve all
normal/depth/regularization features absent from the reference comparison.
Likewise, fusion with Adam must preserve gradient normalization, regularizers,
clamps, moments, skipped updates and every consumer of materialized gradients.

A second public Vulkan trainer documents staged Gaussian loads and a cached
forward-state backward replay: [Slang Splat rendering documentation](https://github.com/MichaelMoroz/slang-splat/blob/main/doc/Rendering.md).
Its camera model, tile layout, losses and optional fixed-point gradient path
differ from ours. Its Vulkan target is useful architectural evidence; this
review does not establish performance or Radeon validation for that project.

Excluded under the current precision requirement: FP16/fixed-point gradients,
new depth-key quantization, approximate exp/log, looser alpha/transmittance
thresholds, dropped pixels or SH bands, and changed densification schedules.
Existing configured storage formats remain separate from introducing any new
precision reduction.

### Acceptance sequence

First capture uninstrumented controls from the same checkpoint and profile
raster-forward, raster-backward and projection-backward with GPU timestamps
without introducing a host wait between every dispatch. When available, inspect
VGPR/SGPR counts, scratch spills, LDS allocation/bank conflicts and cache traffic;
source-level array sizes alone are insufficient to identify an occupancy limit.
Keep instrumentation outside the shipping path.

Change one factor at a time. For memory/layout/work distribution, require exact
fixed-input forward outputs and contribution lists. For backward, preserve the
reduction tree and compare gradients against the CPU/finite-difference tests;
repeat controls establish existing atomic nondeterminism. Do not relax test
tolerances to make a candidate pass. Exercise native and emulated atomics,
SH0-3, depth/normal, image background, densification, invalid inputs, capacity
overflow/replay and cross-device training/viewing. Then compare repeated
200-step timings and the agreed 2,000-step quality protocol (same images,
checkpoint, evaluation backend and visual crops). Keep only gains larger than
control variability. Recheck NVIDIA Vulkan, CUDA and Mac Metal/Vulkan before
promoting any candidate; retain existing paths for unsupported devices.

## Final raster-forward validation and retained changes

The initial sort study above is preserved. Its first wave32 sample was inconclusive; a subsequent replicated study justified retaining the narrower subgroup for compatible AMD Fast raster forward pipelines. No floating-point shader source, CUDA kernel or native Metal kernel was changed. Forward needs eight supported subgroups per workgroup (256 / 32); the existing backward requirement remains four. Unsupported devices retain their existing policy, and a rejected optional pipeline retries the default subgroup with a CLI warning.

Four 200-step controls averaged 148.221 ms per training step; three wave32 runs averaged 137.335 ms, a 7.34% reduction. Mean complete process time was 55.664 versus 53.339 seconds (4.18%). These runs resume the same step-200 checkpoint and collect all 200 steps; they are not 180-step warmup-filtered samples. Separate experiments must not be compounded into a single speedup claim.

### Paired 2,000 additional steps

Both runs resume the same immutable step-200 checkpoint and finish at step 2,200: MCMC, Fast/3DGS, SH0, one-million Gaussian cap, CPU image cache, maximum image dimension 1920 (actual 1440 x 1920). The control already includes the ballot sort; the candidate additionally enables the retained forward subgroup policy. Step time excludes loader wait; process time includes startup and saving.

| Measurement | Control | Candidate |
|---|---:|---:|
| Mean step | 125.320 ms | 114.591 ms |
| Mean loader wait | 10.399 ms | 9.936 ms |
| Complete process | 276.328 s | 253.641 s |
| Peak backend allocation | 1,660,760,320 bytes | 1,668,604,160 bytes |
| PSNR | 17.290001 | 17.299679 |
| SSIM | 0.408864 | 0.409086 |
| LPIPS | 0.634722 | 0.634551 |

Step time decreases 8.56%; complete process time decreases 8.21%. No additional scratch buffer was introduced. The measured peak differs by about 7.5 MiB; this is not a measurement of identical physical residency. The 2,000-step loader average benefits from cache warmup and should not be confused with early 200-step averages.

The same final CUDA evaluator compared 11 views of each model. All 82 images were used for training: these evaluated views are **not disjoint held-out data**, even though the evaluator labels its split `test`. This measures relative fidelity, not generalization. Mean metric changes are tiny; the worst per-view PSNR delta is -0.054865 dB. Mean absolute rendered-channel difference is 1.411/255. Both short SH0 runs remain visibly soft and under-converged. Render contact sheets and face crops were inspected for face detail, silhouette, foreground base and foliage: no added visible degradation was observed. These observations do not establish bit-identical trained models or an absolute guarantee for every scene.

Nine fixed-input forward scenes, including odd partial tiles, produce byte-identical image, alpha, depth, normal, transmittance, contribution and range outputs. Gradient finite-difference checks pass without relaxed tolerances. Earlier 200-step sort-only quality differences are smaller than repeated-control variation. Smaller workgroups, alternate projected-record layouts, explicit backward shuffle rewrites and cached culling thresholds were investigated but are not included: they did not establish a useful additional gain.

### Final failure and platform checks

Seven artifact-only fault probes pass: shader creation failure, layout creation failure, initial and partial compute-pipeline failures, DeviceLost propagation, forward subgroup rejection, and insufficient forward subgroup capacity. Failed shader/layout outputs deliberately contained invalid handles; none was destroyed. Optional failures log fallback; DeviceLost is propagated. No injection hooks or profiling waits remain in application sources.

- Windows portable tests on each physical Vulkan adapter: 54 passed, 32 skipped, three disabled, zero failed. The final nine-scene forward test also passes on both adapters. Policy, cross-device interop and preference/UI checks pass. Final three-step CUDA, NVIDIA Vulkan and AMD Vulkan training smokes complete successfully.
- Mac final general suites (passed/skipped/disabled): Metal graphics + Metal tensor 1236/177/5; Metal + Vulkan 1237/176/5; Vulkan + Metal 1235/176/5; Vulkan + Vulkan 1236/175/5; Metal-only 750/418/4. All have zero failures. Corresponding UI suites also have zero failures. Final Mac Vulkan training smoke completes. Successful backend math and device selection are preserved; shared pipeline error cleanup is hardened.
- Combined Linux CUDA/Vulkan application builds. Its complete 90-object ELF dependency audit finds no eager CUDA-driver dependency and no unresolved library. With driver loading blocked, help/startup and three training steps with refinement complete on llvmpipe. Policy: 13 passed. Stable pair sort including indirect counts and untouched tails: three passed. Two physical-device interop cases skip. WSL provides no physical Vulkan adapter, so this does not establish native Linux hardware performance.
- The unchanged strict Linux floating-point CPU oracle still has three known fused/unfused precision failures (14 passed, 20 skipped, three failed). An artifact-only unfused diagnostic passes 17 with 20 skipped; shipping tolerances and oracle were not changed. This limitation is not reported as a green strict suite.

Backend neutrality, I/O discipline, CUDA log format, error-debt baseline and whitespace checks pass. A pre-existing CUDA context-release shutdown warning also occurs in earlier controls; successful training/evaluation exits and checkpoints are verified.

Evidence is under `.local/gpu-backend-preferences/raster-layout-experiments`: `wave32-replicated-summary.json`, `quality2000-*-summary.json`, `quality2000-metrics-comparison.json`, `quality2000-render-differences.json`, `quality2000-contact-sheet.png`, `quality2000-face-crops.png`, `final-exact-forward.json`, `failure-cases/summary.json`, `final-test-summary.json`, `final-training-smokes.json`, and `linux-final-elf-audit.json`. Its README identifies historical probes that must not be rerun against the final binary. Earlier sort evidence remains under `amd-forward-optimization`.

## Progress since the initial implementation

This history includes image preparation and cache corrections already merged in
#3601, supported by the shared JPEG codec from #3637. Those are prerequisites in
`dev`, not new image-cache changes introduced by this branch. Subsequent upstream
cache/codec improvements are likewise inherited through the `dev` merges.

The first preserved real-dataset AMD run on 15 October (`gpu-validation-dev/outputs/perf-real-amd/perf_bench.json`, application source tag `43a9af65a-dirty`) records 1,036.650822 ms per steady training step plus 371.361742 ms of steady image-loader wait: approximately **1,408.013 ms per iteration** for those two components. This was a 55-step run with five warmup steps and 50 measured steady steps, all 82 dataset images, SH0, 19,555 live Gaussians and a two-million capacity limit. The earlier incorrect handling of prepared images and large allocations was still present. The saved loader configuration, log and benchmark JSON remain available locally.

The final 2,000-step candidate records **114.590912 ms per training step plus 9.935835 ms of loader wait**, approximately **124.527 ms per iteration** for the corresponding components, with the one-million capacity limit and resized images described above. These endpoints document progress from the beginning of implementation, including memory, image preparation/cache and kernel execution improvements. They are different workloads and collection windows, so their ratio is not an isolated optimization benchmark. Startup, checkpoint saving and evaluation are not included in either sum. The controlled paired comparison remains the separate 125.320-to-114.591 ms result above.

## Publication integration against the latest dev

Upstream advanced to `c002ec1b47ebd61cbdaa410f7ca798cdd98518db`. Merge
`a1cb9d345` incorporates #3670, #3672, #3673 and #3678 without conflicts.
Their image-codec and prepared-image cache algorithms are inherited from `dev`;
this branch does not reimplement them. The combined Windows application, UI,
policy, interop and portable test runner rebuild successfully after this merge.

The expanded integration run exposed an image-helper dispatch inconsistency:
a CUDA loader inside a Vulkan-default process allocated CUDA tensors within its
scope, but CPU-decoded mask operations and masked-fill guards still consulted
the process default. The shared helpers now use the scoped execution target and
the actual staging tensor. The CUDA image execution queue and cached-decode
choice follow the same target. Unscoped execution keeps the existing default;
no image formula, kernel, precision or cache encoding algorithm changes.

The existing `MasksMatchHostReference/CUDA` case reproduced the failure with
`--tensor-backend=vulkan`; it passes after the correction, including binary
and undistorted masks. The before-correction JSON/log are retained. A preliminary
artifact runner also stalled in a death-test subprocess; switching to the
repository's unchanged `test_main.cpp` initialization resolves that harness
issue. No test-runner workaround is added to production or repository tests.

Final additional results:

| Check | Passed | Skipped | Disabled | Failed |
|---|---:|---:|---:|---:|
| Windows policy | 13 | 0 | 0 | 0 |
| Windows CUDA/Vulkan source interop, default adapter | 2 | 0 | 0 | 0 |
| Windows Radeon-source interop | 1 | 0 | 0 | 0 |
| Windows UI | 352 | 5 | 2 | 0 |
| Windows Radeon image-loader/cache subset after dispatch correction | 32 | 42 | 0 | 0 |
| Windows NVIDIA full portable operators + loader/cache after correction | 87 | 74 | 3 | 0 |
| Mac Metal-default image-loader/cache subset after correction | 15 | 32 | 0 | 0 |
| Mac Vulkan-default image-loader/cache subset after correction | 15 | 32 | 0 | 0 |

The Radeon full expanded run before the image-helper correction passed the
operator cases but exposed the single CUDA-mask failure; the complete affected
loader/cache subset was rerun afterward. The NVIDIA run afterward includes both
operators and cache cases. These are functional checks, not new timing samples.
The targeted Mac reruns validate the shared scope correction on the existing
combined Metal-graphics build; they do not replace the earlier full Mac matrix
or claim a rebuild of the latest upstream codec changes on Mac.

Policy, cross-device interop, UI and three-step CUDA/NVIDIA-Vulkan/Radeon-Vulkan
training all pass after the upstream merge. The three training configurations
are checked again after the final image-helper change. Backend-neutrality,
I/O-discipline, CUDA-format and error-debt gates remain clean, with no baseline
changes. Final Linux and full Mac matrix results above predate this publication
merge; no new native Linux GPU coverage or performance measurement is claimed.
Evidence: `.local/gpu-backend-preferences/review-upstream` and
`review-final-mask`, including `final-summary.json`, `portable-summary.json`,
the archived failing mask reproduction and the final training `summary.json`.

## Local integration with dev on 20 October 2026

This follow-up remains local: the published PR head is `b52f84251` and is not
updated by this integration. The incoming snapshot is
`cc1dda56e12933bf27142b62b74e880d0abc7cff`, 36 upstream commits beyond the
previous merged `dev` base `c002ec1b4`.
The resolved merge is saved locally as `4bf0e5a3f`; the rendering/test adaptations
are kept in a separate follow-up commit.

The single textual conflict was at the end of
`tests/python/test_preferences_panel.py`. Both the GPU reset/restart tests and
the new upstream splat-drag preference tests are preserved. The other shared
changes were reviewed after automatic merging, including Preferences and all
ten locales, Python UI bindings/stubs, the window/input lifecycle, CUDA image
loading, independent-device transfer gates, RAD/LOD streaming and FastGS/MRNF.

### Semantic integration and local corrections

- Preserve upstream MRNF growth exchange, live crop-box/ROI updates, fisheye/GUT
  corrections, JPEG XL support, native drag/scrubbing and Metal RAD/LOD changes.
  The new Fast backward impact fields increase the parameter block from 576 to
  592 bytes; the host and Slang layouts agree. The added specialization bit 23
  does not overlap the existing low-four-bit stage selector. Existing forward
  and backward subgroup limits remain eight and four respectively.
- Correct LOD memory-budget ownership: the pool is allocated by the Vulkan
  viewer, so its free-memory estimate now queries that viewer's physical
  device and VMA allocator, not the default tensor backend. Aggregate only
  device-local heap budgets/usage and clamp subtraction to zero, matching the
  existing conservative memory-counter convention. Keep upstream page floors,
  budget fractions, current-pool adjustment, quantization and feedback policy.
  This matters when training and viewing use different GPUs/vendors. The
  unavailable-budget fallback still logs and uses the minimum pool.
- Update the separate keymap test's UI mock with the GPU preference/inventory
  APIs now required by the parent panel. Its language test continues to require
  exactly one keymap invalidation while exercising that section independently
  of the parent panel's separate full refresh. Production UI code is unchanged.
- Align the existing CUDA ROI byte-contract test with the new upstream kernel
  signature by passing null rays/depth/alpha for its original pinhole case.
  The old call site no longer compiled against the new declaration. No kernel,
  numerical tolerance or expected byte comparison is changed.

### Local verification

The combined CUDA/Vulkan Windows application and UI/policy/interop targets build.
The artifact-only portable runner includes the current upstream tests for ROI,
training-image operations and MRNF, as well as the existing Fast/gsplat, Adam,
stable sorting, queue readback and prepared-image/cache contracts. Its fixture
paths and entry point follow the repository test configuration; no runner
changes are added to the product.

| Check | Passed | Skipped | Disabled | Failed |
|---|---:|---:|---:|---:|
| GPU/device preferences Python tests | 106 | 0 | 0 | 0 |
| Extended Python UI/localization/keymap/resources | 237 | 1 | 0 | 1 upstream |
| GPU configuration policy | 13 | 0 | 0 | 0 |
| Default source interop | 2 | 0 | 0 | 0 |
| Radeon source interop | 1 | 0 | 0 | 0 |
| Windows UI | 355 | 5 | 2 | 0 |
| Portable numerical/strategy/cache suite, Radeon Vulkan selection | 160 | 84 | 3 | 0 |
| Portable numerical/strategy/cache suite, NVIDIA Vulkan selection | 160 | 84 | 3 | 0 |
| Final LOD page-cache and budget-feedback contracts | 19 | 0 | 0 | 0 |

The extended Python row includes the 106 preference tests; it is not an
additional disjoint count. Its only remaining failure is
`test_every_scrollbar_uses_the_theme_size`: Graphite has a 4-pixel scrollbar
while the upstream test expects 6. Both the test and theme have exactly the
same Git blobs as this upstream snapshot. Neither is changed to conceal the
failure. Temporary-directory permission errors in an initial isolated test
attempt were resolved by using a workspace-local temporary directory.

The two portable selections also execute CUDA-specific fixtures where available.
The newly added removal-impact oracle, ROI masks, training-image contracts and
MRNF strategy cases pass. Only the two existing large pair-sort throughput
benchmarks are excluded; correctness/boundary/stability tests remain enabled.
These runs are functional validation and do not supply new performance claims.
The 19 LOD cases check pool/cache/feedback policy; a new large-RAD physical
residency comparison across the two adapters is not claimed by these tests.

Three-step resumed MCMC training completes and saves on CUDA, NVIDIA Vulkan and
Radeon Vulkan. Fresh six-step MRNF training with growth exchange enabled also
completes on all three, using images limited to 256 pixels and a 19,555-Gaussian
cap. These short smokes confirm initialization, impact-enabled execution and
saving; no actual growth-exchange victim/replacement log was emitted in these
six steps, so an end-to-end replacement event is not claimed. Separate strategy
unit tests exercise exchange selection/quota behavior.

Backend-neutrality, I/O-discipline, CUDA log format, error-debt and whitespace
checks pass without baseline changes. Logs and JSON counts are under
`.local/gpu-backend-preferences/merge-dev-20261020`.

No push, public comment, PR-body change or remote Mac source update is part of
this follow-up. Full Mac/Linux runtime matrices and long training-quality /
performance comparisons are not rerun here. Earlier tables describe their
original revisions and cannot certify the new upstream FastGS/MRNF kernels or
serve as matched timings for this new base. The extra local rendering correction
does not change CUDA/Metal math; upstream's own training changes are retained.
