# GPU device selection: validation on 18 October 2026

Local branch `feat/gpu-backend-preferences`, based on `dev` commit
`4ef50cbbc7f649cfed1d76d1d1770a3bff62d27f`. No push or pull request is part of this work.
The implementation and reproduction checklist are in [GPU device preferences](gpu-device-preferences.md).

The branch subsequently merges `upstream/dev` commit
`d23669db9de3d7cfc0b3517fb439428d65fb743d` through merge commit
`acb933295f2d02c03953230d095a0df7f165ed97`, without conflicts.
No compilation is run after this merge at the user's request. The measurements
and binaries below refer to the pre-merge source; they do not validate the merged revision.

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

## Numerical and integration checks

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

The 110 tensor SPIR-V binaries match the control byte for byte. Of 68 training
SPIR-V outputs, 64 match; the four changed outputs are the raw/optimized pair-sort
variants. Their correction counts a partially occupied subgroup with ceiling
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
14 and CUDA 12.8 (SM86/PTX86 in this WSL environment). An incremental final build
also succeeds. The complete ELF dependency audit follows 90 files: no unresolved
libraries and no eager `libcuda` or `libnvidia-ml` dependencies.

For the process-only driver-absence test, an `LD_PRELOAD` interceptor rejects
`dlopen("libcuda.so.1")`; system driver files are untouched. Application help
exits successfully. The fresh one-iteration smoke selects Vulkan, logs
`GPU fallback: requested=CUDA/Auto, effective=Vulkan, reason=no usable CUDA GPU`,
and saves its final project. Its adapter is Mesa 25.2.8 llvmpipe (LLVM 20.1.2),
a CPU Vulkan implementation. A longer three-iteration smoke **fails** with
SIGSEGV in a driver execution thread while the application waits for
`Tensor::append_gather` during MCMC growth. It also fails with a fixed Gaussian
cap and with the Khronos validation layer enabled; no validation message is
reported before the crash. This does not establish whether the cause is a driver
or application shader defect, and is an unresolved software-Vulkan limitation.
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
rule out shader address errors. No speculative kernel change or software-device
blacklist is added. A minimal isolated operation reproducer and a matched dev
control are the next useful diagnostics when further build work is authorized.
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

### Targeted source correction, not yet rebuilt

The follow-up applies `p.value_bounds = p.bounds` only after the zero-encoding
mark stage, immediately before operation 2 is queued in `adam_vulkan.cpp`.
Operation 3 keeps its original parameters. The valid address points to an already
tracked allocation; there is no extra buffer, allocation, host copy or change to
optimizer arithmetic. CUDA, native Metal and all shader sources are untouched
by this follow-up. Mac Vulkan uses the corrected Vulkan adapter and therefore
must be included in the compiled regression checks.

The existing portable Adam CPU-oracle helper now also accepts the exact growth
geometry (19,555 to 20,532 primitives). A new parameterized regression compares
all packed output bytes and bounds against that CPU oracle for row and swizzled
layouts, both 8-bit and 16-bit codecs. It covers the live partial source block,
new complete blocks, final partial block 80, and preservation of untouched rows
and padding. Existing 700-row fixtures retain their original inputs.

No compilation is run, in accordance with the user's current instruction. The
new C++ regression is **added but not executed**, and the source correction is
**not yet tested in a rebuilt binary**. The GDB causal result above is supporting
evidence, not a substitute for that validation. Backend-neutrality, I/O discipline,
CUDA log-format and error-debt source gates pass; debt baselines are unchanged.

When build work resumes, run the new test and all existing `EncodeZero*` CPU
comparisons on llvmpipe, both physical Windows adapters, and Mac Metal/Vulkan;
then repeat the previously failing refinement smoke on the rebuilt Linux app.
Do not count the old binaries' successful measurements as validation of this
new source revision.

## Evidence locations

Detailed logs, profiles, CSV traces, JSON metrics and contact sheets are under
`.local/gpu-backend-preferences` in the original workspace. Relevant groups are
`training-windows-*`, `mac-results`, `quality-comparison`, `runtime/large-final-*`
and `runtime/large-training-*`. No validation-layer/VUID-clean claim is made;
the Khronos validation layer is not installed on the Windows test machine.
