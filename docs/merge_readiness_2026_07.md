# Vulkan hardening merge readiness

Marker: `FINAL-MERGE-READINESS-2026-07-13`

Date: 2026-07-13

Branch: `vulkan-hardening`

Base: `origin/master`

Reviewed head before this report-only change: `7ea0020ba`

Scope at that head: 187 commits, 376 files, +32,668/-10,316

## Verdict: MERGE-WITH-FIXES

I would approve the code delta only with the four review fixes listed below, which are now on
the branch, and with explicit maintainer acceptance of the test and compatibility conditions in
this report. I would not describe the branch as having a green full test gate, and I would not
merge it as a routine patch release.

The strongest evidence in favor of merging is runtime evidence, not commit volume or the prior
execution log: the source-built Vulkan layer completed the single-PLY, multi-PLY, and GUI-training
flows with GPU-assisted validation, synchronization validation, and fatal-on-error enabled. All
three runs had zero VUIDs, synchronization hazards, shader data races, failure reports, or
application error lines. Two independent pre-branch v1 checkpoints also restored their payload,
strategy state, and training state with the current binary.

The strongest evidence against an unconditional merge is the test harness. The current branch
now completes the monolithic test binary, but 63 of 3,125 tests fail. `origin/master` is not a
usable green control: rebuilt in the same build tree, it records 58 failures and then segfaults in
`TensorLazyRuntimeTest`. Every one of the branch's 33 failures before its former crash point also
fails on master, and the branch fixes 25 master failures in that comparable prefix, but master
cannot complete far enough to prove that all later branch failures are inherited. Targeted
isolation found no branch-specific product failure, but that is weaker than a green full suite.

Before merge, the maintainer must therefore do both of the following:

1. Carry the four review fixes in this report and publish the configuration/API migration note
   below in the PR or release notes.
2. Stop calling the full suite green. Either repair/quarantine the known baseline failures and
   provision their assets, or formally approve a temporary named no-new-failure baseline with
   owners and an expiry. The order-dependent CUDA-state corruption must remain a tracked High
   issue even though this branch no longer crashes the process.

## What this review fixed

| Commit | Defect | Repair |
| --- | --- | --- |
| `8ae28137e` `fix(core): preserve CUDA allocation failure status` | The new centralized retry path converted any final `cudaMalloc` status into `cudaErrorMemoryAllocation`. Device/context/argument failures could incorrectly run pressure reclamation and surface as typed OOM. | Propagate the real final status through direct and pool allocation. Invoke pressure relief and `MemoryAllocationError` only for actual OOM. |
| `77a72ba6b` `docs(dev): correct failure report family contract` | `assertions.md` promised CUDA state and breadcrumbs for every report family, while generic contract, tensor contract, and terminate reports intentionally do not collect those sections. | Document the family-specific envelopes and reserve CUDA state/breadcrumb guarantees for CUDA reports. |
| `8e4fbcc80` `fix(core): migrate legacy checkpoint depth modes` | A real pre-branch checkpoint could not resume because its valid historical `depth_loss_mode="adaptive-warped-l1"` failed the narrowed current parameter contract. | Migrate the exact historical values `pearson` and `adaptive-warped-l1` to current `ssi`, with a warning. Unknown values still fail validation. |
| `7ea0020ba` `fix(tests): retire tensor streams before destruction` | Six lazy-runtime tests destroyed CUDA streams still referenced by the tensor pool. One emitted `cudaErrorDeviceUninitialized`; the monolithic test process later segfaulted. | Apply the production `CudaMemoryPool::release_stream` contract before destroying those streams. The 23-test lazy-runtime group now passes together with zero CUDA error lines, and the full process completes instead of crashing. |

Each code/test commit passed `cmake --build build -j6`, the exact 291-test cross-round filter,
and `pytest -q tests/python/test_async_plugin_loading.py` (10/10). The stream-lifetime fix also
passed all 23 `TensorLazyRuntimeTest` cases together.

## Correctness seam audit

### Pressure coordinator, allocator, diagnostics, and crash handling

The allocation path has one intended retry owner in `src/core/memory_pressure.cpp`. Tensor pool
allocation, direct tensor allocation, and capacity growth make one attempt, ask the coordinator
to reclaim registered caches on OOM, then retry at most once. The pool retains allocation
provenance and stream ownership. CUDA-unavailable latching fails closed before a new allocation
and after the coordinator race window.

The defect found here was at that ownership boundary: the helper returned only a null pointer,
so callers invented OOM as the cause. `8ae28137e` restores the native failure status. No other
status-collapsing caller was found.

The current production clients are intentionally narrow: tensor-pool trim and pinned-host cache
trim. There is no render-safe client, no trainer-safe client, and no production caller of
`MemoryPressureCoordinator::preflight`. The preflight implementation also executes reclaim
callbacks while the registry mutex is held, so callbacks must not re-enter registration or
preflight. That is latent scaffolding, not a production deadlock today, but it must be redesigned
before preflight is wired broadly.

Failure reports have bounded deduplication, bounded CUDA breadcrumbs, family-specific formatting,
and a process-wide CUDA-unavailable latch. Unit tests cover classification, latch behavior,
deduplication, ring wrap/thread smoke, and generic-versus-CUDA report sections. They do not run
the POSIX or Windows crash handler in a death/subprocess test. Windows unhandled-exception and
minidump behavior is unproved.

One deferred defect remains material: `VramProfiler::recordAllocation` and
`recordDeallocation` may allocate and throw, but some FastGS/arena callers are in `noexcept`
cleanup. The correct follow-up is a bounded, transactional, internally non-throwing diagnostic
API. Scattered catch-and-ignore blocks would conceal accounting corruption and are not an
acceptable patch.

### CUDA/Vulkan timelines, stream retirement, and tensor ownership

The renderer's bidirectional timeline handoff was re-derived from submit sites through retirement:

- Vulkan waits for CUDA-produced model/input work before consuming it.
- CUDA waits for Vulkan completion before mutation or reuse.
- upload, export, camera, UI, and interop-owned streams call `release_stream` before destruction.
- tensor allocation records a home stream plus additional users; cross-stream free/reuse bridges
  the recorded users before recycling storage.

The lazy-runtime test defect was evidence that this contract is real: destroying a referenced
stream is not harmless test cleanup. After `7ea0020ba`, the full lazy-runtime group is clean and
the monolithic process no longer crashes at teardown.

The unified CUDA allocation/workspace owner in `core/cuda_allocation.hpp` preserves the four
prior policy combinations: pool-backed versus direct/stream-ordered allocation, existing
profiler labels, failure-injection hooks, allocation kind, and deallocation stream. No label,
hook, or free-order loss was found in `cub_workspace.hpp`, `cuda_scratch.hpp`, or the gsplat
workspace call sites.

Residual stream work is not mechanical: selection still has a default-stream fallback for
raw-pointer kernels; export needs a persistent owner above shared PPISP/band tensors; and PPISP
prediction still uses two process-wide CUDA drains. Those require owner-level timelines and
transaction boundaries.

### Rebase weaves and architecture refactors

The rebase log was checked against the resulting call graph. The skipped duplicate change is
already represented in the surviving history. The output-queue weave releases its mutex before
the blocking push, so it does not retain the producer lock across backpressure.

The just-landed architecture work was checked at its seams:

- allocation retry is owned by core memory pressure, not split between tensor callers;
- generic failure-report formatting is separate from CUDA state collection;
- checkpoint file/header parsing remains owned by core while strategy/component state remains in
  training;
- `IStrategy` no longer carries checkpoint-adoption methods, and the built-in strategies
  explicitly implement `ICheckpointStateAdopter`;
- image transitions route through the shared Vulkan layout policy without changing audited
  stage/access/layout values.

No silent behavior change was found in those refactors beyond the compatibility items called out
below.

## Regression and deletion audit

The cleanup deletion audit covered C++ call sites, exported headers, Python bindings/plugins, MCP
registrations, tests, and documentation. There is no live caller of the removed
`DeferredFreeQueue`, stale viewport depth-sampler parameter, old CUDA check helpers, or deleted
bookkeeping fields. `.codex_tmp/` is untracked; no review scratch file is in the Git tree.

The tensor-contract consolidation was sampled at former high-risk sites rather than trusted by
count. The consolidated predicates preserve device, dtype, rank, shape, contiguity, and range
checks at the audited call sites. I found no weakened predicate. There is no automated
predicate-equivalence inventory proving all 723-to-176 site migrations, so an omitted predicate
remains a Medium residual risk.

The four workspace implementations consolidated into the shared allocation owner retain their
old stream, profiler, failure-injection, and free semantics. Focused scratch-failure and
multi-stream tests pass. Thrust's internal temporary allocator is still outside this owner and is
the observed low-VRAM training cliff.

No newly added raw `assert`, conflict marker, hot-path debug print, TODO/FIXME/XXX/HACK, or AI
attribution was found. Existing TODOs were not relabeled as branch findings.

## Public API and compatibility

### Checkpoints and files on disk

The checkpoint magic and version remain v1 (`CHECKPOINT_VERSION == 1`). The branch did not bump
or reinterpret the binary header, tensor encoding, optimizer encoding, or strategy-name framing.
The stricter parser validates lengths, offsets, dtypes, shapes, capacities, and component schemas
before publication.

Compatibility was tested with two real files written before this branch:

- a 410,004,427-byte MRNF bicycle checkpoint, iteration 30,000, one million Gaussians, using the
  historical `adaptive-warped-l1` depth parameter; current code restored the payload and MRNF
  strategy state after the explicit parameter migration;
- a 416,004,262-byte MCMC garden checkpoint, iteration 7,000, one million Gaussians, predating the
  current depth fields; current code restored it and trained the remaining 23,000 iterations to
  30,000 successfully.

The first case is load compatibility, not numerical identity with the deleted historical loss:
a resumed depth-supervised job now uses SSI. That semantic migration is safer than refusing the
checkpoint, but it must be in release notes.

PLY and RAD version identifiers are unchanged. Their branch changes validate inputs and make
writes atomic; they do not introduce a new writer format.

### Intentional compatibility breaks

These changes need a migration note and downstream rebuild:

| Surface | Change | Compatibility action |
| --- | --- | --- |
| Runtime environment | `LOG_LEVEL` became `LFS_LOG_LEVEL`; bridge `LICHTFELD_*` variables became `LFS_*`; plugin registry and test-data overrides were similarly canonicalized. Many undocumented diagnostic switches were removed. | Update automation and local launch scripts. Do not silently support both forever. |
| CMake | `CUDA_DEVICE_DEBUG` became `LFS_ENABLE_CUDA_DEVICE_DEBUG`; `ENABLE_ALLOCATION_PROFILING` became `LFS_ENABLE_ALLOCATION_PROFILING`; stale tensor validation/tracing, debug-sync, and PTX-only switches were removed. | Update presets/CI. Production Release defaults remain OFF where required. |
| Public C++ source/ABI | Exported `core/cuda_debug.hpp` was removed; profiler and allocator APIs changed; `IStrategy` was narrowed. | Rebuild every downstream consumer. Custom resumable strategies must also implement `ICheckpointStateAdopter`. Treat this as an API/ABI-breaking release. |
| Python | `get_vulkan_capabilities` and plugin startup-status access are additive; no removed binding was found. | Regenerate/use the committed stubs; no migration otherwise. |
| MCP | The 136 discovered tool names remain present. The server now rejects oversized HTTP bodies, subscriptions/queues/polls, Gaussian row/value requests, and non-finite/out-of-range numbers. | Clients that relied on unbounded requests must chunk them. This is a deliberate safety tightening. |
| CLI | No user CLI option was removed. | None. |

The internal ABI stamp helps reject stale in-tree component mixing. It does not make an external
C++ consumer source- or binary-compatible.

## Build and test hygiene

### Developer options

The reviewed build is Release with `BUILD_TESTS=ON`. Its explicit cache has Vulkan validation,
shader debug information, and CUDA failure injection ON; allocation profiling and CUDA `-G` are
OFF. The declarations in `cmake/DeveloperOptions.cmake` are coherent:

- Vulkan validation and shader debug information default OFF in Release and ON in
  Debug/RelWithDebInfo;
- CUDA device debug defaults ON only in Debug;
- failure injection follows `BUILD_TESTS`, so production builds without tests compile it out;
- allocation profiling defaults OFF everywhere.

The build succeeds with `-j6`, but it is not warning-free. The observed warning set includes
missing `LoadFile` aggregate fields, third-party Zep reorder/unused-parameter warnings, and the
minizip `mktemp` linker warning. These were already present in the comparison build and are not a
new branch blocker, but “clean build” should mean exit-success here, not zero diagnostics.

### What the new tests actually prove

| Invariant | Evidence | Honest gap |
| --- | --- | --- |
| Pressure coordinator | 13 tests cover typed metadata, priority/target behavior, no-progress, reentrancy refusal, domain/context filtering, synthetic preflight, lease recovery, latch behavior, and injected tensor OOM. | Fixture reset replaces built-in clients; no test exercises real pool/pinned reclaim plus retry, non-OOM status propagation, production preflight, or a render/training safe-point client. |
| Failure reporting | 9 tests cover classifier, latch, dedup, breadcrumb wrap/thread smoke, generic-versus-CUDA sections, and caller naming. | No subprocess/death test for signal/terminate handlers; no Windows crash-path run. |
| Atomic checkpoint | POSIX fork/SIGKILL coverage kills at every atomic-commit boundary and requires a valid old or new checkpoint. | Windows replacement/durability and real disk-full behavior are untested. |
| Training callback/latch | Terminal callback cancellation, owning-trainer invalidation, queued command lifetime, parameter validation, and concurrent transition serialization are covered. | No long-running TSAN session and no teardown race under real plugin/UI load. |
| Stream ownership | 17 multi-stream allocator/transfer/arena/handshake tests plus 23 lazy-runtime tests pass together after correct retirement. | No compute-sanitizer racecheck/memcheck; external consumers can still violate the documented stream-lifetime contract. |
| Shader race | Source GPU-assisted validation reports zero shared-memory data race on the three required flows. | There is no automated shader-race regression test in CI. |

### Full-suite result

Command: `build/tests/lichtfeld_tests` from `build/`.

- Current branch: 3,125 tests from 254 suites; 3,012 passed, 50 skipped, 63 failed, one disabled;
  process exit 1, no crash.
- `origin/master`, rebuilt in the same tree: 58 failures recorded, then exit 139 during
  `TensorLazyRuntimeTest.DeferredMaterializationKeepsActualExecutionStream` before a summary.
- Comparable prefix: all 33 branch failures are also master failures; the branch passes 25 tests
  that master fails before master crashes.
- Isolation: all four failed checkpoint-resume variants pass in fresh processes; FastGS gradient
  passes alone; the 23 lazy-runtime tests pass together; two scene-removal tests fail on their
  inherited default-camera fixture; viewport/input expectations and rotated-SH asset cases remain
  reproducible in isolation.

The 63 failures include 28 stale tensor/lazy contract expectations, ten mesh fixture/loader
cases, four LOD fixture cases, four order-contaminated checkpoint variants, three known viewport
resize-deferral cases, two default-camera scene-removal cases, two rotated-SH fixture cases, and
individual environment/global-state cases. This categorization is diagnostic, not an allowlist.
The suite must be repaired or formally baselined before anyone calls the branch green.

## Source-built Vulkan and smoke gates

All GUI validation runs used:

```text
VK_LAYER_PATH=/home/paja/projects/Vulkan-ValidationLayers/build/layers
VK_LOADER_LAYERS_ENABLE=VK_LAYER_KHRONOS_validation
VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
LFS_VK_VALIDATION=1
LFS_VK_VALIDATION_FATAL=1
```

The app reported `validation_layers=active`, `debug_utils=active`, and
`validation_errors_fatal=active`. MCP bootstrap was performed for every process: initialize,
resources/list, tools/list, and the runtime/UI/scene/selection resources. Long-running training
state was confirmed through `runtime.job.wait` and retained events.

| Gate | Result |
| --- | --- |
| Build | `cmake --build build -j6`: exit 0. |
| Cross-round gtest | Exact requested filter: 291/291 passed. Historical log entries saying 292 are point-in-time counts, not the current discovered count. |
| Async plugin pytest | 10/10 passed. |
| Full gtest | Completed but red: 63/3,125 failed; details above. |
| Headless smoke | Bicycle/MCMC, 1,000 iterations, max cap 1.5M: exit 0 in 4.3 s; exactly `splat_1000.ply` plus one checkpoint; zero error/critical/fatal/exception line. |
| Single PLY | `splat_64400.ply`, 3,000,000 SH3 Gaussians; live viewport and full-window MCP captures succeeded; clean shutdown; zero VUID/error/hazard/race line. |
| Multi PLY | Three requested files, 9,000,000 Gaussians total; 3/3 loaded, MCP captures succeeded, clean shutdown; zero VUID/error/hazard/race line. |
| GUI training | Exact bicycle/MCMC 500-step non-headless flow with max cap 1.5M; `training.main` finished at iteration 500, output one PLY and one checkpoint, MCP capture succeeded, clean shutdown; zero VUID/error/hazard/race line. |

GPU-AV emitted only performance/configuration warnings about instrumentation and unsupported
optional ray/mesh validation. No validation message had error severity. Every temporary training
output directory was removed after inspection.

## Documentation, comments, and leftovers

The normative flags and assertion docs now match the code. The large Vulkan analysis is an
execution log: its historical test counts and line numbers should not be read as current API
documentation. This report records the current 291-test filter and final head.

No tracked scratch output, validation capture, generated checkpoint, PDF, `.codex_tmp` item, or
runtime log is present. The checkout has many pre-existing untracked user files; they were not
modified or deleted. `.codex_tmp/claude-vulkan` remains useful review evidence but is untracked
and must not be added to the merge.

There are no added AI-attribution trailers or messages in the 187-commit range.

## Commit-history quality

The functional commits are mostly conventional and scoped, but the integration history is not
release-quality presentation:

- Seven non-empty merge commits preserve parallel-session topology. Three are named
  `Merge branch 'slop-*'`; reword those to `cleanup-*` if history presentation matters. Do not
  flatten them blindly because they contain real integration resolutions.
- `c25142c46` (`assert hardening: ...`) is not house-style. Reword to a conventional
  `refactor(asserts): ...` subject.
- `2505699b5` (`style(vis): remove stale Vulkan narration`) mixes comment cleanup with small code
  normalization and should be squashed into its adjacent cleanup commit or reworded.
- `d8f166acd` (`refactor(docs): record final branch review`) was not final; architecture and this
  review followed it. Reword it as an intermediate review record.
- The 3K-4K-line assert/diagnostic commits are coherent mechanical migrations, but they are not
  pleasant to revert selectively. Preserve their generated/mechanical nature in the PR summary.

History rewriting is not required for correctness and was not performed. If the branch has
already been shared, prefer a clear PR narrative over force-rewriting 187 commits.

## Residual risk register

| Severity | Residual risk | Evidence / failure mode | Mitigation |
| --- | --- | --- | --- |
| High | Monolithic test suite is red and order-dependent | 63 failures on branch; master crashes after 58. Erank numerical corruption appears only after prior suites; checkpoint variants fail only in contaminated order. | Repair global CUDA/lazy resets, split asset/integration labels, provision assets, and create an owned temporary baseline until green. |
| High | Pressure resilience is partial | Only allocator-owned pool/pinned clients can reclaim. Thrust internal temp and Vulkan/VMA allocation fail closed but do not retry; production preflight is unwired. | Add owner-specific allocators and GUI safe-point/ticket clients; do not register synchronous callbacks that free timeline-live resources. |
| High | Diagnostics may throw from `noexcept` cleanup | Profiler bookkeeping can allocate during FastGS/arena teardown and terminate the process under host pressure. | Make diagnostics bounded, transactional, and non-throwing internally; test allocation failure in cleanup. |
| High | Platform proof is Linux + one NVIDIA GPU | Windows atomic replacement, crash handling, plugin child-tree cancellation, and GUI validation are untested; devices missing optional Vulkan features are untested. | Run Windows CI/runtime matrix and an AMD/feature-limited Vulkan smoke before broad release. |
| High | Public C++/configuration breakage | Removed header/options/env names and narrowed strategy interface can break downstream builds and automation. | Treat as breaking release; publish migration table; rebuild downstreams; require custom strategy adoption interface for resume. |
| Medium | Legacy depth resume changes semantics | Historical Pearson/adaptive-warped-L1 modes migrate to current SSI. | Release-note it; archive a permanent v1 compatibility fixture; compare a resumed depth job if exact scientific continuity matters. |
| Medium | Tensor-contract consolidation lacks exhaustive equivalence proof | Sampled high-risk predicates are intact, but no generated before/after predicate manifest exists. | Add a contract inventory or negative-input parameter matrix before the next large consolidation. |
| Medium | Crash handler lacks end-to-end tests | Formatting/latch tests do not exercise real signal/terminate/Windows paths. | Add subprocess death tests and verify written report/minidump contents per platform. |
| Medium | Deferred interop and UI ownership | Selection default-stream fallback, export stream ownership, PPISP global drains, preview thread bursts, overlay clipping, and Mesh2Splat lighting remain. | Address under the owning subsystem with interactive/image-parity and cancellation tests. |
| Low | History and warning debt obscures signal | `slop-*` merge names, premature “final” docs, aggregate/Zep/minizip warnings. | Reword if safe; otherwise document; fix warnings separately so new warnings become actionable. |

## Required maintainer actions

### Before merging

1. Confirm this branch contains `8ae28137e`, `77a72ba6b`, `8e4fbcc80`, and `7ea0020ba`.
2. Put the environment, CMake, C++ API, MCP-limit, and legacy-depth migration table into the PR or
   release notes.
3. Record an explicit decision on the 63-test baseline. At minimum, open owned issues for the
   order-dependent CUDA/lazy state, asset provisioning, stale contract expectations, and the
   three viewport debounce cases. Do not let a filtered 291-test gate masquerade as the full
   suite.
4. Ensure `.codex_tmp/`, local PDFs/assets, runtime logs, and generated model outputs remain
   untracked.
5. If downstream C++ plugins/strategies exist, rebuild one representative consumer before the
   release tag.

### Immediately after merging

1. Fix the monolithic test process first: deterministic per-suite global-state reset, then asset
   labels/provisioning, then stale expectations. A crash or order-dependent numerical mismatch is
   higher priority than increasing test count.
2. Route thrust/CUB scratch through an owned allocation policy and add a non-OOM propagation test
   for the centralized allocator.
3. Replace pressure preflight's callback-under-registry-lock contract before adding production
   callers; implement renderer safe-point/ticket reclamation rather than synchronous frees.
4. Make profiler cleanup non-throwing and add crash-handler subprocess tests.
5. Run Windows atomic-checkpoint/crash/plugin-cancellation gates and a feature-limited Vulkan GPU
   gate.

### Interactive-validation backlog

The three required final flows are clean, but they do not close the broader backlog. The next
attended pass should cover fixed-camera color/depth parity for normal, 3DGUT, overlays, split
view, and live training; low-to-extreme overdraw camera jumps and one-shot capture convergence;
resize/minimize/restore; repeated scene/model switch and RAD decode; forced submit/upload/OOM
faults; overlay clipping; Mesh2Splat lighting reference images; close-during-plugin-install on
Windows; and nsys proof of no hidden cumsum/sort host wait or training/UI timeline regression.

## Explicitly not covered

- Windows, AMD, integrated GPUs, multi-GPU, MIG, or device-loss recovery.
- TSAN, ASan/UBSan, compute-sanitizer memcheck/racecheck, or long soak testing.
- A real disk-full/power-loss test on every filesystem; POSIX SIGKILL boundaries are covered.
- Numerical/image parity across every renderer mode, shader feature, camera, and export format.
- Real production pressure reclaim for render/training clients or successful degrade-and-complete
  below the thrust temporary-allocation cliff.
- Exhaustive tensor-contract before/after predicate equivalence.
- External C++ consumer ABI/source compatibility.
- Exact historical depth-loss numerical continuity after checkpoint migration.
- Every deferred item in the VRAM, stability, pressure-resilience, and async-plugin design docs.

That remaining work is real. It does not invalidate the zero-error runtime proof for the audited
paths, but it is why the verdict is `MERGE-WITH-FIXES`, not `MERGE`.
