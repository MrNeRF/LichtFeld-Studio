# Windows training-start crash — investigation report (2026-07-22)

User report: on Windows, importing a COLMAP dataset works, but pressing **Start Training**
aborts with `cudaErrorIllegalAddress (700)` before the first iteration. Full log:
`message.txt` (repo root at time of writing). Reporter build: fork of this tree at
roughly `40d316fb5`.

## TL;DR

Three distinct defects were found. Two are fixed in commit `8b2268869`; the third — the
actual crash — is a tensor-library use-after-free whose exact call site could not be
proven statically. The same commit ships forensics instrumentation that will name it
from the reporter's next log (protocol below).

| # | Defect | Status |
|---|--------|--------|
| 1 | Windows Vulkan import always fails: spec-forbidden `vkGetMemoryWin32HandlePropertiesKHR` query on an opaque handle (VUID 00666), regression from PR #1428 (`c7f8484d1`) | **Fixed** in `8b2268869` |
| 2 | Exportable VMM block lifecycle: async 100 MiB zero-fill never synced before the block can be torn down; teardown ignored all `CUresult`s and never synchronized | **Fixed** (hardening) in `8b2268869` |
| 3 | The crash: stale device pointer read in the model-init corridor (`init_model_from_pointcloud`, capacity>0 path) — a deferred-materialization / pool use-after-free | **Open** — runtime pin pending |

## Event chain (from the log)

1. `TrainerManager::createTrainingSplatTensorAllocator` allocates a 100 MiB CUDA VMM
   exportable block (`device_ptr=0x1306000000`) for zero-copy viewer interop.
2. Vulkan import fails instantly: `vkGetMemoryWin32HandlePropertiesKHR →
   VK_ERROR_INITIALIZATION_FAILED` (defect 1). The fallback drops the block and
   switches to the legacy Vulkan-external allocator.
3. ~400 ms later, still inside the Start-Training click handler, model init from the
   point cloud dies: `cudaErrorIllegalAddress` first observed at
   `cudaStreamSynchronize(transfer_stream)` in `Tensor::to` (tensor.cpp:1354), reached
   via a `Tensor::cpu()` call. Three cascading failure reports follow (sticky context
   error during exception unwind — noise, not additional bugs).

## Defect 1 — Windows interop regression (CONFIRMED)

`c7f8484d1` (PR #1428, 2026-07-22) added a `vkGetMemoryWin32HandlePropertiesKHR` query
with `VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT` to the Windows import path
(vulkan_context.cpp). The Vulkan spec forbids this query for opaque handle types
(VUID-vkGetMemoryWin32HandlePropertiesKHR-handleType-00666 — the exact Win32 analogue
of the fd-variant VUID 00674 that the Linux branch already cites and avoids). The
driver rejects the call, so **every** Windows training start failed interop and took
the fallback corridor — which had never been exercised on any platform (Linux takes
the interop path; headless has no allocator).

Fix: query deleted; memory-type selection uses the buffer requirements, matching the
Linux branch and NVIDIA's `simpleVulkanMMAP` sample (which is also what the code did
before `c7f8484d1`).

Related, deliberately NOT changed: `CUmemAllocationProp::win32HandleMetaData` receives
a `SECURITY_ATTRIBUTES`, while the CUDA 12.8 header documents `POBJECT_ATTRIBUTES`.
NVIDIA's own `memMapIPCDrv` sample uses `SECURITY_ATTRIBUTES`, and the reporter's log
proves the current form exports successfully on real drivers. Flipping it blind on a
platform we cannot compile is regression risk with no demonstrated benefit; a comment
in `exportable_storage.cpp` records the contradiction.

## Defect 2 — VMM lifecycle hardening (CONFIRMED defect, not this crash)

`commit_physical` zeroed the freshly mapped block with an asynchronous `cudaMemset`
and never synchronized; teardown (`CloseHandle → cuMemUnmap → cuMemRelease →
cuMemAddressFree`) ignored every result and never synchronized either, while the grow
path correctly did. Unmapping a VMM range under in-flight work is undefined.

This was initially the leading crash theory, but it is **refuted for this log**: the
breadcrumb window proves several successful default-stream synchronizations between
the teardown and the failure, which is impossible if the teardown had faulted the
context. (The confusion arose because the memset and all VMM driver calls were
breadcrumb-invisible — fixed by the instrumentation below.)

Fix anyway (real defect class): checked `cudaStreamSynchronize` after the zero-fill
with rollback on failure; device sync before teardown; all cleanup `CUresult`s logged.

## Defect 3 — the crash (OPEN, mechanism identified, call site pending)

Evidence chain from the breadcrumbs (`message.txt` lines 100–165):

- Every CUDA call up to crumb #2679 succeeded; the fault is created strictly between
  the last clean sync (#2672) and the failing sync (#2680). The only device-touching
  enqueues in that window are two pool frees (`cudaFreeAsync`) and the failing
  device-to-host copy itself. **Ergo: the D2H read's source pointer was invalid.**
- The crumb fingerprint places the window inside `init_model_from_pointcloud`
  (splat_data.cpp), capacity>0 path: the row-proxy pairs are the median
  `sorted_dists.first[n/2].item()` (:1316), the successful D2H is `positions.cpu()`
  (:1405), and the failing D2H is `colors.cpu()` (:1467) or the lazily-materialized
  `to(Float32).div(255).cuda()` colors chain (:1301) — the clone/unary crumbs directly
  before the failure match its deferred materialization.
- Exonerated: the exportable block (nothing ever pointed into it — the interop
  allocator fails before any tensor is built on it), the legacy Vulkan→CUDA import
  (audited correct: sizes, offsets, dedicated-allocation mirroring, handle lifetime),
  the strided kernel and sort work (bracketed by successful syncs).

Prime suspect class: **`ptr()`/`data_ptr()` on a deferred tensor replaces the tensor's
storage during materialization** (tensor.cpp:509–523). A raw device pointer captured
before a later materialization of the same graph is stale, and because the pool's
bucket tiers really do `cudaFreeAsync` to the driver, a stale pointer is a hard
illegal-address on Windows rather than benign garbage. Exactly one call site defends
against this (tensor_expr_impl.hpp:347–353 pre-materializes, with a comment saying
why); every other raw-pointer capture is unprotected. Static audit could not prove
which capture site fired in this log.

Adjacent defects found during the audit (backlog, `[[project_tensorlib_hardening]]`):

- **D1** — the tensor copy constructor deep-copies deferred `TensorState` while
  sharing the graph node id; the lazy registry tracks one state per id → copies
  desync, double materialization, cache identity confusion (tensor.cpp:603–670).
- **D2** — `cached_materializations.emplace` never overwrote stale entries
  (lazy_executor.cpp:644). **Fixed** in `8b2268869` (`insert_or_assign` + a
  plan-context gtest).
- **D4** — `from_blob` / non-owning tensors are invisible to `record_stream` and the
  free machinery; same fault signature available elsewhere.

The `ptr()` semantics question needs a design pass (non-mutating ptr vs stable
addresses vs defensive pre-materialization vs type-level split) — deferred until the
runtime pin tells us which pattern actually fires.

## Closing the identification — protocol for the reporter

Commit `8b2268869` instruments exactly what was invisible: breadcrumbs now carry
stream + `(dst, src, bytes)` on every checked transfer, transfer failures name the
tensor being copied, deferred-materialization storage moves are recorded, and an
address-range **necrology** annotates faulting pointers against live and recently
freed ranges directly in the failure report.

Two runs on a build containing `8b2268869`:

1. **Normal run.** The failure report then decides mechanically:
   - the failing D2H's byte count names the tensor: 714,828 B → positions/colors
     (`[59569,3]` f32); 238,276 B → dists; 4 B → the median item read;
   - if the source pointer is annotated as inside a range **freed before the copy**
     → UAF confirmed, and the freeing crumb names the temporary whose owner died;
   - if the source is inside a **live** range → check for overrun vs in-bounds; the
     in-bounds case means a crumb-invisible op poisoned the context.
2. Only if run 1 is inconclusive: `LFS_CUDA_SYNC_DEBUG=cuda-sync` (syncs after every
   op; converts the async fault into an exact file:line).

## Non-blocking follow-ups (review round 2)

- CHECK breadcrumbs publish post-call: a hung op leaves no crumb for itself.
- Exportable-block VA registered by two owners on the successful-interop path; the
  Vulkan wrapper's unregister creates a brief false-DEAD window.
- Grow path still doesn't check `cuMemSetAccess`/`cudaMemset`/`cudaMemcpy` results.
- Second pinned intentional-leak path lacks the "LIVE ≡ still mapped" comment.

## UPDATE 2026-07-23 — instrumented-build log received: original crash NOT reproduced; new defect D5 found

The reporter ran the instrumented build (`8b2268869`). Result:

1. **The original training-start crash did not reproduce.** COLMAP import + Start Training
   worked; training ran to iteration 2997 and stopped only because the user pressed Stop.
   The interop regression fix (defect 1) plus the D2 lazy-cache `insert_or_assign` fix are
   the plausible killers of the original corridor. Status: original crash CLOSED pending a
   longer confirmation run; the D1/D3 hardening campaign continues independently (the
   defect classes are real regardless of which one fired in the July 22 log).

2. **New defect D5 (proven end-to-end from this log): thread-exit CUDA teardown latches
   the whole app dead.** Chain:
   - Training stop → the trainer worker thread exits → `thread_local core::Tensor`
     rasterizer caches destruct during `LdrShutdownThread` (Windows loader lock):
     `gsplat_rasterizer.cpp:106,383-385`, `fast_rasterizer.cpp:315-318,543`,
     `gsplat/Intersect.cpp:117`.
   - Pool deallocate → `SizeBucketedPool::enforce_cache_budget` →
     `cudaFreeAsync` → `cudaErrorInitializationError (3)` (CUDA calls are unsupported in
     DLL thread-detach context).
   - `is_cuda_unavailable_error(3)` == true → `latch_cuda_unavailable`
     (cuda_error.cpp:660) flips the one-way process-global kill switch.
   - `CudaMemoryPool::allocate` checks the latch (memory_pool.hpp:121) → every later
     allocation fails — observed as `Failed to read COLMAP...: cuda-device out of memory:
     failed to allocate 0.0 MiB (native 3)` on dataset reload, plus Python-side
     `ResourceError: Out of memory`. The driver itself was healthy (VRAM query in the
     failure report: 8122 MiB free).
   - Fix direction (folded into hardening Round 2): (a) explicit thread-cache release at
     the end of the trainer thread body instead of TLS-dtor teardown; (b) free-path CUDA
     failures never feed the unavailable latch (log + leak instead); (c) optional
     per-thread teardown sentinel so allocator frees become no-op-leak on dying threads.
