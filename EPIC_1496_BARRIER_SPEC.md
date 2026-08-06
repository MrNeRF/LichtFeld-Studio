# Epic #1496: Computed Vulkan barriers — frozen implementation spec (v2)

> STATUS 2026-08-06: implemented on this branch. §0's "82 hand-written calls" describes the
> pre-migration state; `gs_renderer.cpp` now has zero — every dispatch declares tagged accesses
> and the planner computes barriers. Kept as the design record; dispositions of post-migration
> sweep findings live in `debug/epic1496/findings.md`.

Branch: `epic-1496-computed-barriers`. Scope: `src/rendering/rasterizer/vulkan/src/` (buffer
barriers), `src/visualizer/window/vulkan_image_barrier_tracker.*` + `vulkan_context.*` (image
tracker + leak census, issues #1478/#1488, epic sub-task 4).

v2 integrates the adversarial critique (`debug/epic1496/critique.md`): shared-scratch parent
tracking (was blocker B3/C3), redefined audit gate (was F1-F3), batch-begin state reset (was
A5/H4), handoff-access rule (was H7), plus all majors. Recon inputs:
`debug/epic1496/recon_a_barrier_catalog.md`, `debug/epic1496/recon_b_image_tracker.md`.

## 0. Problem

`gs_renderer.cpp` has 82 hand-written `bufferMemoryBarrier` calls; each dispatch's buffer list is
typed three times with conservatively guessed src masks (`TRANSFER_COMPUTE_SHADER_WRITE` = "might
have been copy, might have been compute, wait for both"). The catalog found 36 sites where the
barrier list and the dispatch's actual bindings disagree. Divergence is a silent race in release
builds; over-wide masks serialize. The dependency info already exists at the call site — compute
the barrier.

**Non-goals**: no command reordering (chain is strictly sequential; phase order is a blending
constraint); no render-graph; no changes to submission/batch lifecycle semantics or shader
bindings; no CUDA-side sync changes.

## 1. Deliverables in dependency order

1. **P1 — image tracker rework** (#1478, #1488, sub-task 4). Independent of P2/P3.
2. **P2 — `BufferBarrierPlanner`**: pure host-side hazard planner, GPU-free unit-testable.
3. **P3 — tagged dispatch path** + mixed-mode safety + first migrated chain (LOD select).
4. **P4 — chain-by-chain migration**; `bufferMemoryBarrier()` stays callable throughout.

## 2. P2: BufferBarrierPlanner (new `barrier_planner.{h,cpp}` in rasterizer vulkan lib)

### 2.1 Usage tags

```cpp
enum class BufferUse : uint8_t {
    ComputeRead, ComputeWrite, ComputeReadWrite,
    TransferRead, TransferWrite,
    IndirectRead,        // DRAW_INDIRECT stage, INDIRECT_COMMAND_READ access
    HostRead,            // host readback (barrier only; fence/timeline wait is separate, see §5 G3)
    ConditionalRead,     // conditional-rendering predicate consumption
};
```

Each maps to one `{VkPipelineStageFlags2, VkAccessFlags2}` scope, matching the single-use rows of
`toStageMask`/`toAccessMask` (gs_pipeline.cpp:1574-1644). Tests pin equivalence by CALLING those
functions, not re-hardcoding masks.

### 2.2 Tracked-buffer registry

```cpp
struct BufferState {
    Scope writer;                          // last unsynchronized writer ({NONE,NONE} if none)
    VkPipelineStageFlags2 reader_stages;   // readers since last write (WAR sources)
    VkPipelineStageFlags2 visible_stages;  // scopes the writer's data is already visible to
    VkAccessFlags2 visible_access;
};
std::unordered_map<VkBuffer, BufferState> states_;
```

- Keyed on bare `VkBuffer`, WHOLE-buffer granularity; emitted barriers use
  `offset=0, size=VK_WHOLE_SIZE`, `srcQueueFamilyIndex = dstQueueFamilyIndex =
  queue_family_index` (matches legacy emission shape).
- **Tracked buffers** are:
  1. Owned allocations from `createBuffer`/`resizeDeviceBuffer` (always `offset=0`, dedicated
     VkBuffer — verified buffer.cpp:242,257,335). Registered at create, forgotten in
     `destroyBuffer`.
  2. **Adopted external parents**: `VulkanGSPipeline` gains
     `void trackExternalParent(VkBuffer)` / `void untrackExternalParent(VkBuffer)`. The
     visualizer calls these for the shared-scratch imported buffer when it binds/rebinds/grows it
     (`bindSharedScratchBuffers`, vksplat_viewport_renderer.cpp:~3439-3499; scratch
     generation bumps at 3278/3341/3380 ⇒ untrack+track around re-import). ALL region views
     (`makeRegionView`/`makeResizableRegionView`) of that parent then collapse to the parent's
     whole-buffer state. Rationale: production rebinds most hot chain buffers to views of this
     one parent; without adoption the planner would stay conservative on the main path
     (critique B3). Whole-buffer collapse is safe: execution dependencies are stage-global in
     Vulkan; the memory scope is a superset. Cross-API (CUDA) writes to the scratch are ordered
     by timeline-semaphore edges exactly as today; chained barriers preserve visibility for
     stages outside the wait scope (critique H9).
  3. Everything else (un-adopted external views, readback buffers destroyed via raw
     `vmaDestroyBuffer`) stays UNTRACKED ⇒ conservative on every access.
- Registered-but-never-accessed state is the EMPTY state (`writer=NONE, readers=NONE,
  visible=NONE`): a first write emits no barrier (nothing outstanding — critique A6); a first
  read emits none either (no hazard; reading garbage is a logic bug, not a sync bug).

### 2.3 plan() semantics

```cpp
struct DeclaredAccess { const _VulkanBuffer* buffer; BufferUse use; };
std::vector<VkBufferMemoryBarrier2> plan(std::span<const DeclaredAccess> accesses);
```

**Simultaneity rule (frozen)**: all accesses in one `plan()` call are first MERGED per VkBuffer
(reads OR together; read+write or any write+read combination merges to the ReadWrite form of the
widest stage set), then hazard-checked ONCE against the pre-plan state; state updates apply only
after the full barrier set is determined. One dispatch never barriers against itself.
`VK_NULL_HANDLE` buffers are skipped (legacy parity).

Hazard rules per merged access `A` on tracked buffer `B` (untracked ⇒ conservative row):

| prior state | A | barrier? | src | dst | state after |
|---|---|---|---|---|---|
| UNTRACKED buffer | any | YES | stage=ALL_TRANSFER\|COMPUTE, access=TRANSFER_WRITE\|SHADER_WRITE (== toStage/AccessMask(TRANSFER_COMPUTE_SHADER_WRITE)) | A.scope | none kept |
| writer=W, A.scope ⊆ visible | read | NO | — | — | reader_stages \|= A.stage |
| writer=W, A.scope ⊄ visible | read | YES | W | A.scope | visible \|= A.scope; reader_stages \|= A.stage |
| writer=NONE | read | NO | — | — | reader_stages \|= A.stage |
| writer=W | write / read-write | YES (WAW+RAW+WAR merged) | stage = W.stage \| reader_stages, access = W.access | A.scope | writer = A.scope (full R/W access bits as declared); readers=NONE; visible=NONE |
| writer=NONE, readers ≠ NONE | write / read-write | YES (WAR, execution-only) | stage = reader_stages, access = NONE | A.scope | same as above |
| writer=NONE, readers=NONE | write / read-write | NO | — | — | writer=A.scope; readers=NONE; visible=NONE |

"⊆ visible" ⇔ `(A.stage & ~visible_stages)==0 && (A.access & ~visible_access)==0`.

### 2.4 Batch-boundary rule (frozen; critique A5/H4)

`beginCommandBatch` already records a global reuse barrier covering
`ALL_TRANSFER|COMPUTE|DRAW_INDIRECT` with full R/W access (gs_pipeline.cpp:814-849) because up to
`kCommandBatchSlotCount=3` batches overlap on one queue; that barrier does NOT cover
CONDITIONAL_RENDERING or HOST. Planner hook on every `beginCommandBatch` (including HostGuard
restarts): every tracked entry is reset to

```
writer  = {stage: ALL_TRANSFER|COMPUTE, access: TRANSFER_WRITE|SHADER_WRITE}   // conservative
visible = the reuse barrier's dst scope (TRANSFER|COMPUTE|DRAW_INDIRECT stages + their R/W accesses)
readers = NONE
```

Consequence: compute/transfer/indirect reads in the new batch elide (the reuse barrier synced
them); ConditionalRead and HostRead do NOT match `visible` and emit a chained barrier from the
conservative writer — closing the under-sync window the critique found. This same reset also
covers all extra-batch mutators (visualizer compose raw barriers on pixel buffers
vksplat_viewport_renderer.cpp:5288-5307,5796-5811, CUDA interop, other queues): no `visible`
state survives a batch boundary beyond what the reuse barrier guarantees. The reuse barrier's
masks are NOT changed by this epic.

### 2.5 API surface

```cpp
void track(VkBuffer);        // createBuffer + trackExternalParent
void forget(VkBuffer);       // destroyBuffer + untrackExternalParent
void invalidate(VkBuffer);   // legacy-path hook: entry -> UNTRACKED-equivalent conservative mark
void onBatchBegin();         // §2.4 reset
void reset();
std::vector<VkBufferMemoryBarrier2> plan(std::span<const DeclaredAccess>);
struct Stats { uint64_t barriers_emitted, accesses_elided, conservative_fallbacks; };
Stats stats() const;         // resettable snapshot for per-chain audits + frame telemetry
```

`invalidate` marks the entry so the next access takes the conservative row (implementation may
erase to a tombstone that `track` state re-creates; behavior, not representation, is frozen).

### 2.6 Audit gate (replaces the epic's literal "derived ≤ hand-written on any chain"; critique F)

The epic's runtime assert is ill-defined for data-dependent wave counts and penalizes legitimate
de-hoisting. Frozen replacement, honoring the epic's intent (derivation must never be more
barriers than the hand-written code FOR THE SAME work):

1. **Unit-test gate (GPU-free, scripted dispatch)**: for each migrated chain, record it with
   forged handles under FROZEN branch configurations (explicit wave count W ∈ {1,3}, gut/light/
   batched/overlay flags fixed per test) and assert
   `derived VkBufferMemoryBarrier2 struct count ≤ hand-written baseline constant` where the
   baseline is computed from the catalog for that exact configuration and documented as a named
   constant in the test (e.g. `kAuditSelectLodThresholdTop = 19` [12+5+2],
   `kAuditSelectLodThresholdWithReadback = 24` [+4+1], `kAuditMapLodIndices = 4` [3+1]).
   Baselines for wave chains use the catalog formulas (critique F2) evaluated at the frozen W.
2. **Edge-coverage check (same tests)**: every hand-written hazard edge (producer scope →
   consumer scope per buffer, from the catalog) must be covered by some derived barrier whose
   src/dst scopes are supersets. This catches under-sync that a pure count can't.
3. **Runtime (debug builds)**: assert each tagged dispatch emits ≤ 1 `vkCmdPipelineBarrier2`
   call (coalescing invariant); expose `Stats` per frame for the nsys A/B and for a
   conservative-fallback trend check. No data-dependent count asserts at runtime.

## 3. P3: tagged dispatch path

### 3.1 New overloads (gs_pipeline.h)

```cpp
struct TaggedBinding { _VulkanBuffer buffer; BufferUse use; };
void executeCompute(dims, uniformsPtr, uniformSize, pipeline, const std::vector<TaggedBinding>&);
void executeComputeIndirect(indirect_buffer, indirect_offset, uniformsPtr, uniformSize, pipeline,
                            const std::vector<TaggedBinding>&);
```

Tagged path: `plan()` over bindings (indirect variant adds an implicit
`{indirect_buffer, IndirectRead}`), emit ONE `vkCmdPipelineBarrier2` if non-empty, then
bind/push/dispatch via a shared private impl. The untagged overloads become thin wrappers:
`invalidate()` every buffer, then shared impl with no planning (unchanged behavior: caller must
barrier). No trailing barrier after tagged dispatches — next consumer plans its own, EXCEPT
handoff accesses (§3.4).

### 3.2 Transfer/fill/host recording

`clearDeviceBuffer` (fill), `resizeAndCopyDeviceBuffer` (copy + tail fills), and the readback
copy/`vkCmdFillBuffer` sites inside MIGRATED chains record planner accesses
(`TransferRead`/`TransferWrite`/`HostRead`) instead of hand-written barriers, via a small
`planTransfer(...)` helper on the pipeline. These helper hooks ship IN THE SAME COMMIT as the
first migrated chain (critique B1). `HostRead` emits the barrier only; host coherence still
requires the existing fence/timeline wait (`endCommandBatch`) — documented, tested (G3).

### 3.3 Emission through the dispatch seam

Route legacy `bufferMemoryBarrier` (today calls `vkCmdPipelineBarrier2` directly,
gs_pipeline.cpp:1693,1742) AND planner emission through `vulkan_dispatch_.cmd_pipeline_barrier2`
(constructor already installs `VulkanDispatch::real()`; `beginCommandBatch` already requires the
PFN non-null). Extend `VulkanDispatch` with `cmd_bind_pipeline`, `cmd_push_constants`,
`cmd_dispatch`, `cmd_dispatch_indirect`, `cmd_fill_buffer`, `cmd_copy_buffer`;
`VulkanDispatch::real()` fills them; tagged path asserts non-null like beginCommandBatch;
existing scripted tests keep working (they never call the new paths); new chain tests mock them.
`vk_cmd_push_descriptor_set_` is already an injectable member — scripted tests install a no-op.

### 3.4 Mixed-mode safety rules (frozen invariants)

1. Legacy `bufferMemoryBarrier` (both overloads): `invalidate()` every named buffer.
2. Untagged `executeCompute`/`executeComputeIndirect`: `invalidate()` every bound buffer.
3. `destroyBuffer`/`untrackExternalParent`: `forget()` — handle reuse cannot inherit state.
4. **No hand-written barrier is deleted until every write to its buffers within the chain
   (dispatch, fill, copy) is planner-recorded** (critique B1).
5. **Handoff rule (critique H7)**: before deleting a hand-written POST-barrier, enumerate its
   dst consumers. If a consumer is not itself a planning call site — conditional-rendering
   predicate reads (`ConditionalRenderingScope`, gs_renderer.cpp:40-60; hand-written unlock at
   3404-3412) are the known case — the migrated chain declares an explicit planned handoff
   access (`{predicates, ConditionalRead}` at scope begin), or the hand-written post-barrier
   stays until the consumer migrates.
6. Timeline-semaphore waits need no planner action (visibility via semaphore edges + chained
   barriers, as today).
7. Visualizer-side raw barriers never touch the planner; the §2.4 batch reset covers them.

### 3.5 First migrated chain

`executeMapLodIndices` (gs_renderer.cpp:1004-1042) + `executeSelectLodThreshold` (1044-1148)
including its `recordLodSelectionReadback` helper and `clearDeviceBuffer` fills. Usage tags come
from the Slang shaders (`lod_map_indices.slang`, `lod_select_threshold.slang`,
`lod_compact_touch.slang`) — catalog has the per-binding read/write annotations. Note for
reviewers: the LOD input tables (node_bounds, node_links, chunk_to_page, page_*) are external
region views and stay conservative unless their parents are adopted; exactness lands on the
owned/adopted outputs (critique H6). Off the hottest path; most conservative barrier in the file.

## 4. P1: image tracker (#1478, #1488, sub-task 4)

Recon facts: shared tracker (`VulkanContext::image_barriers_`) has 22 register/forget expressions
in 4 files + 3 local tracker instances with 12 more (issue's "14/5" is stale); `reset()` has no
production callers. Only one live image can hold a handle value at a time ⇒ map stays keyed on
`VkImage`, generation stored in the entry; mismatch = stale-key miss.

### 4.1 Generation-keyed tracker (#1478)

```cpp
struct Entry { uint64_t generation; ImageState state; bool external; };
std::unordered_map<VkImage, Entry> images_;
```

- `registerImage(image, generation, aspect, layout, external)` overwrites the entry and STORES
  the generation. `forgetImage(image, generation)` erases only on generation match (a stale
  forget from an old owner must not evict the new image — unit-tested). `transitionImage(...,
  generation, ...)` / `imageLayout(image, generation, fallback)`: generation mismatch ⇒ treat as
  untracked (`LFS_VK_DEBUG_ASSERT` in debug; conservative `layoutAccess` default / fallback
  layout in release). ALL generation parameters non-defaulted (compiler finds every site).
- **Generation discipline (critique D2/D3)**: `registerImage` and the owner's field update use
  the same value in the same function; `forgetImage` uses the generation captured from the owner
  field BEFORE destroy/clear; restore-on-failure re-registers (point_cloud:~1797, mesh:~378)
  pass the SAME generation.
- **Generation sources** (create where missing, same commit):
  - vksplat `OutputImageSlot`: `slot.generation` is dual-use (recreate bump :4976 vs compose
    content overwrite :5229/:5368) — add dedicated `image_generation` bumped only in
    `ensureOutputImages`; compose keeps `generation`. ALL tracker calls for slot images
    (register/forget/transitions/imageLayout probe ~5163, readback restores ~5937/6094/6264,
    compose scope transitions 5167/5180) use `image_generation` (critique D1; test G9).
  - point-cloud `OutputSlotResources`: same dual-bump trap (:1118 vs :1904) — same split.
  - `VulkanContext`: new `swapchain_epoch_` bumped in `createSwapchain`; used by swapchain +
    depth registrations and beginFrame/endFrame/capture transitions.
  - mesh offscreen `Impl`: new counter bumped in `ensureTargets`.
  - Local trackers (UI texture, scene uploader, RmlUi): member counters; ephemeral stack
    trackers may use constant generation 1 (critique D6).
- `clearSwapchainOnly()` semantics unchanged (drops non-external entries).

### 4.2 Reader-scope accumulation (sub-task 4)

`ImageState` gains `reader_stages`/`reader_access`.

- Layout-derived overload: transitions to read-only dst scopes accumulate into `reader_*`
  WITHOUT clearing writer info; a subsequent transition with write semantics uses
  `src = {last_stage|reader_stages, last_access|reader_access}` then clears `reader_*`.
- Same-layout early return STILL accumulates the destination scope (explicit-scope overload: the
  passed destination; layout overload: `layoutAccess(layout, Destination)`) — closes the sharp
  edge at vulkan_image_barrier_tracker.hpp:49-51 where a second same-layout reader vanishes.
- Explicit-scope overload (cross-queue semaphore edges): caller scopes are used VERBATIM as
  src/dst (never OR readers into a caller-provided src — the caller owns cross-queue reasoning,
  critique D8); destination still accumulates into `reader_*`/state for later layout-derived use.
- Testability: tracker gains an injectable `PFN_vkCmdPipelineBarrier2` member (default = real
  symbol) so tests capture emitted `VkImageMemoryBarrier2` with forged handles.

### 4.3 Leak census (#1488)

Pure `GpuObjectCensus` (new `src/visualizer/window/gpu_object_census.{hpp,cpp}`, `LFS_VIS_API`):
`onCreate(kind, scope)` / `onDestroy(kind, scope)` / `report()`; destroy-without-create clamps
at zero and flags. Hooks live ONLY in `VulkanContext`: `createExternalImage`/
`destroyExternalImage`, `createExternalBuffer`/`importExternalBuffer`/`destroyExternalBuffer`,
`createExternalTimelineSemaphore`/`destroyExternalSemaphore`. `ExternalSemaphore` gains
`diagnostic_scope` + a scope parameter on `createExternalTimelineSemaphore` (callers:
viewport_interop_service.cpp:422, vulkan_ui_texture.cpp:667, vksplat_viewport_renderer.cpp:1715,
4227, + whatever the build finds). Report in `shutdown()` (vulkan_context.cpp:586+) after
`vkDeviceWaitIdle`, before `vkDestroyDevice`: one `LOG_WARN` per surviving `{kind, scope}`;
zero survivors ⇒ one `LOG_DEBUG`. Makes ordering bugs visible; does not fix ordering. VRAM byte
gauges are not object counts.

## 5. Test plan (TDD: tests written and reviewed BEFORE implementation)

Files pre-registered in tests/CMakeLists.txt. House checkups apply (fresh-binary proof,
filter-matched-something, claimed==present, discriminating-power comment per test).

1. `tests/test_vksplat_barrier_planner.cpp` — every §2.3 row incl.: conservative == legacy
   TRANSFER_COMPUTE_SHADER_WRITE (via toStageMask/toAccessMask calls); RAW/WAW exact; WAR after
   N readers unions stages, src access NONE; read-read elision + new-stage read re-barriers;
   access-superset read after narrower visibility re-barriers (A2 sharp edge); ReadWrite merge;
   duplicate-binding merge; first-write-after-track no barrier; fill→compute-read src is
   TRANSFER-only; forget+handle-reuse conservative; invalidate conservative; track/untrack of an
   external parent + two region views collapsing to parent state; onBatchBegin reset semantics —
   compute read elides, ConditionalRead/HostRead do NOT elide (G2/G6); indirect-arg
   ComputeWrite→IndirectRead barrier (G1); cumsum-style multi-phase ReadWrite ping-pong (G4);
   HostRead barrier shape + "no coherence without fence" note (G3); struct shape snapshot
   (queue family, WHOLE_SIZE) (G10); stats counters.
2. `tests/test_vksplat_tagged_dispatch.cpp` — scripted-dispatch (TestablePipeline pattern):
   tagged executeCompute emits exactly one barrier2 call before dispatch; untagged invalidates
   (G8 mixed chain: tagged → legacy barrier → tagged = conservative third); indirect implicit
   access; handoff ConditionalRead declaration; batch-boundary reset via begin/end cycles (G6).
   After P3: record migrated LOD chain with forged handles; assert struct count ≤ named baseline
   constants (§2.6.1) and edge coverage (§2.6.2).
3. `tests/test_vulkan_image_tracker.cpp` — §4.1/4.2: re-register new generation = no
   inheritance; stale forget no-op (register gen2 after missed forget gen1 → state preserved);
   mismatched transition = untracked/conservative; reader accumulation across same-layout
   early-return then write-transition union; explicit-scope verbatim src (empty external src not
   polluted by readers); clearSwapchainOnly keeps external; compose content-generation must not
   be used for tracker calls (G9).
4. `tests/test_vulkan_gpu_census.cpp` — balanced ⇒ empty; survivors ⇒ {kind, scope, count};
   scope isolation; underflow clamp+flag.

## 6. Migration order (P4)

One commit per chain; sync-validation between. Wave-loop mega-hoists (`executeLegacyDepthWaves`
L1387: 24 entries; `executeMacroDepthWaves` L2988: 18) disappear as per-wave dispatches plan
exactly; their audit baselines are branch-frozen unit tests only (§2.6). `executeCumsum`'s
`additional_begin_barriers` callers pass tagged accesses once migrated; parameter removed with
the last caller. Single-owner readback helpers migrate with their owning chains.

1. LOD: `executeMapLodIndices`, `executeSelectLodThreshold` (+`recordLodSelectionReadback`) — P3
2. `executeSelectionMask`, `executeSelectionPolygonRasterize`
3. `executeProjectionForward` (+ sentinel fill 1191-1197, `recordVisibleCountReadback`)
4. `executeCullSplats`, `executeProjectionForwardSurvivors`
5. `executeCumsum`, `executeCalculateIndexBufferOffset[Visible]`, `executePrepareTileSort`
6. `executeSortPrimitivesByDepth[Visible]`, `executeSortIndirectCount*`
7. `executeApplyDepthOrdering`, `executeMacroCoverage`, `executeWavePartition`
   (+ conditional-predicate handoff, `recordInstanceCountReadback`)
8. `executeMacroDepthWaves`, `executeLegacyDepthWaves` (hottest, LAST)
9. `synchronizeTileInstanceGate` + remaining readback helpers

Per chain: (a) tag from shader-derived usage (catalog), (b) delete hand-written barriers under
rules §3.4.4-5, (c) frozen-config audit test, (d) build + targeted gtest filters, (e)
sync-validation GUI run with PLY load — zero validation noise.

## 7. Gates (epic acceptance)

- Vulkan synchronization validation clean: `scripts/run_vulkan_validation.sh` (sync + GPU-assisted
  already wired), GUI + PLY load + orbit + resize + GT compare, zero errors.
- Audit: §2.6 unit-test gates green; runtime coalescing assert active in debug runs.
- `nsys profile --trace=vulkan,nvtx,cuda` on the 7k smoke run: rasterize/compose NVTX ranges vs
  master — gate is "not worse"; win may be ~0 (measure, don't assume).
- Build: `~/.local/bin/lfs-build-guard -j 3` ONLY (user order: max -j3; other project has
  priority). Builds are centrally coordinated — implementation agents do not build unless their
  packet says so.
- Targeted gtest filters only; suites findable via `ctest -N -R`.
