# Epic #1496 Barrier Spec — Adversarial Critique

**Subject:** `EPIC_1496_BARRIER_SPEC.md`  
**Method:** claim-by-claim verification against current tree (gs_pipeline, buffer, gs_renderer, image tracker, vulkan_wait, vksplat).  
**Read-only.** No repo files modified.

---

## A. Hazard table (spec §2.3) vs Vulkan 1.3 sync2 + batch lifecycle

### A1. WAR with `srcAccess = NONE` — OK

- **Severity:** NIT (positive verification)
- **Evidence:** Spec §2.3 row “writer=NONE, readers≠NONE → write → access=NONE”. Matches sync2: WAR needs an execution dependency only; memory availability is not required when the prior accesses were pure reads.
- **Fix:** Keep as written; unit-test with multi-stage reader unions (including `DRAW_INDIRECT` and `CONDITIONAL_RENDERING`).

### A2. Visibility-set elision of repeat reads — OK with one sharp edge

- **Severity:** MINOR
- **Evidence:** Spec notes “A ⊆ visible” and post-read `visible |= A`. Correct for pure reads after the same writer. Elision must **not** apply when `A.access` is a strict superset (e.g. first `ComputeRead` then `ComputeReadWrite`): the bitwise test `(A.access & ~visible_access)==0` handles that.
- **Sharp edge:** Spec never states whether multiple `DeclaredAccess` entries in one `plan()` call are hazard-checked **simultaneously** against the pre-plan state, or applied sequentially. Simultaneous is the only safe model for one dispatch. Sequential update would invent false RAW/WAW inside a single command.
- **Fix:** Freeze: “All accesses in one `plan()` are merged per `VkBuffer`, then hazard-checked once against **pre-plan** state; state is updated only after the barrier set is fully determined.”

### A3. Read-write row — mostly OK; merge rule is slightly lossy

- **Severity:** MINOR
- **Evidence:** Spec: “any write ⇒ write” when merging duplicates; read-write row “apply write row”. For hazards against **prior** state, treating R+W as write is fine (same dispatch needs no self-barrier). After-state `writer = Write-only` (SHADER_WRITE) vs `ReadWrite` (SHADER_READ|SHADER_WRITE) only matters if a later consumer inspects writer access for something other than the next hazard row — currently the next write uses `W.access` as src access, which is correct for WAW.
- **Fix:** Prefer merge to `ComputeReadWrite` when both read and write appear, so after-state matches the actual last access mask; document either choice.

### A4. Conservative-unknown src vs `TRANSFER_COMPUTE_SHADER_WRITE` — OK

- **Severity:** NIT (positive)
- **Evidence:** Spec claims exact match to today’s expansion. Code:
  - `toStageMask(TRANSFER_COMPUTE_SHADER_WRITE)` → `ALL_TRANSFER | COMPUTE_SHADER` (`gs_pipeline.cpp:1616–1632`)
  - `toAccessMask(TRANSFER_COMPUTE_SHADER_WRITE)` → `TRANSFER_WRITE | SHADER_WRITE` (`gs_pipeline.cpp:1582–1598`)
  - Spec §2.3: `stage=TRANSFER|COMPUTE, access=TRANSFER_WRITE|SHADER_WRITE` — matches.
- **Fix:** Test must call the same `toStageMask`/`toAccessMask` helpers (or duplicate tables kept in lockstep), not re-hardcode masks.

### A5. Cross-batch barriers (batch N+1 vs batch N) — GPU OK; planner host state needs explicit rules

- **Severity:** MAJOR
- **Evidence:**
  1. **Same-queue barrier vs prior submissions:** Barriers recorded in CB N+1 execute in queue submission order after CB N’s commands. Sync2 memory dependencies apply to prior work on that queue matching `srcStageMask` once submitted. This is standard; the 3-slot ring does not break it.
  2. **3-slot async path:** `endCommandBatch` with `use_fence=false` + timeline signal returns without waiting (`gs_pipeline.cpp:1469–1475`), stashing `slot.pending_signal`. Concurrent slots can be in-flight on the same queue. `beginCommandBatch` waits only for **that slot’s** prior timeline (`waitForPendingBatchSlot`, `gs_pipeline.cpp:943–991`), not for other slots.
  3. **Shared scratch across slots:** Comment + global reuse barrier at batch begin (`gs_pipeline.cpp:814–849`) explicitly covers `TRANSFER|COMPUTE|DRAW_INDIRECT` with full R/W access — the production fix for overlapping batches on shared buffers.
  4. **Gap:** reuse barrier stages **omit** `VK_PIPELINE_STAGE_2_CONDITIONAL_RENDERING_BIT_EXT` and `HOST` (`gs_pipeline.cpp:820–823`). Wave predicates use conditional-rendering stage (`gs_renderer.cpp:3409–3412`, masks at `gs_pipeline.cpp:1641–1642`).
  5. **Host planner:** Spec never says `reset()` / invalidate-on-batch-boundary. If batch N ends with `ConditionalRead` recorded as visible, batch N+1 may **elide** a second conditional read. GPU reuse barrier does **not** include CONDITIONAL_RENDERING → possible under-sync across async slots for predicates.
- **Fix:** Either (a) add `CONDITIONAL_RENDERING` (and document HOST is host-side only) to the begin-batch reuse barrier, or (b) planner `beginCommandBatch` hook that clears `visible_*` / treats all state as “writer known, not visible” / full invalidate. Spec must pick one and test multi-slot timeline submits.

### A6. `registered, never accessed` → conservative on first write — over-sync only

- **Severity:** NIT
- **Evidence:** First write after `createBuffer` needs no prior memory dependency; emitting conservative is safe but inflates audit counts.
- **Fix:** Optional: first write with no readers → no barrier (already have row “writer=NONE, readers=NONE → write → NO”). Align “never accessed” with that row instead of “same as unknown”.

### A7. Emitted barrier ranges: WHOLE_SIZE vs legacy offset/size

- **Severity:** MINOR
- **Evidence:** Legacy `bufferMemoryBarrier` uses `buffer.offset` / `buffer.size` (`gs_pipeline.cpp:1680–1681`, `1723–1725`). Spec forces `offset=0, size=VK_WHOLE_SIZE`. Safe for owned dedicated buffers; stricter than legacy for `size < capacity` (still correct).
- **Fix:** Document; keep WHOLE_SIZE for owned-only design.

### A8. Queue family indices omitted from planner emission

- **Severity:** MINOR
- **Evidence:** Legacy sets both sides to `queue_family_index` (`gs_pipeline.cpp:1678–1679`). Spec does not specify. `VK_QUEUE_FAMILY_IGNORED` is fine for no ownership transfer; mixing styles is cosmetic but tests comparing structs will fail.
- **Fix:** Spec: emit `srcQueueFamilyIndex = dstQueueFamilyIndex = queue_family_index` to match legacy.

---

## B. Mixed-mode safety (spec §2.5)

### B1. Direct GPU writes outside the four instrumented classes

| Site | Path | Touches owned? | Caught by invalidate rules? |
|---|---|---|---|
| `clearDeviceBuffer` fill | `buffer.cpp:404` | Yes (owned / whatever is in `deviceBuffer`) | Spec migrates in P3 — **required before any tagged consumer of cleared buffers** |
| `resizeAndCopyDeviceBuffer` copy/fill | `buffer.cpp:444,474,493` | Yes | Same — migrate in P3 |
| Sentinel `vkCmdFillBuffer` | `gs_renderer.cpp:1194–1196` | Yes (`primitive_depth_keys`) | Yes if surrounding hand-written barriers remain until chain migrates (`1191`, `1197`) — they `invalidate` under §2.5 |
| Readback `vkCmdCopyBuffer` sources | `gs_renderer.cpp:329–348,486,771,812,2504` | **Source often owned/shared-scratch**; **dst readback never registered** (`vmaDestroyBuffer` paths `260+`, not `destroyBuffer`) | Pre-barriers to TRANSFER_READ on owned sources; post HOST_READ on unregistered dst. Until chain migrates, legacy barriers invalidate sources. **Copy itself does not invalidate** if barriers are deleted early |
| `vkCmdUpdateBuffer` | `gs_renderer.cpp:355–359` | Unregistered readback only | N/A for planner |
| Staging upload path | `buffer.cpp:170–225` staging only; no in-tree `vkCmdCopyBuffer` from stager into owned in this file beyond resizeAndCopy | — | Spec’s “upload/download paths” claim is vague; actual device copies are resizeAndCopy + renderer copies |

**Severity:** MAJOR for migration order  
**Fix:** Freeze invariant: **no hand-written barrier may be deleted until every write to those buffers in the chain (fill/copy/dispatch) is planner-recorded.** P3 must ship `clearDeviceBuffer` + `resizeAndCopyDeviceBuffer` planner hooks in the same commit as the first tagged chain. Call out `vkCmdFillBuffer` at `gs_renderer.cpp:1194` and all `vkCmdCopyBuffer` in readback helpers as explicit migration checklist items.

### B2. vksplat raw buffer barriers — **do** touch planner-relevant storage

- **Severity:** MAJOR
- **Evidence:**
  - Compose: `buffers_.pixel_state` / `pixel_depth` barriers at `vksplat_viewport_renderer.cpp:5288–5307` then compute read for compose.
  - Depth readback: barrier on `depth_buffer` at `5796–5811` (src empty — timeline wait is the real edge).
  - Fill helper: sync1 barrier after fill at `481–496`.
  - These buffers are members of `VulkanGSPipelineBuffers` (`buffer.cpp:75–77` owned list). In production they are often **shared-scratch region views** (`bindSharedScratchBuffers` `3442+`, `3497–3499`), not `createBuffer` owned allocations — planner never tracks them (external).
  - When private scratch is used (`allocation != NULL`), compose still issues raw barriers **outside** `VulkanGSPipeline::bufferMemoryBarrier` → **no `invalidate()`**. Planner state can claim visibility that ignores compose’s intervening compute reads (usually same-frame handoff is semaphore/reuse-barrier covered; host-state elision across frames is the risk).
- **Fix:** Spec must state: (1) visualizer barriers never touch the planner API; (2) either invalidate `pixel_state`/`pixel_depth` via an explicit hook when compose runs against privately owned buffers, or document “private pixel_* unsupported / always shared-scratch”; (3) planner does not own compose queue sync.

### B3. Shared-scratch path makes most “hot” buffers external

- **Severity:** BLOCKER (for exactness / audit; correctness stays conservative)
- **Evidence:** `bindSharedScratchBuffers` destroys owned allocs and installs `makeResizableRegionView` on one `imported_buffer` (`vksplat_viewport_renderer.cpp:3439–3444`). Nearly all sort/tile/pixel/scan buffers become **offset≠0 views of one external `VkBuffer`**. Spec §2.2: external → never registered → always conservative-unknown.
- **Consequence:** On the production viewport path, the planner does **not** compute exact barriers for the hot chains; every access re-emits TRANSFER\|COMPUTE write src. That is **not under-sync**, but:
  1. Epic performance thesis collapses for the main path.
  2. Barrier **struct counts** go **up** vs hoisted hand-written barriers → **§2.6 audit assert fails** systematically (see F).
- **Fix:** Spec must either:
  - **Register the shared-scratch parent `VkBuffer` once** (generation tied to `shared_scratch_.generation` at `3278/3341`) and track whole-buffer state for all region views of it, or
  - Explicitly scope exact planning to private/`createBuffer` buffers and **disable / redefine** BarrierChainAudit for chains that bind shared scratch, or
  - Region-aware keys `(VkBuffer, offset, size)` for external views.

### B4. Mixed-mode rule gap: untagged executeCompute invalidates then runs with **no** barrier

- **Severity:** MINOR (existing behavior)
- **Evidence:** Spec §3.1: untagged becomes invalidate + shared impl **without planning**. Today untagged also has no automatic barriers (callers hand-write). Preserves behavior.
- **Fix:** None for correctness; document that untagged remains “caller must barrier”.

---

## C. Owned-only registry / whole-buffer keying

### C1. `createBuffer` always `offset = 0` dedicated `VkBuffer` — TRUE

- **Severity:** NIT (positive)
- **Evidence:** `buffer.cpp:242` sets `offset = 0`; `vmaCreateBuffer` one buffer per call (`257–258`). `destroyBuffer` clears offset (`335`). Resize destroys + recreates (`354–357`).

### C2. Nonzero-offset views only from visualizer view helpers — TRUE in practice

- **Severity:** NIT (positive for keying **of owned**)
- **Evidence:** `makeBufferView` / `makeRegionView` / `makeResizableRegionView` / `makeBorrowedBufferView` at `vksplat_viewport_renderer.cpp:812–868` set arbitrary `offset`. Rasterizer `createBuffer` never does.
- **Comment already admits coalesced views:** `gs_pipeline.cpp:1937–1940` (“coalesced views into a parent allocation”).

### C3. Whole-buffer keying is sound **only** for owned; production rebinds fields to multi-region external

- **Severity:** BLOCKER (same root as B3)
- **Evidence:** Many `VulkanGSPipelineBuffers` fields flip between owned (`allocation != NULL`) and external views of **one** parent with disjoint offsets (`bindSharedScratchBuffers`). Whole-buffer keying on the parent would over-sync regions (safe); **not registering** the parent under-utilizes the planner and breaks audit (B3).
- **Fix:** Spec must define registration for shared-scratch parent + generation, and what happens on `destroyBuffer` when rebinding views (`3439` destroys previous owned before view assign — `forget` must run).

### C4. Readback buffers intentionally unregistered — OK

- **Severity:** NIT
- **Evidence:** Spec cites raw `vmaDestroyBuffer` at gs_renderer readback teardown. Confirmed pattern at `289–291` etc. Host-mapped readbacks never go through `destroyBuffer`.

---

## D. Image tracker (spec §4)

### D1. `image_generation` split vs compose overwrite — necessary and sufficient **if** all tracker calls use it

- **Severity:** MAJOR if call-site discipline is incomplete
- **Evidence:** Dual-use today: `++slot.generation` at recreate `4976`; compose assigns `output.generation = ++output_generations_[...]` at `5229` and `5368`. After compose, `slot.generation` is a **content** counter, not image identity — generation-keyed tracker **cannot** use `slot.generation` as-is (recon B §2.1).
- **Spec fix is right:** dedicated `image_generation` bump only in `ensureOutputImages`; compose keeps content gen on another field.
- **Must specify:** every `transitionImage` / `imageLayout` / `registerImage` / `forgetImage` for output color+depth passes **`image_generation`**, including compose path `transitionToProducer` / `releaseToFragmentSampling` / `imageLayout` probe at ~5163, and readback restore transitions (`5937+` per recon).
- **Fix:** Freeze a table of call sites → which generation field; fail compile with non-defaulted params (spec already requires non-defaulted).

### D2. Resize sequence with generation-in-entry

- **Severity:** MINOR (ordering detail)
- **Evidence:** Current order: forget → destroy → create → register → `++generation` (`4906–4976`). With generation-in-entry:
  - Correct order: `old_gen = slot.image_generation`; `forget(img, old_gen)`; destroy; create; `++slot.image_generation` **or** bump then `register(new, new_gen)` consistently; never register with a gen that was already forgotten as stale.
- **Risk:** If `++generation` stays **after** register using the **old** gen value, then compose/content code reading the field is fine, but the next forget must use the gen stored **in the tracker entry**, not a post-compose content gen.
- **Fix:** Spec: `registerImage` stores the generation argument; slot field is updated to that same value in the same function; forget always uses the slot’s `image_generation` captured **before** destroy.

### D3. `forgetImage` generation-match vs release/reset

- **Severity:** MINOR
- **Evidence:** `releaseOutputSlot` `1769–1775`, `reset` `2022–2028`: forget then destroy then `slot = {}`. Generation-match forget works if called with the slot’s image gen **before** zeroing. Stale forget after re-register of recycled handle with new gen is a no-op — desired #1478 behavior.
- **Fix:** Document capture-before-clear; add unit test: register gen=2 same handle after forget gen=1 missed → state preserved.

### D4. Restore-on-failure re-register — OK under same generation

- **Severity:** NIT (positive)
- **Evidence:** Spec §4.1; recon point_cloud `1797`, mesh `378`. Layout rewind must pass **same** `image_generation` — not `++`.

### D5. Same-layout early return must accumulate readers — correct fix for sharp edge

- **Severity:** NIT (positive)
- **Evidence:** Today early-out at `vulkan_image_barrier_tracker.cpp:141–142` / `190–191` returns without updating anything. Spec §4.2 fixes multi-reader WAR. Good.

### D6. Local trackers compile impact

- **Severity:** MINOR
- **Evidence:** Recon B: local trackers in UI texture, scene uploader, RmlUi (+ stack ephemeral). Non-defaulted generation forces all sites to change. Ephemeral stack tracker in RmlUi upload can use gen=1 constant if map is short-lived.
- **Fix:** Spec: ephemeral trackers may use a fixed generation `1` for the lifetime of the stack object; still must pass the argument.

### D7. `clearSwapchainOnly` + generation-in-entry

- **Severity:** NIT
- **Evidence:** Erases non-external by handle (`tracker.cpp:69–77`). Generation stored in entry is dropped with the entry — fine for swapchain recreate.

### D8. Reader accumulation vs scope-taking empty source (cross-queue)

- **Severity:** MINOR
- **Evidence:** Compose / readback pass **empty** source scopes when semaphore is the edge (`recon_b` §1.3). Reader accumulation must not invent fake src accesses that fight empty external dependencies.
- **Fix:** Spec: when caller passes explicit scopes, use them as barrier src/dst; reader_* accumulation still updates from destination only; do not OR reader_* into src when caller provided empty external src (cross-queue). Clarify interaction.

---

## E. Seam extension (spec §3.2)

### E1. Routing `bufferMemoryBarrier` through `vulkan_dispatch_.cmd_pipeline_barrier2`

- **Severity:** MINOR (behavior change only if dispatch is incomplete)
- **Evidence:** Today `bufferMemoryBarrier` calls **`vkCmdPipelineBarrier2` directly** (`gs_pipeline.cpp:1693`, `1742`), **not** the dispatch table. `beginCommandBatch` already requires non-null `cmd_pipeline_barrier2` (`789–795`) and uses it for the reuse barrier (`849`). Constructor sets `VulkanDispatch::real()` (`200`). `initializeExternal` does **not** reset dispatch (`226–279`). `setVulkanDispatch` replaces wholesale (`203–204`).
- **Risk:** Scripted tests that zero-init dispatch and only set submit/begin paths will start failing once barriers go through dispatch (good for coverage). No production path calls `bufferMemoryBarrier` before dispatch is `real()` or test-set.
- **Fix:** Spec OK; note dual path today (direct vs dispatch) is inconsistent — unifying is good.

### E2. Extending `VulkanDispatch` with bind/push/dispatch/fill/copy

- **Severity:** MINOR (source rebuild; test churn)
- **Evidence:** `VulkanDispatch` is a POD of PFNs (`vulkan_wait.hpp:94–118`). Adding fields is not stable C ABI across TUs without rebuild — all in-tree users recompile. `VulkanDispatch::real()` (`vulkan_wait.cpp:156–174`) must assign new symbols. Tests (`test_vksplat_failed_submit_no_publish.cpp:169–180`) zero-init and only set a subset — **safe** as long as those tests never call tagged dispatch that uses null PFNs. `test_vulkan_wait_bounded.cpp` same pattern.
- **Fix:** Spec: (1) `real()` fills all new PFNs; (2) tagged path asserts non-null like beginCommandBatch; (3) scripted chain tests must mock the new entry points. Not a freeze blocker.

### E3. `vk_cmd_push_descriptor_set_` stays outside VulkanDispatch

- **Severity:** NIT
- **Evidence:** Push descriptors use a separate device proc (`gs_pipeline.cpp:261–262`, executeCompute `1951`). Spec’s recordable chain still cannot fully script descriptors without extending further.
- **Fix:** Optional follow-up; not required for barrier-only tests if barrier/dispatch are enough.

---

## F. Audit counter (spec §2.6) — not well-defined as written

### F1. Wave-loop baselines are data-dependent and under-specified

- **Severity:** BLOCKER
- **Evidence:** Spec §2.6 / §6 admits “function of waves or wave-0 only; implementer documents.” That is not a freeze-ready gate. Catalog:
  - `executeLegacyDepthWaves`: hoisted **24** structs at L1387 once; per-wave barriers at 1487/1531/1555/1594/1629/1664/1679/1746; nested `executeSortIndirectCountImpl` (6 barrier sites) **per wave**; optional batched `executeCumsum` with `additional_begin_barriers`.
  - `executeMacroDepthWaves`: hoisted **18** at L2988; similar per-wave pattern + cumsum.
- **Wave count** is GPU/content dependent (`armed` / needed waves), not a compile-time constant.

### F2. Proposed exact baseline formulas (catalog-derived)

Define for audit:

```text
// Barrier *structs* (VkBufferMemoryBarrier2 entries), not vkCmdPipelineBarrier2 calls.
// Nested helpers counted inside the parent chain when inlined in the same batch.

legacy_depth_waves(W, batched):
  // W = number of waves actually recorded this call
  base = 24                         // L1387 hoist (once)
  per_wave = 12                     // L1487 (12 entries; batch_* may add more — use max recorded set size)
           + 2                      // L1531 sorted keys/idx
           + 1                      // L1555 batch_offsets (batched path only; 0 if not)
           + 5                      // L1594
           + 2                      // L1629
           + 2                      // L1664
           + 1                      // L1679
           + 2                      // L1746 (once after loop? catalog shows after loop — count once)
  // Catalog places 1746 after the wave loop — treat as once, not ×W
  sort_impl(W) = W * (4 + 4 + 2 + 2 + 2 + 2)
                 // L2210(4)+L2251(≤4)+L2255(2)+L2271(2)+L2294(2)+L2315(2)
                 // L2251 unsorted keys optional if wave_barriers_hoisted — two modes
  cumsum_batched = batched ? executeCumsum_barrier_structs(levels) : 0
  // executeCumsum local: 1 begin (5-ish) + up to 6 mid-phase pair barriers depending on element count
  total = base + per_wave_variable(W) + sort_impl(W) + cumsum_batched

macro_depth_waves(W):
  base = 18                         // L2988
  + per-wave set from L3034(14), 3106(2), 3125(1), 3133(1), 3184(3), 3197(2)
  + sort_impl(W) + cumsum_with_tile_ranges_extra
```

**Still incomplete** without freezing batched vs light vs gut vs overlay branches — each changes L1487 size and which dispatches run.

### F3. “derived ≤ handwritten” fights exact migration **and** shared-scratch conservative mode

- **Severity:** BLOCKER
- **Evidence:** Spec wants fewer/equal structs while also removing hoists and planning per dispatch. Exact planning often emits **more, tighter** barriers than one fat hoist (more structs, less over-sync). Shared-scratch unknown path emits **even more**. Gate as written will fail valid migrations.
- **Fix:** Replace gate with one of:
  1. **Edge coverage:** every hand-written RAW/WAW/WAR edge has a derived barrier with src/dst stage/access ⊇ required (spec §5 already mentions edge-coverage for LOD).
  2. **Upper bound on over-wide masks:** count of barriers whose src is full TRANSFER_COMPUTE_WRITE decreases.
  3. Per-chain **branch-frozen** baselines (explicit `W`, gut/light/batch flags) for debug assert only in unit tests with forged W — not production GPU paths with variable W.

### F4. LOD baseline 19 is incomplete

- **Severity:** MAJOR
- **Evidence:** Spec: `executeSelectLodThreshold` = 12+5+2 = 19. Catalog local barriers match L1077/1112/1134. Nested `recordLodSelectionReadback` adds **4+1 = 5** structs (L795, L820). Spec waves “plus nested when in-chain” without a number. clearDeviceBuffer fills are not barrier structs today but become planner TransferWrite accesses after P3 (may add barriers).
- **Fix:** Publish named constants, e.g.:
  - `kAuditSelectLodThresholdTop = 19`
  - `kAuditSelectLodThresholdWithReadback = 24`
  - MapLodIndices = 3+1 = 4
  - Document P3 clear recording is allowed to add ≤N transfer barriers.

---

## G. Test plan (spec §5) — missing cases that hide subtle sync bugs

### G1. Indirect dispatch argument buffer RAW/WAR

- **Severity:** MAJOR (missing test)
- **Evidence:** Many chains write dispatch args then `executeComputeIndirect` (`gs_renderer.cpp` sort/cull/waves). Spec §3.1 adds implicit `{indirect, IndirectRead}` — good — but tests only say “tagged executeCompute emits one barrier2”.
- **Add:** Write `Indirect` buffer as ComputeWrite → next `executeComputeIndirect` must barrier to `DRAW_INDIRECT|INDIRECT_COMMAND_READ` with src SHADER_WRITE; double-declare same buffer in bindings + implicit merge.

### G2. Conditional-rendering predicate

- **Severity:** MAJOR (missing)
- **Evidence:** `executeWavePartition` post barrier to `CONDITIONAL_RENDERING_READ` (`gs_renderer.cpp:3409–3412`). Begin-batch reuse omits that stage (A5).
- **Add:** ComputeWrite → ConditionalRead exact masks; elision of second ConditionalRead; **cross-batch** elision must still be safe (or fail open with barrier).

### G3. HOST_READ vs fence

- **Severity:** MAJOR (missing)
- **Evidence:** Readback barriers to HOST_READ (`gs_renderer.cpp:360–361`, `491`, `776`, `820`). Host observation also requires `endCommandBatch` fence/timeline wait (`gs_pipeline.cpp:1383+`). Barrier alone is insufficient for host reads.
- **Add:** Test that planner emits HOST stage barrier; document that HostRead does **not** replace fence; negative test: no “planner makes host coherent without wait” claim.

### G4. `executeCumsum` ping-pong / multi-level

- **Severity:** MAJOR (missing)
- **Evidence:** Cumsum uses block_sums as R/W across phases; in-level1 dispatch binds block_sums as both in and out (`catalog` L3264 pattern for visible variant; classic cumsum L1906–1999).
- **Add:** Multi-phase plan sequence with same buffer ReadWrite each phase → barriers between phases; single plan() with duplicate binding merges once; `additional_begin_barriers` migration path (tile_ranges CSW→CSR).

### G5. Shared-scratch / external view path

- **Severity:** BLOCKER if B3 unaddressed
- **Add:** Plan against unregistered handle → conservative src equals `TRANSFER_COMPUTE_SHADER_WRITE`; two region views same `VkBuffer` different offsets both conservative; parent registration (if adopted) over-syncs both regions on one write.

### G6. Cross-batch planner state + 3 slots

- **Severity:** MAJOR (missing)
- **Add:** Scripted: batch N writes buffer, end with timeline no fence; batch N+1 reads — must not elide below reuse-barrier guarantees; conditional-rendering case.

### G7. Sentinel fill + projection

- **Severity:** MINOR (missing)
- **Add:** Compute access → TransferWrite fill → ComputeReadWrite; src after fill is TRANSFER only (epic headline exactness).

### G8. Invalidate + legacy barrier mixed chain

- **Severity:** MINOR (partially covered)
- **Add:** Migrated dispatch, then legacy `bufferMemoryBarrier`, then migrated dispatch — middle invalidate forces conservative on third.

### G9. Image tracker: compose content gen must not be passed to tracker

- **Severity:** MINOR
- **Add:** After compose bumps content generation, transitions still use stable `image_generation`; death test if content gen used and mismatches entry.

### G10. Barrier emission queue family / WHOLE_SIZE shape

- **Severity:** NIT
- **Add:** Snapshot expected struct fields for one RAW barrier.

---

## H. Other reject reasons

### H1. Production path vs owned-only design (restate)

- **Severity:** BLOCKER
- **Evidence:** B3/C3. Spec reads as if `VulkanGSPipelineBuffers` members are owned `createBuffer` results. Production vksplat rebinds most of them to external multi-region scratch. Architecture section never mentions shared scratch.
- **Fix:** Mandatory recon subsection + design choice before freeze.

### H2. Audit gate contradicts exact multi-barrier emission

- **Severity:** BLOCKER
- **Evidence:** F1–F3. “≤ handwritten struct count” is the wrong metric for removing hoists.

### H3. Simultaneous multi-access semantics unspecified

- **Severity:** MAJOR
- **Evidence:** A2. Without freeze, two implementers diverge; one under-syncs.

### H4. No planner hook on `beginCommandBatch` / `HostGuard` batch restart

- **Severity:** MAJOR
- **Evidence:** `HostGuard` ends and restarts batches mid-operation (`gs_pipeline.h:382–423`) during resize/destroy. GPU gets reuse barrier; planner host visibility may elide incorrectly across the restart for stages not in the reuse mask (A5).
- **Fix:** Define planner behavior on batch begin (clear visibility or full invalidate).

### H5. Spec claims “bufferMemoryBarrier would call through dispatch” as current state — false

- **Severity:** MINOR (doc bug)
- **Evidence:** §3.2 wording “currently would call… directly; route BOTH” is slightly confused; code **does** call directly today. Intent is fine.

### H6. First migrated chain still mostly conservative on inputs

- **Severity:** MINOR (expectation management)
- **Evidence:** LOD inputs `node_bounds`, `node_links`, `chunk_to_page`, page tables come from `makeRegionView` external meta (`vksplat` ~2477+). Only lod_gpu_* / chunk_touch owned outputs get exact tracking. Still a valid P3 pilot — document so reviewers don’t expect full exactness.

### H7. `BufferUse` lacks combined multi-stage dst tags used by hand-written posts

- **Severity:** MINOR
- **Evidence:** Hand-written `TRANSFER_COMPUTE_SHADER_INDIRECT_READ` (`gs_renderer.cpp:3404`) is a multi-stage dst. Planner uses single-use scopes + successive consumers. Correct if every consumer plans; wrong if one post-barrier is expected to unlock transfer+compute+indirect at once for **unmigrated** consumers after a migrated producer with no trailing barrier (§3.1: no post-dispatch barrier).
- **Fix:** When a migrated chain is followed by **legacy** code that assumed a fat post-barrier, either keep a final planner `plan()` for all consumer scopes or delay deleting the hand-written post-barrier until consumers migrate. Spec’s “legacy pre-barriers cover compute writes” is true for **src** conservatism, not for **dst stage** coverage of conditional/indirect/transfer without a barrier into those stages.

### H8. Out-of-scope “image barriers inside gs_renderer (none)” — OK

- **Severity:** NIT
- Confirmed: buffer-only in gs_renderer; images live in visualizer tracker.

### H9. Timeline waits need no planner action — OK with caveat

- **Severity:** NIT
- **Evidence:** Spec §2.5. True for CUDA-interop producers. Caveat: does not make planner aware of external writes — next access should stay conservative or invalidate after wait if external wrote an **owned** buffer (rare). Shared interop tensors are external views → already conservative.

### H10. Leak report / ExternalSemaphore scope — OK

- **Severity:** NIT
- Spec §4.3 matches recon B gaps; shutdown hook after idle before destroy is right (`vulkan_context.cpp:586+`).

---

## Verdict

### **NOT FREEZE-READY**

### Blockers (must resolve in the frozen text)

1. **B3/C3/H1 — Shared-scratch external rebinding:** Spec’s owned-only registry does not match production buffer binding. Must design parent registration / audit exclusion / region keys before implementation.
2. **F1–F3/H2 — BarrierChainAudit “derived ≤ handwritten struct count”:** Ill-defined for data-dependent wave counts; actively conflicts with de-hoisting and with conservative external path. Replace with edge-coverage and/or branch-frozen test-only baselines with explicit formulas (F2 as starting point).
3. **A5/H4 — Cross-batch planner visibility vs reuse barrier stage mask:** CONDITIONAL_RENDERING (and batch-begin policy) must be specified or under-sync is possible with 3 in-flight timeline slots.

### Majors (should fix before freeze; freeze-with-waiver only if explicitly deferred)

- Mixed-mode checklist for fill/copy sites and visualizer compose barriers on `pixel_*` (B1–B2).
- Simultaneous `plan()` semantics (A2/H3).
- LOD / chain baseline constants fully numeric including nested readbacks (F4).
- Test plan gaps G1–G6.
- Legacy post-barrier multi-dst stage coverage when deleting trailing barriers (H7).
- Image tracker: force `image_generation` (not content gen) on **all** transition/imageLayout sites (D1).

### What is solid enough to keep

- Hazard table core rows (WAR NONE, RAW from writer, visibility elision, conservative = `TRANSFER_COMPUTE_SHADER_WRITE`) — verified against `toStageMask`/`toAccessMask`.
- `createBuffer` offset=0 owned keying for true private allocs.
- Image generation split for #1478 dual-use bug.
- Reader accumulation same-layout fix.
- Routing barriers through existing `cmd_pipeline_barrier2` seam; extending `VulkanDispatch` is workable.
- Mixed-mode invalidate on legacy `bufferMemoryBarrier` / untagged executeCompute as a principle (needs shared-scratch + visualizer amendments).

---

*End of critique.*
