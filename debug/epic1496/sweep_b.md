# Sweep B — Dead/stale code + latent bugs after generation-keyed tracker migration (#1478/#1488)

**Branch:** `epic-1496-computed-barriers`  
**Scope:** `src/visualizer/window/` + `src/visualizer/rendering/` + `src/visualizer/gui/`  
**Tools:** `ast-grep` (`sg -p '…' -l cpp`) primary; `rg` for call-site inventory / comments  
**Mode:** READ-ONLY — no production file modifications  
**Baseline:** recon `debug/epic1496/recon_b_image_tracker.md` (pre-migration bare-handle map; dual-use `slot.generation`)

Format per finding: **file:line**, **pattern**, **evidence**, **verdict** (`DEAD` / `STALE` / `BUG` / `OK`), **proposed minimal action**.

---

## Executive summary

| Area | Result |
|---|---|
| `VulkanImageBarrierTracker::reset()` production callers | **Still none** (DEAD API surface) |
| Wrong generation for tracker (`slot.generation` vs `image_generation`) | **No production tracker call uses content-gen** — dual-field split landed |
| register/forget pair-check | Shared owners **OK**; **RmlUi render layers forget with gen 0** (BUG) |
| `transitionImageLayoutImmediate` bypass | **3+1 sites** — intentional cross-API handoff; do not blindly migrate |
| Census create/destroy structural pairing | Success paths **OK**; **fail-path destroy → onDestroy without onCreate** (BUG) |
| Stale comments / dual maps | RmlUi parallel `m_image_barrier_generations` + layer gap; issue text still pre-migration |

---

## 1. `VulkanImageBarrierTracker::reset()`

### F-B01 — `reset()` has zero production (and zero unit-test) callers

- **file:line:** `src/visualizer/window/vulkan_image_barrier_tracker.hpp:41`, `…/vulkan_image_barrier_tracker.cpp:128-130`
- **pattern:** `rg 'image_barriers.*\.reset\(|imageBarriers\(\)\.reset\(|tracker\.reset\('` over `src/` + `tests/`; `sg -p '$_.reset()' -l cpp` (noise only)
- **evidence:** Sole hit is the method definition. `clearSwapchainOnly()` is the only production clear path (`vulkan_context.cpp:4489` in `destroySwapchain`). `tests/test_vulkan_image_tracker.cpp` never calls `reset()`. Matches recon: “`reset()` is never called from production.”
- **verdict:** `DEAD`
- **proposed minimal action:** Prefer **keep + wire** over pure deletion: call `image_barriers_.reset()` at end of `VulkanContext::shutdown()` after the census report (hygiene; device is dying). Alternatively mark as test/utility only with a one-line comment “no production caller; intentional full wipe.” Removing without a caller or test is fine if you want a thinner API — nothing relies on it today.

---

## 2. Generation plumbing — dual-use / wrong key

### F-B02 — vksplat content gen vs resource gen: migration complete

- **file:line:** `vksplat_viewport_renderer.hpp:577-580`; tracker uses at `vksplat_viewport_renderer.cpp:1771,1774,2024,2027,4910,4913,4971-4981,5169+,5930+,6090+,6264+`
- **pattern:** `sg -p '$_.registerImage($$$)'` / `forgetImage` / `transitionImage`; `rg 'slot\.generation|output\.generation|image_generation'`
- **evidence:**
  - `generation` = content identity (compose bumps via `output_generations_[]` at `:5238`, `:5377`; published on results).
  - `image_generation` = resource identity; bumped only in `ensureOutputImages` (`++slot.image_generation` at `:4971`) before `registerImage`.
  - All `imageBarriers().{register,forget,transition,imageLayout}` on output slots pass `slot/output.image_generation`, never `generation`.
- **verdict:** `OK`
- **proposed minimal action:** None. Keep the header comments that document the split.

### F-B03 — point-cloud same split: OK

- **file:line:** `point_cloud_vulkan_renderer.cpp:462-466`, `:1111-1119` (register), `:1135/:1148` (forget), `:1915` (content `++slot.generation`)
- **pattern:** same as F-B02
- **evidence:** Tracker always uses `slot.image_generation`. Content gen still separate for result consumers.
- **verdict:** `OK`
- **proposed minimal action:** None.

### F-B04 — mesh offscreen / UI texture / scene uploader: dedicated image gens OK

- **file:line:**  
  - mesh: `mesh_offscreen_renderer.cpp:66`, `:144-145`, `:326-330`  
  - UI: `vulkan_ui_texture.cpp:177`, `:541/:964/:990`, `:610-611/:720-721`  
  - scene uploader: `vulkan_scene_image_uploader.cpp:33-34`, `:65-67`, `:212-213/:251`
- **pattern:** `sg -p '$_.registerImage($$$)'` / `forgetImage`
- **evidence:** Single resource-identity counter per owner; forget uses the same field as register. Scene uploader correctly picks `scene_image_external_generation` vs `owned_image_generation` in `clearSceneImageBinding`.
- **verdict:** `OK`
- **proposed minimal action:** None.

### F-B05 — swapchain uses `swapchain_epoch_`, not a dual-use content counter

- **file:line:** `vulkan_context.hpp:438-440`; register `:4048-4053`, `:4189-4192`; transitions pass `swapchain_epoch_`
- **pattern:** `rg 'swapchain_epoch_'`
- **evidence:** Bumped once per `createSwapchain`; no forget (uses `clearSwapchainOnly`). External entries survive rebuild via `Entry.external`.
- **verdict:** `OK`
- **proposed minimal action:** None.

### F-B06 — No production site still keys the tracker with content-only `*.generation`

- **file:line:** n/a (negative result)
- **pattern:** `rg 'imageBarriers\(\)\.(registerImage|forgetImage|transitionImage|imageLayout)\([^;]*\.generation[^_]'` — empty; `rg '(registerImage|forgetImage)\([^)]*\.generation\b'` — empty
- **evidence:** Content gens remain for descriptors/publish (`depth_blit`/`split_view` `external_image_generation`, result payloads) but are not tracker keys.
- **verdict:** `OK`
- **proposed minimal action:** None. (Optional: assert in code review checklist that new tracker call sites never take compose/content gen.)

### F-B07 — RmlUi parallel bare-handle gen map is leftover dual plumbing

- **file:line:** `rmlui_vk_backend.hpp:160`, `:661-662`; write `:761-762`, `:1263-1264`; read `:2417-2425`; erase `:2239-2240`
- **pattern:** `rg 'm_image_barrier_generations|m_barrier_generation'`
- **evidence:** After migration, tracker already stores generation in `Entry`. RmlUi still maintains:
  1. `texture_data_t::m_barrier_generation`
  2. `unordered_map<VkImage,uint64_t> m_image_barrier_generations`
  Textures update both on create. Layers (F-B08) update only the map via lazy transition. Dual bookkeeping is pre-migration-style and a footgun.
- **verdict:** `STALE`
- **proposed minimal action:** Collapse to one source of truth (prefer field on `texture_data_t` + set it for layers at create; drop the side map once all owners write the field). Keep counter `m_image_barrier_generation`.

---

## 3. Forgotten forget — structural register/destroy pairing

### F-B08 — **BUG:** RmlUi render-layer destroy forgets with generation `0`

- **file:line:**  
  - create (no gen assign): `rmlui_vk_backend.cpp:2318-2394` (`EnsureRenderLayer`)  
  - lazy gen mint (map only): `:2413-2427` (`TransitionImageLayout`)  
  - destroy path: `:2632-2636` → `QueueTextureForDeferredDeletion(new texture_data_t(layer.m_color|depth))` → `:2239` `forgetImage(..., texture.m_barrier_generation)`
- **pattern:** structural pair: `registerImage`/`TransitionImageLayout` gen source vs `forgetImage` gen source per owner
- **evidence:**
  1. `EnsureRenderLayer` never sets `layer.m_color.m_barrier_generation` / depth equivalent (stay `0`).
  2. First `TransitionImageLayout` mints `generation = ++m_image_barrier_generation` into **map only**, then `registerImage(image, generation, ...)`.
  3. `DestroyRenderLayer` copies `texture_data_t` with `m_barrier_generation == 0`.
  4. `Destroy_Texture` calls `forgetImage(image, 0)` → generation mismatch → **no erase** of the live entry (see tracker `forgetImage` match rule `cpp:148-149`).
  5. Map erase still happens (`:2240`), so the side map and tracker diverge until handle reuse overwrites via `registerImage`.
- **verdict:** `BUG` (latent; soft until handle reuse races a transition on a stale entry)
- **proposed minimal action:** On layer image create (both color + depth), assign:
  ```cpp
  layer.m_color.m_barrier_generation = ++m_image_barrier_generation;
  m_image_barrier_generations[layer.m_color.m_p_vk_image] = layer.m_color.m_barrier_generation;
  // same for depth_stencil
  ```
  Optionally `registerImage` at create with `UNDEFINED`. Destroy already uses `m_barrier_generation`.

### F-B09 — Shared tracker owners: forget pairs OK

| Owner | register | forget | gen source | verdict |
|---|---|---|---|---|
| `VulkanContext` swapchain/depth | `:4050`, `:4189` | `clearSwapchainOnly` `:4489` | `swapchain_epoch_` | `OK` (by design) |
| `VksplatViewportRenderer` | `:4972-4981` | `:1771-1774`, `:2024-2027`, `:4910-4913` | `slot.image_generation` | `OK` (forget before destroy) |
| `point_cloud` | `:1112-1119`, restore `:1806-1813` | `destroySlot` `:1135/:1148` | `slot.image_generation` | `OK` |
| `mesh_offscreen` | `:327-330`, restore `:382-385` | `destroyTargets` `:144-145` | `image_generation` | `OK` |
| `VulkanUiTexture` | `:611/:721`, reseed `:511` | `:541/:964/:990` | `image_generation_` | `OK` |
| `VulkanSceneImageUploader` | `:213/:251` | `clearSceneImageBinding` `:67` | external vs owned gen | `OK` |
| RmlUi **textures** (not layers) | create `:761-762/:1263-1264` | `:2239` | `m_barrier_generation` | `OK` |
| RmlUi **layers** | lazy `:2425` | `:2239` with gen 0 | **mismatch** | `BUG` → F-B08 |
| RmlUi stack upload | `:1359` (ephemeral tracker) | N/A (stack dies) | const gen | `OK` |

- **pattern:** `sg -p '$_.registerImage($$$)'` + `sg -p '$_.forgetImage($$$)'` over `src/visualizer`
- **verdict:** table above
- **proposed minimal action:** Only F-B08 needs code change.

### F-B10 — Scene uploader owned destroy: forget after `vmaDestroyImage` (order OK, note)

- **file:line:** `vulkan_scene_image_uploader.cpp:98-116`
- **pattern:** structural read of destroy vs `clearSceneImageBinding`
- **evidence:** Owned path destroys view/image then calls `clearSceneImageBinding()` which forgets using the still-held handle value, then zeros. Correct for map erase; slightly surprising order only.
- **verdict:** `OK`
- **proposed minimal action:** Optional style: forget before destroy for consistency with other owners.

---

## 4. `transitionImageLayoutImmediate` bypass sites (list + judgement only)

| Site | Context | Judgement |
|---|---|---|
| `viewport_interop_service.cpp:430-434` | New interop target: `UNDEFINED→GENERAL` + signal timeline | **Keep immediate.** Cross-API one-shot handoff outside frame recording; images never registered on shared tracker. Local consumers (scene uploader) re-register with external generation when binding. |
| `viewport_interop_service.cpp:509-513` | Upload path: `target.layout→GENERAL` + signal | **Keep immediate.** Same CUDA export/import ownership dance; tracker would not replace semaphore edges. |
| `viewport_interop_service.cpp:555-560` | Publish path: `GENERAL→SHADER_READ_ONLY` + wait CUDA signal | **Keep immediate.** Consumer sampling is on graphics with separate layout bookkeeping (`published_image_layout` / uploader bind). |
| `vulkan_ui_texture.cpp:677-680` | Interop ensure: `UNDEFINED→GENERAL` + signal before CUDA import | **Keep immediate for init.** After success, local tracker `registerImage(..., GENERAL)` at `:721`. Do **not** leave long-lived UI samples only on immediate path. |

- **pattern:** `sg -p '$_.transitionImageLayoutImmediate($$$)' -l cpp src/visualizer`
- **verdict:** all `OK` as bypasses (intentional), not migration leftovers
- **proposed minimal action:** Document in a short comment near `transitionImageLayoutImmediate` declaration: “not a tracker client; for out-of-frame CUDA↔Vulkan handoffs only.” No code move unless a future design merges interop into the shared tracker (out of scope).

---

## 5. Census (`#1488`) — create/destroy structural pairing

### F-B11 — Success-path instrumentation is centralized and complete

- **file:line:**  
  - Image: onCreate `vulkan_context.cpp:3060`; onDestroy `:3083-3085`  
  - Buffer create: onCreate `:3229`; import onCreate `:3385`; onDestroy `:3404-3406`  
  - Semaphore: onCreate `:3503`; onDestroy `:3520-3522`  
  - Shutdown report `:604-628`
- **pattern:** `sg -p '$_.onCreate($$$)'` / `onDestroy`; `sg -p '$_.createExternalImage($$$)'` etc.
- **evidence:** Every successful `createExternal*` / `importExternalBuffer` ends with exactly one `onCreate`. Every `destroyExternal*` with `was_live` does one `onDestroy`. Call sites do not touch census directly (good).
- **verdict:** `OK` (success paths)
- **proposed minimal action:** None for success paths.

### F-B12 — **BUG:** fail-path `destroyExternal*(out)` before `onCreate` flags underflow

- **file:line:** e.g. `createExternalImage` mid-fail `destroyExternalImage(out)` at `:2983,:2991,:3003,:3011,:3025,:3037,:3052` then success-only onCreate `:3060`. Same pattern for buffer (~7–11 fails) and semaphore (3 fails).
- **pattern:** structural: success path reaches `onCreate` once; destroy always `onDestroy` if `was_live`
- **evidence:** Partial allocation sets `image`/`memory`/`view` non-null → `was_live == true` → `onDestroy` without prior `onCreate` → `underflow_flagged_ = true` (`gpu_object_census.cpp:16-17`). Any real create failure permanently poisons the flag for the process lifetime (never cleared except `census.reset()`, unused in production).
- **verdict:** `BUG`
- **proposed minimal action:** Gate census on a dedicated flag, e.g. `bool counted = false` set only after `onCreate`, and `onDestroy` only if `counted`. Or private `scrubExternalImage` for fail paths that skips census. Prefer the flag so double-destroy and fail scrub stay safe.

### F-B13 — `underflowFlagged()` never observed at shutdown

- **file:line:** `gpu_object_census.hpp:40`; shutdown report `vulkan_context.cpp:604-628` only logs `report()` survivors
- **pattern:** `rg 'underflowFlagged'` → tests only + definition
- **evidence:** Production never logs or asserts underflow. Combined with F-B12, false underflows are invisible; true destroy-without-create bugs are also invisible.
- **verdict:** `STALE` (API half-wired)
- **proposed minimal action:** In shutdown census block: `if (gpu_object_census_.underflowFlagged()) LOG_WARN(...)`. After F-B12 fix so fail-paths do not false-trip.

### F-B14 — Owner call-site create/destroy balance (External* inventory)

| Owner | Creates | Destroys | Notes |
|---|---|---|---|
| `VksplatViewportRenderer` | output images, many buffers, timelines | `releaseOutputSlot` / `reset` / ensure fail rollback | OK structurally |
| `ViewportInteropService` | image+sem per target | `VulkanSceneInteropTarget::destroy` | OK; no tracker register |
| `vulkan_ui_texture` | interop image+sem | `destroyImage` | OK |
| `vulkan_external_tensor` | buffer / fail destroys | dtor + fail | OK |
| Point cloud / mesh | VMA only | n/a for census | OK |

- **pattern:** `sg -p '$_.createExternal*($$$)'` + destroy counterparts
- **verdict:** `OK` at call-site level (instrumentation bug is inside `VulkanContext`, F-B12)
- **proposed minimal action:** None at owners.

---

## 6. Stale comments / pre-epic narrative in tree

### F-B15 — `GITHUB_ISSUES.md` still describes pre-migration tracker

- **file:line:** `GITHUB_ISSUES.md:185-199` (outside strict src scope but cited by recon)
- **pattern:** `rg '14 register|bare handle|external_images_'`
- **evidence:** Still claims “14 register/forget…5 files” and shows `unordered_map<VkImage, ImageState>` + `unordered_set` external set. Code now uses `unordered_map<VkImage, Entry>` with generation in entry; counts higher.
- **verdict:** `STALE` (docs)
- **proposed minimal action:** Close/update issue text or mark “fixed on epic-1496”; do not trust counts for remaining work.

### F-B16 — Tracker/UI “re-register every transition” defeats reader accumulation

- **file:line:** `vulkan_ui_texture.cpp:511-512`; `rmlui_vk_backend.cpp:2425-2426`
- **pattern:** `registerImage` immediately before `transitionImage` with caller `old_layout`
- **evidence:** Each transition reseeds tracker state from caller-supplied `old_layout`, wiping `last_*` / `reader_*` accumulated by the tracker. Correct if callers always pass accurate old layout; wastes/ignores sub-task-4 reader accumulation on those local trackers.
- **verdict:** `STALE` (pre-tracker-trust pattern)
- **proposed minimal action:** Prefer `transitionImage` alone when image already registered; use `registerImage` only on create/bind. Low priority if validation is clean.

### F-B17 — Production comments that *are* current (not stale)

- **file:line:** `vksplat_viewport_renderer.hpp:577-580`, `point_cloud_vulkan_renderer.cpp:462-465`, `vulkan_context.hpp:438-439`, `gpu_object_census.hpp:30`, shutdown `#1488` comment `:604`
- **pattern:** `rg '#1478|#1488|Resource identity|Not a tracker'`
- **evidence:** Accurately describe post-migration roles.
- **verdict:** `OK`
- **proposed minimal action:** Keep.

### F-B18 — Spec still accurate on `reset()`; dual-use section describes the *fix* already applied

- **file:line:** `EPIC_1496_BARRIER_SPEC.md:241-267`
- **pattern:** read of §4.1 generation sources
- **evidence:** Spec’s “add `image_generation`” is now implemented; language reads as design-time. Not a code bug.
- **verdict:** `STALE` (spec prose tense only)
- **proposed minimal action:** Optional past-tense pass when epic closes.

---

## 7. Checklist coverage matrix

| Checklist item | Status | Primary findings |
|---|---|---|
| `reset()` production callers | **Still none** | F-B01 DEAD |
| Leftover generation plumbing / wrong key | Content vs image gens **split correctly** on shared owners; RmlUi dual map stale | F-B02–F-B07 |
| Forgotten forget pair-check | **One real miss:** RmlUi layers | F-B08 BUG; F-B09 OK table |
| `transitionImageLayoutImmediate` should-use-tracker | **List only — keep as bypass** | §4 |
| Census create/destroy structural | Success OK; fail-path underflow BUG; underflow unread | F-B11–F-B14 |
| Stale comments / old bare-handle narrative | Issue text + reseed pattern | F-B15–F-B18 |

---

## 8. Priority action list (minimal)

1. **P0 / BUG** — F-B08: assign `m_barrier_generation` (and map entry) when creating RmlUi layer color/depth images so destroy forget matches.
2. **P0 / BUG** — F-B12: do not `onDestroy` on External* fail-path scrubs that never `onCreate`’d.
3. **P1 / STALE** — F-B13: log `underflowFlagged()` at shutdown once F-B12 is fixed.
4. **P2 / DEAD** — F-B01: wire `image_barriers_.reset()` into shutdown or document/remove.
5. **P2 / STALE** — F-B07: collapse RmlUi dual gen map after F-B08.
6. **P3** — F-B16 reseeding; F-B15 issue text; F-B18 spec tense.

---

## 9. Method notes

- Primary structural queries:
  - `sg -p '$_.registerImage($$$)' -l cpp src/visualizer`
  - `sg -p '$_.forgetImage($$$)' -l cpp src/visualizer`
  - `sg -p '$_.transitionImageLayoutImmediate($$$)' -l cpp src/visualizer`
  - `sg -p '$_.onCreate($$$)'` / `onDestroy` / `createExternalImage|Buffer|TimelineSemaphore`
- Negative proofs used `rg` for wrong-field tracker args and for `reset()` callers.
- No builds or tests executed (read-only sweep).
- Line numbers are for the tree at sweep time on `epic-1496-computed-barriers`; re-verify if the branch moves.
