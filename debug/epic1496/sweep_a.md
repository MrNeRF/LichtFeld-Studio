# Sweep A — Epic #1496 computed-barrier migration leftovers

**Branch:** `epic-1496-computed-barriers` (tip includes wave/cumsum/cull/selection migrations)  
**Scope:** `src/rendering/rasterizer/vulkan/` (+ tests only where needed to verify residual callers)  
**Method:** ast-grep structural patterns primary; rg where patterns are enum-name / text residual  
**Mode:** READ-ONLY — no repo modifications  

---

## Executive summary

| Area | Result |
|------|--------|
| Production `bufferMemoryBarrier` callers | **Zero** (defs + tests only) |
| `additional_begin_barriers` / `begin_cumsum` | **Gone** from production (docs/recon only) |
| `wave_barriers_hoisted` | **Live** (not dead) — true for wave paths, false for non-wave |
| Dead `BarrierMask` variants | **7** unreferenced outside enum + `toStageMask`/`toAccessMask` |
| Dead `BufferUse` | **`HostWrite`** (0 references) |
| `kAudit*` | **Tests only** — OK |
| Tag vs slang sample (13 simple dispatches) | **Match** |
| Tag vs slang (raster overlays + keygen wave) | **Mis-tags** (over-write; not under-sync) |
| Duplicate / misused planTransfer | Dual-stage handoffs are intentional merges — OK |

---

## Findings

### F1 — Dead `BarrierMask` variants (enum + converters only)

**file:line:** `src/rendering/rasterizer/vulkan/src/gs_pipeline.h:112–128` (enum); bodies at `gs_pipeline.cpp:1614–1683`  
**ast-grep / pattern:** `enum BarrierMask { $$$ }` + rg `\bVARIANT\b` over `src/rendering/rasterizer/vulkan` + `tests`  
**evidence:** The following variants appear **only** in the enum definition and the `toAccessMask` / `toStageMask` switch bodies — **no** production call site and **no** test value:

| Variant | Sites outside enum+converters |
|---------|-------------------------------|
| `TRANSFER_READ_WRITE` | none |
| `TRANSFER_COMPUTE_SHADER_READ` | none |
| `TRANSFER_COMPUTE_SHADER_READ_WRITE` | none |
| `HOST_WRITE` | none |
| `HOST_READ_WRITE` | none |
| `COMPUTE_SHADER_INDIRECT_READ` | none |
| `TRANSFER_COMPUTE_SHADER_INDIRECT_READ` | none |

Still referenced as **values** (tests and/or comments):  
`TRANSFER_READ/WRITE`, `COMPUTE_SHADER_*`, `HOST_READ`, `INDIRECT_DISPATCH_READ`, `CONDITIONAL_RENDERING_READ`, and **`TRANSFER_COMPUTE_SHADER_WRITE`** (conservative-src golden in `tests/test_vksplat_barrier_planner.cpp` + comment in `barrier_planner.cpp:52`).

**verdict:** DEAD (7 variants)  
**proposed minimal action:** Keep while `bufferMemoryBarrier` + `toStageMask`/`toAccessMask` remain (tests still exercise converters). When legacy API is removed (F2), delete the seven unused enumerators and the matching arms in both converters in the same PR. Do not remove `TRANSFER_COMPUTE_SHADER_WRITE` until tests stop using it as the conservative golden.

---

### F2 — `bufferMemoryBarrier` overloads: production-dead, tests-alive

**file:line:**  
- Decl: `gs_pipeline.h:171`, `gs_pipeline.h:177`  
- Def: `gs_pipeline.cpp:1686`, `gs_pipeline.cpp:1744`  
- Callers: `tests/test_vksplat_tagged_dispatch.cpp:621` (pair overload), `:662` (`BufferBarrier` overload)

**ast-grep pattern:** `bufferMemoryBarrier($$$)`  
**rg fallback:** `bufferMemoryBarrier\s*\(` across repo  

**evidence:**

| Location | Role |
|----------|------|
| `gs_pipeline.{h,cpp}` | declarations + implementations (still call `invalidate()` then `cmd_pipeline_barrier2`) |
| `gs_renderer.cpp` | **0** call sites |
| Other `src/` | **0** call sites |
| Tests | Exactly **two** tests: `LegacyBufferMemoryBarrierInvalidates`, `LegacyBufferBarrierOverloadInvalidates` (G8 mixed-mode) |

Both overloads are exercised; both are required for the epic mixed-mode contract until those tests are rewritten against `planTransfer`/`invalidate` alone.

**verdict:** STALE (API retained by epic; production-dead)  
**proposed minimal action:**  
1. **Now:** add a deprecation comment on both decls in `gs_pipeline.h`, e.g. “Legacy epic-#1496 mixed-mode only; production chains use tagged plan; tests only.”  
2. **Follow-up PR (post-epic):** delete both overloads + `BufferBarrier` struct once G8 tests are rewritten to call `barrierPlanner().invalidate()` + a single planned barrier; then collapse unused `BarrierMask` (F1).

---

### F3 — `additional_begin_barriers` / `begin_cumsum` leftovers

**file:line:** production — none  
**pattern:** rg `additional_begin_barriers|begin_cumsum`  
**evidence:** Zero hits under `src/`. Residuals only in:

- `EPIC_1496_BARRIER_SPEC.md:342` (migration note: parameter removed with last caller)  
- `debug/epic1496/recon_a_barrier_catalog.md`, `debug/epic1496/critique.md` (pre-migration catalog)

Current `executeCumsum` signature (`gs_renderer.h:335–340`, `gs_renderer.cpp:1761–1765`) is `(buffers, input, output, record_timestamps = true)` only. Helper lambda `execute_cumsum_phase` is live (not a leftover).

**verdict:** OK (fully removed from code)  
**proposed minimal action:** None for code. Optional doc hygiene: strike “82 hand-written barriers” problem statement in `EPIC_1496_BARRIER_SPEC.md:14` (now false — see F10).

---

### F4 — `wave_barriers_hoisted` is **not** dead plumbing

**file:line:**  
- Param: `gs_renderer.h:358`, `gs_renderer.cpp:2027`  
- Branch: `gs_renderer.cpp:2129–2137`  
- Callers:  
  - `true` — `executeLegacyDepthWaves` `:1486`, `executeMacroDepthWaves` `:2878`  
  - `false` — `executeSortIndirectCount` wrapper `:2014`

**ast-grep pattern:** `wave_barriers_hoisted`  
**evidence:** When `true`, after histogram clear the impl `planTransfer`s key reads so keygen→radix is not lost if the wave path only planned reuse at a higher level. When `false`, upsweep’s own `ComputeRead` on keys covers the edge. Both branches are reachable.

**verdict:** OK (live, intentional)  
**proposed minimal action:** None. Optional rename to `plan_keygen_handoff` if the “hoisted” wording confuses post-migration readers; pure clarity, not cleanup.

---

### F5 — Stale mega-hoist comments (documentation residue)

**file:line:** `gs_renderer.cpp:1423`, `gs_renderer.cpp:2816`  
**pattern:** rg `Mega-hoist|hoist`  
**evidence:** Comments state mega-hoist removed and per-dispatch planning is exact — accurate narrative residue, not dead code.

**verdict:** OK  
**proposed minimal action:** Leave or trim in a docs-only pass.

---

### F6 — `kAudit*` constants

**file:line:** `tests/test_vksplat_tagged_dispatch.cpp` only (e.g. `:853+`, `:1167+`)  
**pattern:** rg `kAudit` under `src/rendering/rasterizer/vulkan` → **0**  
**evidence:** Audit ceilings live exclusively in tagged-dispatch unit tests (§2.6). None in production headers.

**verdict:** OK  
**proposed minimal action:** None.

---

### F7 — `BufferUse::HostWrite` never used

**file:line:** `barrier_planner.h:20–28` (enum has no `HostWrite`); switch arms use `HostRead` only  
**pattern:** rg `BufferUse::HostWrite` / `HostWrite` → **0** in src+tests  

Note: legacy `BarrierMask::HOST_WRITE` exists (F1) but the planner never modeled host-write as a `BufferUse`.

**verdict:** OK for `BufferUse` (never added). Related dead surface is `BarrierMask::HOST_WRITE` (F1).  
**proposed minimal action:** Do not add `HostWrite` to `BufferUse` unless a real host-write producer appears; remove `BarrierMask::HOST_WRITE` with F1/F2.

---

### F8 — BUG: `overlay_flags` tagged `ComputeWrite` on raster/compose paths (shader is read-only)

**file:line (production tags that disagree with slang):**

| Site | Path |
|------|------|
| `gs_renderer.cpp:1541` | legacy light raster |
| `:1559` | tile-batch raster overlays |
| `:1592` | compose overlays |
| `:1646`, `:1675` | 3DGUT / plain alphablend overlays |
| `:2921`, `:2944` | macro raster / compose overlays |

**Correct Write tags (projection producers):** `:1279`, `:2540` — match `vertex_shader.slang` `layout(binding=13) RWStructuredBuffer out_overlay_flags`.

**ast-grep / method:** extract `std::vector<TaggedBinding>{ $$$ }` + cross-check slang  
**shader evidence:**

```text
alphablend_shader.slang:165  layout(binding=13) StructuredBuffer<uint> gaussian_overlay_flags;  // R
tile_batch_shader.slang:108  StructuredBuffer gaussian_overlay_flags  // R
macro_raster.slang:77        StructuredBuffer gaussian_overlay_flags  // R
macro_compose.slang:40       StructuredBuffer gaussian_overlay_flags  // R
```

Reads only (`gaussian_overlay_flags[splat_id]`).

**hazard impact:** Over-tag (planner thinks raster **writes** flags). Causes extra WAW/RAW barriers; **not** under-sync. After raster, writer state is still “compute write,” which is coincidentally similar to the real projection write, so later readers stay ordered — but the mid-frame state is wrong and can inflate barrier counts / defeat elision.

**verdict:** BUG (incorrect tag; over-sync class)  
**proposed minimal action:** Change raster/compose overlay tags from `BufferUse::ComputeWrite` → `BufferUse::ComputeRead` at the seven sites above. Keep projection tags as `ComputeWrite`. Add a unit assertion or comment binding index ↔ slang RW.

---

### F9 — BUG: `wave_buffer` tagged `ComputeReadWrite` in keygen (shader is read-only)

**file:line:**  
- `gs_renderer.cpp:1475` — `executeLegacyDepthWaves` / `pipeline_generate_keys_wave`  
- `gs_renderer.cpp:2866` — `executeMacroDepthWaves` / `pipeline_generate_macro_keys_wave`

**shader evidence:**

```text
tile_shader.slang:67   layout(binding=7) StructuredBuffer<uint32_t> depth_wave_dispatch; // R
macro_tile_shader.slang:147  layout(binding=8) StructuredBuffer depth_wave_dispatch;     // R
```

Keygen only **reads** rank/base/instance fields; writes are to unsorted keys/indices (bindings 5–6), correctly tagged `ComputeWrite`.

**hazard impact:** Over-tag. After keygen, planner marks the depth-wave record buffer as compute-written even though the real last writer was wave-partition / prior compute. Subsequent `IndirectRead` still gets a barrier (safe, wider than needed). Can block useful elision on the wave record between keygen and sort.

**verdict:** BUG (incorrect tag; over-sync class)  
**proposed minimal action:** Change `{wave_buffer, BufferUse::ComputeReadWrite}` → `BufferUse::ComputeRead` at both sites. Re-check audit baselines if they encode barrier counts for wave chains.

---

### F10 — Sampled tagged dispatches vs slang: 13/13 simple chains OK

**method:** structural extract of `executeCompute(…, pipeline_*, TaggedBinding{…})` + slang `layout(binding=N)` RW flags  

| Dispatch | Line | Shader | Result |
|----------|------|--------|--------|
| map LOD | 1103 | `lod_map_indices.slang` | OK |
| select LOD | 1149 | `lod_select_threshold.slang` | OK (counts/chunk_touch RW) |
| compact touch | 1182 | `lod_compact_touch.slang` | OK |
| selection mask | 1712 | `selection_mask.slang` | OK |
| polygon raster | 1750 | `selection_polygon_rasterize.slang` | OK |
| prepare tile sort | 1979 | `prepare_tile_sort.slang` (2-bind variant) | OK |
| prepare tile sort visible | 3095 | 3-bind variant | OK |
| radix hist clear | 2121 | `radix_histogram_clear.slang` | OK |
| visible flags | 2249 | `visible_flags.slang` | OK |
| prepare visible sort | 2276 | `prepare_visible_sort.slang` | OK |
| compact visible | 2298 | `compact_visible_primitives.slang` | OK |
| apply depth | 2390 | `apply_depth_ordering.slang` | OK |
| prepare visible chain | 2606 | `prepare_visible_chain.slang` | OK |
| cull + prepare | 2441/2460 | `cull_splats.slang` entries | OK |
| radix upsweep/spine/downsweep | 2155/2171/2185 | `*.comp` readonly/restrict/writeonly | OK |

**verdict:** OK for sampled set  
**proposed minimal action:** None for these. Prefer fixing F8/F9 next.

---

### F11 — Dual-use `planTransfer` handoffs (TransferRead + ComputeRead) — not misuse

**file:line examples:**  
- `gs_renderer.cpp:1988–1992` (`tile_sort_count`)  
- `:2287–2290` (`visible_count`)  
- `:3106–3110`, `:3189–3191`  

**pattern:** consecutive `DeclaredAccess` on same buffer with different `BufferUse` in one `planTransfer`  
**evidence:** Planner merges per `VkBuffer` (`barrier_planner.cpp:164–195`) by OR-ing stages/access; `is_write` is OR. Intent: open both transfer and compute consumers in one hazard (handoff rule §3.4.5). Not a duplicate call and not API misuse.

**verdict:** OK  
**proposed minimal action:** None. Optional comment “merged dual-stage handoff” if reviewers keep tripping.

---

### F12 — No duplicate consecutive planTransfer pairs that re-declare the same sole access

**pattern:** scan `planTransfer` sites; flag pairs within 20 lines  
**evidence:** Close pairs are always pre-copy vs post-host-read (different uses, sandwiching `vkCmdCopyBuffer`), or dual-stage merge (F11). No accidental double-plan of identical single-use sets found.

**verdict:** OK  
**proposed minimal action:** None.

---

### F13 — PerfTimer stages still referenced

**file:line:** `perf_timer.h` stage list; uses in `gs_renderer.cpp` (`PerfTimer::Timer<…>` for Projection, Raster, Cumsum, Sort, Cull, etc.)  
**pattern:** rg `PerfTimer::Timer<`  
**evidence:** No orphaned stage enum from the migration; CPU stage timers around sort passes remain live.

**verdict:** OK  
**proposed minimal action:** None.

---

### F14 — Planner `stats()` production-unused (tests-only)

**file:line:** `barrier_planner.h:47–51`, `:68`; callers only in `tests/test_vksplat_*.cpp`  
**evidence:** Production never reads `barriers_emitted` / `accesses_elided` / `conservative_fallbacks`. By design for audit tests.

**verdict:** OK  
**proposed minimal action:** None (keep for §2.6 audit).

---

### F15 — Spec problem statement stale

**file:line:** `EPIC_1496_BARRIER_SPEC.md:14`  
**evidence:** Claims “`gs_renderer.cpp` has 82 hand-written `bufferMemoryBarrier` calls.” Sweep finds **0** production callers (F2). Spec still useful as historical design doc; the opening is wrong post-P4.

**verdict:** STALE (docs)  
**proposed minimal action:** One-line amend: “pre-migration: 82 …; post-P4: zero production callers; API retained for mixed-mode tests.”

---

## Checklist coverage

| Checklist item | Status |
|----------------|--------|
| BarrierMask variants unreferenced outside converters/tests | **F1** — 7 dead |
| bufferMemoryBarrier remaining callers | **F2** — tests only; deprecation plan |
| begin_cumsum / additional_begin_barriers | **F3** — removed |
| wave_barriers_hoisted plumbing | **F4** — live, not dead |
| perf-timer / kAudit / includes orphans | **F6, F13** OK; no barrier-related unused includes flagged |
| tag vs slang (sample 10+) | **F10** OK; **F8, F9** mis-tags |
| duplicate planTransfer / planner misuse | **F11–F12** OK |

---

## Recommended fix order (minimal)

1. **F8 + F9** (tag correctness) — small, high signal; re-run wave audit tests if counts shift.  
2. **F2 comment** — deprecation note on legacy barrier API.  
3. **F15** — optional spec header fix.  
4. **F1 + F2 removal** — only after epic closes and G8 tests no longer need legacy overloads.

---

## Tools / patterns used

```text
ast-grep -p 'bufferMemoryBarrier($$$)' --lang cpp
ast-grep -p 'wave_barriers_hoisted' --lang cpp
ast-grep -p 'enum BarrierMask { $$$ }' --lang cpp
ast-grep -p 'BufferBarrier{$$$}' --lang cpp
ast-grep -p 'auto $NAME = [&]($$$ARGS) { $$$BODY };' --lang cpp   # cumsum phase lambda live

rg 'bufferMemoryBarrier\s*\(' 
rg '\b(TRANSFER_READ_WRITE|HOST_WRITE|…)\b'
rg 'additional_begin_barriers|begin_cumsum|wave_barriers_hoisted|kAudit|BufferUse::'
rg 'planTransfer\s*\('
# + Python structural extract of TaggedBinding vs slang layout(binding=N)
```

---

*End of sweep_a.md — read-only; no files under the repo were modified.*
