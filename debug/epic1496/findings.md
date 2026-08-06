#Epic #1496 campaign findings(side discoveries)

## Pre-existing: VUID-vkCmdDispatch-storageBuffers-06936 in vksplat.cumsum (2026-08-06)
GPU-assisted validation (scripts/run_vulkan_validation.sh) reports out-of-range storage buffer
access in the cumsum pipeline, 10+ hits per session, on a tree whose rasterizer code is
bit-identical to master (only image-tracker call sites changed at that point). Not caused by the
epic. Candidate real bug: cumsum shader indexing past the bound descriptor range for some element
counts. USER ORDER: fix on this branch (no filing/postponing).

RESOLVED 2026-08-06 after two investigation rounds (inv_cumsum_report.md, inv_r2_report.md in
the session scratchpad; verdicts mirrored here):
- The app is correct. A `require_backing` contract added to `executeCumsum` (permanent) proves
  every classic scan's input/output/block-sum backings are large enough at record time; it never
  fired while the VUID still appeared. SPIR-V artifacts clean;
recording is single - threaded;
  core+sync validation is clean on the same workloads.
- The reports are GPU-assisted-validation false positives in the pinned layer
  (vulkan-validationlayers 1.4.341.0): GPU-AV pairs stale push-descriptor snapshots with
  instrumented dispatches — Vulkan-ValidationLayers issue #11433 (verified open, same VUID,
  same push-descriptor mechanism, reporter confirms errors vanish without push descriptors).
- Fix on this branch: scripts/run_vulkan_validation.sh now defaults to core + synchronization
  validation (the epic's gate) with GPU-AV behind an explicit --gpu-av flag documenting the
  false-positive class. Revisit the flag default when the layer pin advances past a #11433 fix.

## Sweep resolution (2026-08-06, tree 24a506368 + fix packet)

Disposition of verified findings from `sweep_a.md`, `sweep_b.md`, `sweep_c.md`.

### Sweep A
| ID | Verdict | Disposition |
|----|---------|-------------|
| F1 | DEAD BarrierMask ×7 | **Fixed** — deleted TRANSFER_READ_WRITE, TRANSFER_COMPUTE_SHADER_READ, TRANSFER_COMPUTE_SHADER_READ_WRITE, HOST_WRITE, HOST_READ_WRITE, COMPUTE_SHADER_INDIRECT_READ, TRANSFER_COMPUTE_SHADER_INDIRECT_READ + converter arms. Kept TRANSFER_COMPUTE_SHADER_WRITE (tests). Enum doc updated. |
| F2 | STALE bufferMemoryBarrier API | **Resolved: retained by design** — the epic (sub-task 3) requires `bufferMemoryBarrier()` to stay callable; the mixed-mode invalidate tests exercise it. Removal is a post-epic follow-up once nothing needs legacy emission. |
| F3–F7, F10–F14 | OK | No code change. |
| F8 | overlay_flags ComputeWrite on raster/compose | **Fixed** — 7 sites retagged to ComputeRead; projection producers remain Write. |
| F9 | wave_buffer ReadWrite in keygen | **Fixed** — both keygen sites → ComputeRead. |
| F15 | STALE spec prose | **Fixed** — status banner added to EPIC_1496_BARRIER_SPEC.md marking §0 as the pre-migration record. |

### Sweep B
| ID | Verdict | Disposition |
|----|---------|-------------|
| F-B01 | DEAD reset() | **Fixed** — deleted `VulkanImageBarrierTracker::reset()` decl+def (zero callers). |
| F-B02–B06, B09–B11, B14, B17 | OK | No change. |
| F-B07 + F-B08 | STALE dual map + layer gen 0 | **Fixed** — layers mint `m_barrier_generation` from `m_image_barrier_generation` at create (color+depth); removed `m_image_barrier_generations` map; TransitionImageLayout takes generation; external swapchain has `m_external_swapchain_barrier_generation`. |
| F-B12 | fail-path census underflow | **Fixed** — `census_counted` on ExternalImage/Buffer/Semaphore; set only after onCreate; onDestroy only if counted. Census-level underflow contract already covered by `GpuObjectCensus` unit tests; app ordering is the fix. |
| F-B13 | underflowFlagged never read | **Fixed** — shutdown census block logs a WARN when the underflow flag is set. |
| — | census live proof | **Verified 2026-08-06** — clean exit via `lichtfeld.force_exit()` prints "GpuObjectCensus: no live external GPU objects at shutdown" (zero leaks). Note: `request_exit()` intentionally routes through the UI ExitOperator (confirm flow); `force_exit()` is the unconditional path. Not a bug. |
| F-B14 | owner-side External* inventory | **Resolved: covered** — the census itself is the runtime inventory; per-owner static tables would duplicate it. |
| F-B15 | GITHUB_ISSUES.md stale tracker text | **Resolved: local scratch** — untracked working notes, not shipped by this branch; the issues themselves close when the PR lands. |
| F-B16 | UI/RmlUi re-register-per-transition defeats reader accumulation | **Accepted-risk (documented)** — pre-existing pattern, over-sync only (never under-sync); changing UI transition seeding needs focused visual validation out of scope here. |
| F-B18 | spec tense on reset()/dual-use | **Fixed** via the F15 status banner. |

### Sweep C
| ID | Verdict | Disposition |
|----|---------|-------------|
| C1.1 | shallow Buffer copy | **Accepted-risk** — doc comment on `Buffer<T>` pointing at sweep_c.md (no refactor). |
| C1.2 | ScopedStagingBuffer copy | **Fixed** — copy/move deleted. |
| C1.3 | ManagedBuffer copy | **Fixed** — copy deleted; move nulls source. |
| C2.1 | visible cumsum require_backing | **Fixed** — host contract on input/output/block_sums/block_sums2 vs visible_capacity-derived sizes. |
| C2.2 | prepare_visible_sort deviceSize | **Fixed** — throw if `visible_prefix.deviceSize() < num_splats`. |
| C2.3 | prepare_tile_sort deviceSize | **Fixed** — classic + visible paths throw if indexed buffer too small. |
| C3 S1 prepare_tile_sort.slang | — | **VERIFIED-GUARDED** — `if (uniforms.num_splats > 0u)` before `[num_splats-1]` (L31–36 classic; L23–29 visible). |
| C3 S2 prepare_visible_sort.slang | — | **VERIFIED-GUARDED** — `if (uniforms.num_splats > 0u)` before `[num_splats-1]` (L19–20). |
| C3 S3 compact_visible_primitives.slang | missing capacity clamp | **Fixed** — `if (compact_idx >= uniforms.num_splats) return;` before writes (lod_* style). |
| C4 | missing sizeof asserts | **Fixed** — `static_assert` for LodCompact (16) and SelectionPolygonRasterize (32). |


## Sweep round 2 resolution (2026-08-06)

Reports: `sweep_d.md`, `sweep_e.md`, `sweep_f.md`, `sweep_g.md`.

### Sweep G
| ID | Disposition |
|----|-------------|
| G3-1/G3-2/G3-3 | **Fixed** — binding 2 (`count` / `tile_instance_count`) retagged `ComputeWrite` → `ComputeRead` on tile_ranges_and_batch_counts, tile_ranges, and macro_ranges host dispatches. |

### Sweep F
| ID | Disposition |
|----|-------------|
| F-F07 | **Fixed** — `release(gpu_lod_tree_.page_age)` before `gpu_lod_tree_ = {}`. |
| F-F01 | **Fixed** — deleted `lodPageCacheSnapshot` decl+def. |
| F-F02 | **Fixed** — deleted `ChannelPolicy::has_flip_y` + writes. |
| F-F03–F06, §4 | **OK** (no change). |

### Sweep D
| ID | Disposition |
|----|-------------|
| F-D01 | **Fixed** — deleted `_Stager`, `stager`, `allocStagingBuffer`, cleanup arm. |
| F-D08 | **Fixed** — `writeTimestamp` → `void`; callers no longer branch on bool. `writeTimestampNoExcept` remains `bool`. |
| F-D09 | **Fixed** — `diagnosticStageScope` cases for `CullSplats` + `ProjectionSurvivors`. |
| F-D10 | **Fixed** — removed unused includes from `gs_pipeline.h` (`algorithm`, `cstring`, `map`, `variant`) and `buffer.h` (`cmath`, `cstring`, `map`, `mutex`). |
| F-D11 | **Fixed** — deleted unreferenced `_VulkanBuffer::operator==` + stale descriptor-cache comment. |
| F-D02–D07, D12 | **OK / STALE retained** (not in mandatory list). |

### Sweep E
| ID | Disposition |
|----|-------------|
| E-E01 | **OK** — no change. |
| E-E02 | **Fixed** — removed CMake rule, spirv map entry, pipeline member+create, and `seed_primitive_indices.slang`. |
| E-E03 | **Fixed** — deleted dead `LFS_VISIBLE_BOUNDED` branches from `tile_shader.slang` (no CMake product; HiGS uses macro keygen). |
| E-E04 | **Kept with contract comment** — `#if LFS_3DGUT_PROJECTION` in `cull_splats.slang` documents future HiGS-cull+3DGUT combo; still compiled as 0 today. |
| E-E05–E09 | **Not in mandatory list** this packet. |


## Performance A/B (2026-08-06, splat_30000.rad 1M-splat LOD scene, default view, 43 frames/run)

App GPU timestamps (`vksplat.gpu.*`), identical 28s protocol per run:

| Stage | master (2 runs) | branch (3 runs) |
|---|---|---|
| RasterizeForward | 2.463 / 2.439 ms | 0.977 / 1.246 / 1.184 ms |
| all 14 other stages | — | identical within noise |

The rasterize/compose wave stage runs ~2x faster: the deleted 24/18-entry mega-hoist barriers
and per-wave exact masks remove GPU over-waiting exactly where the epic's premise predicted the
serialization lived. Whole-frame gains are smaller (GUI/compose/CPU unchanged). Protocol and raw
logs in the session scratchpad (perf_*_summary.txt).
