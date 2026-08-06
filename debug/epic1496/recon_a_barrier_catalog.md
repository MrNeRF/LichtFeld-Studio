# Epic1496 Recon A — Barrier / Dispatch Catalog

**Scope:** `src/rendering/rasterizer/vulkan/src/gs_renderer.cpp` + definitions in `gs_pipeline.cpp`  
**Read-only recon.** Line numbers from current tree.  
**`bufferMemoryBarrier` call sites in gs_renderer.cpp: 82** (definitions only in gs_pipeline.cpp; zero other call sites in repo).

## Overloads

From `gs_pipeline.h`:

1. **pair overload:** `bufferMemoryBarrier(vector<pair<Buffer, BarrierMask>> buffers, BarrierMask dstMask)` — shared dst for all entries
2. **BufferBarrier overload:** `bufferMemoryBarrier(vector<BufferBarrier>)` where `BufferBarrier { buffer, src_mask, dst_mask }` — per-entry dst

`BarrierMask` values: `TRANSFER_{READ,WRITE,READ_WRITE}`, `COMPUTE_SHADER_{READ,WRITE,READ_WRITE}`, `TRANSFER_COMPUTE_SHADER_{READ,WRITE,READ_WRITE,INDIRECT_READ}`, `HOST_{READ,WRITE,READ_WRITE}`, `INDIRECT_DISPATCH_READ`, `COMPUTE_SHADER_INDIRECT_READ`, `CONDITIONAL_RENDERING_READ`.

## Shader sources

Under `src/rendering/rasterizer/vulkan/shader/src/`:

- `shader/src/radix_sort/config.glsl`
- `shader/src/radix_sort/downsweep.comp`
- `shader/src/radix_sort/spine.comp`
- `shader/src/radix_sort/upsweep.comp`
- `shader/src/slang/alphablend_shader.slang`
- `shader/src/slang/alphablend_shader_bwd_per_pixel.slang`
- `shader/src/slang/alphablend_shader_bwd_per_splat.slang`
- `shader/src/slang/alphablend_shader_bwd_tensor.slang`
- `shader/src/slang/apply_depth_ordering.slang`
- `shader/src/slang/compact_visible_primitives.slang`
- `shader/src/slang/config.slang`
- `shader/src/slang/copy_visible_indices.slang`
- `shader/src/slang/cull_splats.slang`
- `shader/src/slang/cumsum.slang`
- `shader/src/slang/expected_depth_finalize.slang`
- `shader/src/slang/lod_compact_touch.slang`
- `shader/src/slang/lod_map_indices.slang`
- `shader/src/slang/lod_select_threshold.slang`
- `shader/src/slang/macro_compose.slang`
- `shader/src/slang/macro_raster.slang`
- `shader/src/slang/macro_tile_shader.slang`
- `shader/src/slang/prepare_tile_sort.slang`
- `shader/src/slang/prepare_visible_chain.slang`
- `shader/src/slang/prepare_visible_sort.slang`
- `shader/src/slang/radix_histogram_clear.slang`
- `shader/src/slang/seed_primitive_indices.slang`
- `shader/src/slang/selection_mask.slang`
- `shader/src/slang/selection_polygon_rasterize.slang`
- `shader/src/slang/spherical_harmonics.slang`
- `shader/src/slang/tile_batch_shader.slang`
- `shader/src/slang/tile_shader.slang`
- `shader/src/slang/utils.slang`
- `shader/src/slang/vertex_shader.slang`
- `shader/src/slang/visible_flags.slang`

Usage annotations on dispatches are **inferred** from pipeline name + binding role (and cross-checked against barrier masks). Marked UNKNOWN only when ambiguous.

## Nested helper calls (important for chains)

| Caller | Nested helpers with their own barriers/dispatches |
|--------|-----------------------------------------------------|
| `executeSelectLodThreshold` | `recordLodSelectionReadback` |
| `executeLegacyDepthWaves` | `executeSortIndirectCountImpl` (per wave), `executeCumsum` (batched path) |
| `executeCalculateIndexBufferOffset` | `executeCumsum`, `executePrepareTileSort` |
| `executeSortIndirectCount` | `executeSortIndirectCountImpl` |
| `executeSortPrimitivesByDepth` | `executeCumsum`, `executeSortIndirectCount`→Impl, `recordVisibleCountReadback` |
| `executeSortPrimitivesByDepthVisible` | `recordVisibleCountReadback`, `executeSortIndirectCount`→Impl |
| `executeMacroDepthWaves` | `executeSortIndirectCountImpl`, `executeCumsum` |
| `executeWavePartition` | `recordInstanceCountReadback` |
| `executeCumsum` callers with `additional_begin_barriers` | `executeLegacyDepthWaves` / `executeMacroDepthWaves` pass `{{tile_ranges, CSW→CSR}}` |

---

## Chain: `recordInstanceCountReadback`

_Defined at gs_renderer.cpp:301_  
**Local counts:** barriers=1, dispatches=0, copies=2, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 329 | copybuffer | vkCmdCopyBuffer | `    vkCmdCopyBuffer(command_buffer,                     buffers.tile_sort_count.deviceBuffer.buffer,                    ` |
| 345 | copybuffer | vkCmdCopyBuffer | `    vkCmdCopyBuffer(command_buffer,                     wave_buffer.buffer,                     instance_count_readback_` |
| 355 | updatebuffer | vkCmdUpdateBuffer | `    vkCmdUpdateBuffer(command_buffer,                       instance_count_readback_buffer_.buffer,                     ` |
| 360 | barrier (pair) | `instance_count_readback_buffer_` src=TRANSFER_WRITE | **dst=HOST_READ** | `instance_count_readback_buffer_` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L360** overload=`pair`
- `instance_count_readback_buffer_`: `TRANSFER_WRITE` → `HOST_READ`

</details>

## Chain: `synchronizeTileInstanceGate`

_Defined at gs_renderer.cpp:458_  
**Local counts:** barriers=1, dispatches=0, copies=1, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 486 | copybuffer | vkCmdCopyBuffer | `    vkCmdCopyBuffer(command_buffer,                     count.buffer,                     instance_gate_readback_buffer_` |
| 491 | barrier (pair) | `instance_gate_readback_buffer_` src=TRANSFER_WRITE | **dst=HOST_READ** | `instance_gate_readback_buffer_` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L491** overload=`pair`
- `instance_gate_readback_buffer_`: `TRANSFER_WRITE` → `HOST_READ`

</details>

## Chain: `recordVisibleCountReadback`

_Defined at gs_renderer.cpp:743_  
**Local counts:** barriers=1, dispatches=0, copies=1, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 771 | copybuffer | vkCmdCopyBuffer | `    vkCmdCopyBuffer(command_buffer,                     buffers.visible_count.deviceBuffer.buffer,                     v` |
| 776 | barrier (pair) | `visible_count_readback_buffer_` src=TRANSFER_WRITE | **dst=HOST_READ** | `visible_count_readback_buffer_` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L776** overload=`pair`
- `visible_count_readback_buffer_`: `TRANSFER_WRITE` → `HOST_READ`

</details>

## Chain: `recordLodSelectionReadback`

_Defined at gs_renderer.cpp:784_  
**Local counts:** barriers=2, dispatches=0, copies=1, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 795 | barrier (pair) | `b.lod_gpu_counts` src=COMPUTE_SHADER_WRITE; `b.lod_compact_counts` src=COMPUTE_SHADER_WRITE; `b.lod_compact_protected` src=COMPUTE_SHADER_WRITE; `b.lod_compact_misses` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_READ** | `b.lod_gpu_counts`, `b.lod_compact_counts`, `b.lod_compact_protected`, `b.lod_compact_misses` |
| 812 | copybuffer | vkCmdCopyBuffer | `        vkCmdCopyBuffer(command_buffer, src.buffer,                         lod_selection_readback_buffer_.buffer, 1, &c` |
| 820 | barrier (pair) | `lod_selection_readback_buffer_` src=TRANSFER_WRITE | **dst=HOST_READ** | `lod_selection_readback_buffer_` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L795** overload=`pair`
- `b.lod_gpu_counts`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`
- `b.lod_compact_counts`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`
- `b.lod_compact_protected`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`
- `b.lod_compact_misses`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`

**L820** overload=`pair`
- `lod_selection_readback_buffer_`: `TRANSFER_WRITE` → `HOST_READ`

</details>

## Chain: `executeMapLodIndices`

_Defined at gs_renderer.cpp:1004_  
**Local counts:** barriers=2, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1023 | barrier (pair) | `b.lod_logical_indices` src=TRANSFER_COMPUTE_SHADER_WRITE; `chunk_to_page` src=TRANSFER_COMPUTE_SHADER_WRITE; `out_indices` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b.lod_logical_indices`, `chunk_to_page`, `out_indices` |
| 1030 | dispatch | `pipeline_lod_map_indices` | [0] `b.lod_logical_indices` (READ)<br>[1] `chunk_to_page` (READ)<br>[2] `out_indices` (WRITE) |
| 1040 | barrier (pair) | `out_indices` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `out_indices` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1023** overload=`pair`
- `b.lod_logical_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `chunk_to_page`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `out_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1040** overload=`pair`
- `out_indices`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeSelectLodThreshold`

_Defined at gs_renderer.cpp:1044_  
**Local counts:** barriers=3, dispatches=2, copies=0, fills/clears=3

**Nested:** ends with `recordLodSelectionReadback` (barriers 795, 820 + copies).

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1064 | fill-clearDeviceBuffer | fill-clearDeviceBuffer | `auto& counts = clearDeviceBuffer(buffers.lod_gpu_counts, 2);` |
| 1072 | fill-clearDeviceBuffer | fill-clearDeviceBuffer | `auto& chunk_touch = clearDeviceBuffer(buffers.lod_chunk_touch, chunk_touch_count);` |
| 1077 | barrier (pair) | `node_bounds` src=TRANSFER_COMPUTE_SHADER_WRITE; `node_links` src=TRANSFER_COMPUTE_SHADER_WRITE; `chunk_to_page` src=TRANSFER_COMPUTE_SHADER_WRITE; `counts` src=TRANSFER_WRITE; `out_indices` src=TRANSFER_COMPUTE_SHADER_WRITE; `out_logical_indices` src=TRANSFER_COMPUTE_SHADER_WRITE; `out_weights` src=TRANSFER_COMPUTE_SHADER_WRITE; `chunk_touch` src=TRANSFER_WRITE; `out_levels` src=TRANSFER_COMPUTE_SHADER_WRITE; `page_age` src=TRANSFER_COMPUTE_SHADER_WRITE; `page_frames` src=TRANSFER_COMPUTE_SHADER_WRITE; `page_to_chunk` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `node_bounds`, `node_links`, `chunk_to_page`, `counts`, `out_indices`, `out_logical_indices`, `out_weights`, `chunk_touch`, `out_levels`, `page_age`, `page_frames`, `page_to_chunk` |
| 1093 | dispatch | `pipeline_lod_select_threshold` | [0] `node_bounds` (READ)<br>[1] `node_links` (READ)<br>[2] `chunk_to_page` (READ)<br>[3] `counts` (READ_WRITE)<br>[4] `out_indices` (WRITE)<br>[5] `out_logical_indices` (WRITE)<br>[6] `out_weights` (WRITE)<br>[7] `chunk_touch` (READ_WRITE)<br>[8] `out_levels` (WRITE)<br>[9] `page_age` (READ)<br>[10] `page_frames` (READ)<br>[11] `page_to_chunk` (READ) |
| 1112 | barrier (pair) | `counts` src=COMPUTE_SHADER_WRITE; `out_indices` src=COMPUTE_SHADER_WRITE; `out_logical_indices` src=COMPUTE_SHADER_WRITE; `out_weights` src=COMPUTE_SHADER_WRITE; `out_levels` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `counts`, `out_indices`, `out_logical_indices`, `out_weights`, `out_levels` |
| 1123 | fill-clearDeviceBuffer | fill-clearDeviceBuffer | `auto& compact_counts = clearDeviceBuffer(buffers.lod_compact_counts, 4);` |
| 1134 | barrier (pair) | `chunk_touch` src=COMPUTE_SHADER_WRITE; `compact_counts` src=TRANSFER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `chunk_touch`, `compact_counts` |
| 1137 | dispatch | `pipeline_lod_compact_touch` | [0] `chunk_touch` (READ)<br>[1] `compact_counts` (WRITE)<br>[2] `compact_protected` (WRITE)<br>[3] `compact_misses` (WRITE) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1077** overload=`pair`
- `node_bounds`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `node_links`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `chunk_to_page`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `counts`: `TRANSFER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `out_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `out_logical_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `out_weights`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `chunk_touch`: `TRANSFER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `out_levels`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `page_age`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `page_frames`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `page_to_chunk`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1112** overload=`pair`
- `counts`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `out_indices`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `out_logical_indices`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `out_weights`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `out_levels`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1134** overload=`pair`
- `chunk_touch`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `compact_counts`: `TRANSFER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

</details>

## Chain: `executeProjectionForward`

_Defined at gs_renderer.cpp:1150_  
**Local counts:** barriers=4, dispatches=1, copies=0, fills/clears=1

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1169 | barrier (pair) | `b.xyz_ws` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.sh0` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.shN` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.rotations` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.scaling_raw` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.opacity_raw` src=TRANSFER_COMPUTE_SHADER_WRITE; `transform_indices` src=TRANSFER_COMPUTE_SHADER_WRITE; `node_mask` src=TRANSFER_COMPUTE_SHADER_WRITE; `overlay_params` src=TRANSFER_COMPUTE_SHADER_WRITE; `model_transforms` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.xyz_ws`, `b.sh0`, `b.shN`, `b.rotations`, `b.scaling_raw`, `b.opacity_raw`, `transform_indices`, `node_mask`, `overlay_params`, `model_transforms` |
| 1191 | barrier (pair) | `primitive_depth_keys` src=COMPUTE_SHADER_READ_WRITE | **dst=TRANSFER_COMPUTE_SHADER_WRITE** | `primitive_depth_keys` |
| 1194 | fillbuffer | vkCmdFillBuffer | `    vkCmdFillBuffer(command_buffer, primitive_depth_keys.buffer,                     primitive_depth_keys.offset, primit` |
| 1197 | barrier (pair) | `primitive_depth_keys` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `primitive_depth_keys` |
| 1218 | barrier (pair-conditional) | `lod_indices (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE; `lod_logical_indices (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE; `lod_levels (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE; `lod_weights (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `lod_indices (if valid)`, `lod_logical_indices (if valid)`, `lod_levels (if valid)`, `lod_weights (if valid)` |
| 1266 | dispatch | `buffers.quant_pool ? (use_gut_projection ? pipeline_projection_forward_quant_3dgut : pipeline_projection_forward_quant) : (use_gut_projection ? pipeline_projection_forward_3dgut : pipeline_projection_forward)` | [0] `projection_buffers` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1169** overload=`pair`
- `b.xyz_ws`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.sh0`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.shN`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rotations`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.scaling_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.opacity_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `transform_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `node_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `overlay_params`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `model_transforms`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1191** overload=`pair`
- `primitive_depth_keys`: `COMPUTE_SHADER_READ_WRITE` → `TRANSFER_COMPUTE_SHADER_WRITE`

**L1197** overload=`pair`
- `primitive_depth_keys`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1218** overload=`pair-conditional`
- `lod_indices (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `lod_logical_indices (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `lod_levels (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `lod_weights (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeLegacyDepthWaves`

_Defined at gs_renderer.cpp:1277_  
**Local counts:** barriers=9, dispatches=10, copies=0, fills/clears=0

**Nested:** each wave calls `executeSortIndirectCountImpl` (barriers 2210–2315). Batched path also calls `executeCumsum` (with `additional_begin_barriers={{tile_ranges,CSW→CSR}}`).

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1387 | barrier (BufferBarrier) | `b.xy_vs` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.inv_cov_vs_opacity` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.rect_tile_space` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.index_buffer_offset` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.primitive_sort_indices` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.rgb` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.depths` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.xyz_ws` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.rotations` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.scaling_raw` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.opacity_raw` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `selection_mask` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `preview_mask` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `selection_colors` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `overlay_flags` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `overlay_params` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `transform_indices` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `model_transforms` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.sorting_keys_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_keys_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b._sorting_histogram` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE; `b._sorting_histogram_cumsum` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE | `b.xy_vs`, `b.inv_cov_vs_opacity`, `b.rect_tile_space`, `b.index_buffer_offset`, `b.primitive_sort_indices`, `b.rgb`, `b.depths`, `b.xyz_ws`, `b.rotations`, `b.scaling_raw`, `b.opacity_raw`, `selection_mask`, `preview_mask`, `selection_colors`, `overlay_flags`, `overlay_params`, `transform_indices`, `model_transforms`, `b.sorting_keys_1`, `b.sorting_keys_2`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `b._sorting_histogram`, `b._sorting_histogram_cumsum` |
| 1487 | barrier (BufferBarrier-built) | `b.sorting_keys_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_keys_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `tile_ranges` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_state` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_depth` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_depth_weight` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `n_contributors` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b._sorting_histogram` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b._sorting_histogram_cumsum` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `batch_* if batched (counts/offsets/descriptors/dispatch/pixel_state/n_contributors)` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE | `b.sorting_keys_1`, `b.sorting_keys_2`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `tile_ranges`, `pixel_state`, `pixel_depth`, `pixel_depth_weight`, `n_contributors`, `b._sorting_histogram`, `b._sorting_histogram_cumsum`, `batch_* if batched (counts/offsets/descriptors/dispatch/pixel_state/n_contributors)` |
| 1506 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kKeygenWordOffset)` `pipeline_generate_keys_wave` | [0] `b.xy_vs` (READ)<br>[1] `b.inv_cov_vs_opacity` (READ)<br>[2] `b.rect_tile_space` (READ)<br>[3] `b.index_buffer_offset` (READ)<br>[4] `b.primitive_sort_indices` (READ)<br>[5] `unsorted_keys` (WRITE)<br>[6] `unsorted_indices` (WRITE)<br>[7] `wave_buffer` (READ_WRITE) |
| 1531 | barrier (BufferBarrier) | `b.sorted_keys()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.sorted_gauss_idx()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `b.sorted_keys()`, `b.sorted_gauss_idx()` |
| 1539 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(
                                       indirect::DepthWave::kPerTileWordOffset)` `pipeline_compute_tile_ranges_and_batch_counts [buffers.is_unsorted_1]` | [0] `b.sorted_keys()` (READ)<br>[1] `tile_ranges` (WRITE)<br>[2] `count` (WRITE)<br>[3] `batch_counts` (WRITE) |
| 1555 | barrier (BufferBarrier) | `batch_offsets` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `batch_offsets` |
| 1558 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(
                                       indirect::DepthWave::kPerTileWordOffset)` `pipeline_tile_batch_descriptors` | [0] `tile_ranges` (READ)<br>[1] `batch_offsets` (READ)<br>[2] `batch_descriptors` (WRITE)<br>[3] `batch_dispatch` (WRITE) |
| 1572 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset)` `light_pipeline[buffers.is_unsorted_1]` | [0] `b.sorted_gauss_idx()` (READ)<br>[1] `tile_ranges` (WRITE)<br>[2] `b.xy_vs` (WRITE)<br>[3] `b.inv_cov_vs_opacity` (WRITE)<br>[4] `b.rgb` (WRITE)<br>[5] `b.depths` (WRITE)<br>[6] `pixel_state` (READ_WRITE)<br>[7] `pixel_depth` (READ_WRITE)<br>[8] `n_contributors` (READ_WRITE)<br>[9] `pixel_depth_weight` (READ_WRITE)<br>[10] `selection_mask` (READ)<br>[11] `preview_mask` (READ)<br>[12] `selection_colors` (READ)<br>[13] `overlay_flags` (WRITE)<br>[14] `overlay_params` (READ) |
| 1594 | barrier (BufferBarrier) | `batch_descriptors` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `batch_dispatch` src=COMPUTE_SHADER_WRITE→dst=INDIRECT_DISPATCH_READ; `pixel_state` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_depth` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `n_contributors` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE | `batch_descriptors`, `batch_dispatch`, `pixel_state`, `pixel_depth`, `n_contributors` |
| 1621 | dispatch-indirect | indirect=`batch_dispatch` offset=`indirect::byteOffset(indirect::TileBatchDispatch::kRasterWordOffset)` `batch_pipeline[buffers.is_unsorted_1]` | [0] `batch_bindings` (UNKNOWN) |
| 1629 | barrier (BufferBarrier) | `batch_pixel_state` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `batch_n_contributors` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `batch_pixel_state`, `batch_n_contributors` |
| 1655 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset)` `overlays_active ? pipeline_compose_tile_batches : pipeline_compose_tile_batches_plain` | [0] `compose_bindings` (READ) |
| 1664 | barrier (BufferBarrier) | `b.sorted_keys()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.sorted_gauss_idx()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `b.sorted_keys()`, `b.sorted_gauss_idx()` |
| 1672 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(
                                       indirect::DepthWave::kRangeWordOffset)` `pipeline_compute_tile_ranges[buffers.is_unsorted_1]` | [0] `b.sorted_keys()` (READ)<br>[1] `tile_ranges` (WRITE)<br>[2] `count` (WRITE) |
| 1679 | barrier (BufferBarrier) | `tile_ranges` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `tile_ranges` |
| 1689 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset)` `gut_pipeline[buffers.is_unsorted_1]` | [0] `b.sorted_gauss_idx()` (READ)<br>[1] `tile_ranges` (WRITE)<br>[2] `b.xy_vs` (WRITE)<br>[3] `b.inv_cov_vs_opacity` (WRITE)<br>[4] `b.rgb` (WRITE)<br>[5] `b.depths` (WRITE)<br>[6] `b.xyz_ws` (READ)<br>[7] `b.rotations` (READ)<br>[8] `b.scaling_raw` (READ)<br>[9] `b.opacity_raw` (READ)<br>[10] `pixel_state` (READ_WRITE)<br>[11] `pixel_depth` (READ_WRITE)<br>[12] `n_contributors` (READ_WRITE)<br>[13] `pixel_depth_weight` (READ_WRITE)<br>[14] `selection_mask` (READ)<br>[15] `preview_mask` (READ)<br>[16] `selection_colors` (READ)<br>[17] `overlay_flags` (WRITE)<br>[18] `overlay_params` (READ)<br>[19] `transform_indices` (READ)<br>[20] `model_transforms` (READ) |
| 1720 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kFullscreenWordOffset)` `raster_pipeline[buffers.is_unsorted_1]` | [0] `b.sorted_gauss_idx()` (READ)<br>[1] `tile_ranges` (WRITE)<br>[2] `b.xy_vs` (WRITE)<br>[3] `b.inv_cov_vs_opacity` (WRITE)<br>[4] `b.rgb` (WRITE)<br>[5] `b.depths` (WRITE)<br>[6] `pixel_state` (READ_WRITE)<br>[7] `pixel_depth` (READ_WRITE)<br>[8] `n_contributors` (READ_WRITE)<br>[9] `pixel_depth_weight` (READ_WRITE)<br>[10] `selection_mask` (READ)<br>[11] `preview_mask` (READ)<br>[12] `selection_colors` (READ)<br>[13] `overlay_flags` (WRITE)<br>[14] `overlay_params` (READ) |
| 1746 | barrier (BufferBarrier) | `pixel_depth` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_depth_weight` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `pixel_depth`, `pixel_depth_weight` |
| 1751 | dispatch | `pipeline_expected_depth_finalize` | [0] `pixel_depth` (READ_WRITE)<br>[1] `pixel_depth_weight` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1387** overload=`BufferBarrier`
- `b.xy_vs`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.inv_cov_vs_opacity`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rect_tile_space`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.index_buffer_offset`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.primitive_sort_indices`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rgb`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.depths`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.xyz_ws`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rotations`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.scaling_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.opacity_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `selection_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `preview_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `selection_colors`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `overlay_flags`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `overlay_params`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `transform_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `model_transforms`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.sorting_keys_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_keys_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._sorting_histogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`
- `b._sorting_histogram_cumsum`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`

**L1487** overload=`BufferBarrier-built`
- `b.sorting_keys_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_keys_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `tile_ranges`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_state`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_depth`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_depth_weight`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `n_contributors`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._sorting_histogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._sorting_histogram_cumsum`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `batch_* if batched (counts/offsets/descriptors/dispatch/pixel_state/n_contributors)`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1531** overload=`BufferBarrier`
- `b.sorted_keys()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.sorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1555** overload=`BufferBarrier`
- `batch_offsets`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1594** overload=`BufferBarrier`
- `batch_descriptors`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `batch_dispatch`: `COMPUTE_SHADER_WRITE` → `INDIRECT_DISPATCH_READ`
- `pixel_state`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_depth`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `n_contributors`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1629** overload=`BufferBarrier`
- `batch_pixel_state`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `batch_n_contributors`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1664** overload=`BufferBarrier`
- `b.sorted_keys()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.sorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1679** overload=`BufferBarrier`
- `tile_ranges`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L1746** overload=`BufferBarrier`
- `pixel_depth`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_depth_weight`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeSelectionMask`

_Defined at gs_renderer.cpp:1759_  
**Local counts:** barriers=2, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1771 | barrier (pair) | `b.xyz_ws` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.rotations` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.scaling_raw` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.opacity_raw` src=TRANSFER_COMPUTE_SHADER_WRITE; `transform_indices` src=TRANSFER_COMPUTE_SHADER_WRITE; `node_mask` src=TRANSFER_COMPUTE_SHADER_WRITE; `primitives` src=TRANSFER_COMPUTE_SHADER_WRITE; `model_transforms` src=TRANSFER_COMPUTE_SHADER_WRITE; `selection_out` src=TRANSFER_COMPUTE_SHADER_WRITE; `polygon_mask` src=COMPUTE_SHADER_READ_WRITE; `ring_pick_out` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b.xyz_ws`, `b.rotations`, `b.scaling_raw`, `b.opacity_raw`, `transform_indices`, `node_mask`, `primitives`, `model_transforms`, `selection_out`, `polygon_mask`, `ring_pick_out` |
| 1787 | dispatch | `pipeline_selection_mask` | [0] `b.xyz_ws` (READ)<br>[1] `transform_indices` (READ)<br>[2] `node_mask` (READ)<br>[3] `primitives` (READ)<br>[4] `model_transforms` (READ)<br>[5] `b.rotations` (READ)<br>[6] `b.scaling_raw` (READ)<br>[7] `selection_out` (WRITE)<br>[8] `polygon_mask` (READ)<br>[9] `b.opacity_raw` (READ)<br>[10] `ring_pick_out` (WRITE) |
| 1805 | barrier (pair) | `selection_out` src=COMPUTE_SHADER_WRITE; `ring_pick_out` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_READ** | `selection_out`, `ring_pick_out` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1771** overload=`pair`
- `b.xyz_ws`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.rotations`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.scaling_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.opacity_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `transform_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `node_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `primitives`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `model_transforms`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `selection_out`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `polygon_mask`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `ring_pick_out`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1805** overload=`pair`
- `selection_out`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`
- `ring_pick_out`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`

</details>

## Chain: `executeSelectionPolygonRasterize`

_Defined at gs_renderer.cpp:1810_  
**Local counts:** barriers=2, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1816 | barrier (pair) | `polygon_vertices` src=TRANSFER_COMPUTE_SHADER_WRITE; `polygon_mask` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `polygon_vertices`, `polygon_mask` |
| 1823 | dispatch | `pipeline_selection_polygon_rasterize` | [0] `polygon_vertices` (READ)<br>[1] `polygon_mask` (WRITE) |
| 1833 | barrier (pair) | `polygon_mask` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `polygon_mask` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1816** overload=`pair`
- `polygon_vertices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `polygon_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1833** overload=`pair`
- `polygon_mask`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeCumsum`

_Defined at gs_renderer.cpp:1836_  
**Local counts:** barriers=7, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 1858 | dispatch | `pipeline` | [0] `phase_buffers` (UNKNOWN) |
| 1887 | barrier (BufferBarrier-built) | `additional_begin_barriers... (caller-provided)` src=(caller)→dst=(caller); `input_buffer` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `output_buffer` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE; `b._cumsum_blockSums (if uses_level_1)` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE; `b._cumsum_blockSums2 (if uses_level_2)` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE | `additional_begin_barriers... (caller-provided)`, `input_buffer`, `output_buffer`, `b._cumsum_blockSums (if uses_level_1)`, `b._cumsum_blockSums2 (if uses_level_2)` |
| 1915 | barrier (pair) | `b._cumsum_blockSums` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b._cumsum_blockSums` |
| 1928 | barrier (pair) | `output_buffer` src=COMPUTE_SHADER_WRITE; `b._cumsum_blockSums` src=COMPUTE_SHADER_READ_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `output_buffer`, `b._cumsum_blockSums` |
| 1959 | barrier (pair) | `b._cumsum_blockSums` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b._cumsum_blockSums` |
| 1972 | barrier (pair) | `b._cumsum_blockSums` src=COMPUTE_SHADER_READ_WRITE; `b._cumsum_blockSums2` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b._cumsum_blockSums`, `b._cumsum_blockSums2` |
| 1986 | barrier (pair) | `b._cumsum_blockSums2` src=COMPUTE_SHADER_READ_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b._cumsum_blockSums2` |
| 1999 | barrier (pair) | `output_buffer` src=COMPUTE_SHADER_WRITE; `b._cumsum_blockSums` src=COMPUTE_SHADER_READ_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `output_buffer`, `b._cumsum_blockSums` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L1887** overload=`BufferBarrier-built`
- `additional_begin_barriers... (caller-provided)`: `(caller)` → `(caller)`
- `input_buffer`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `output_buffer`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`
- `b._cumsum_blockSums (if uses_level_1)`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`
- `b._cumsum_blockSums2 (if uses_level_2)`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`

**L1915** overload=`pair`
- `b._cumsum_blockSums`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1928** overload=`pair`
- `output_buffer`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._cumsum_blockSums`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1959** overload=`pair`
- `b._cumsum_blockSums`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1972** overload=`pair`
- `b._cumsum_blockSums`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._cumsum_blockSums2`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1986** overload=`pair`
- `b._cumsum_blockSums2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L1999** overload=`pair`
- `output_buffer`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._cumsum_blockSums`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

</details>

## Chain: `executeCalculateIndexBufferOffset`

_Defined at gs_renderer.cpp:2029_  
**Local counts:** barriers=0, dispatches=0, copies=0, fills/clears=0

**No local barriers.** Entirely delegates to `executeCumsum` then `executePrepareTileSort`.

_No direct barrier/dispatch/copy sites (wrapper only)._

## Chain: `executePrepareTileSort`

_Defined at gs_renderer.cpp:2052_  
**Local counts:** barriers=2, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2073 | barrier (pair) | `b.index_buffer_offset` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.index_buffer_offset` |
| 2077 | dispatch | `pipeline_prepare_tile_sort` | [0] `b.index_buffer_offset` (READ)<br>[1] `b.tile_sort_count` (WRITE) |
| 2085 | barrier (pair) | `b.tile_sort_count` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_COMPUTE_SHADER_READ** | `b.tile_sort_count` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2073** overload=`pair`
- `b.index_buffer_offset`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2085** overload=`pair`
- `b.tile_sort_count`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_READ`

</details>

## Chain: `executeSortIndirectCount`

_Defined at gs_renderer.cpp:2089_  
**Local counts:** barriers=0, dispatches=0, copies=0, fills/clears=0

**No local barriers.** Thin wrapper → `executeSortIndirectCountImpl`.

_No direct barrier/dispatch/copy sites (wrapper only)._

## Chain: `executeSortIndirectCountImpl`

_Defined at gs_renderer.cpp:2111_  
**Local counts:** barriers=6, dispatches=4, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2210 | barrier (BufferBarrier) | `globalHistogram` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE; `partitionHistogram` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE; `b.unsorted_keys()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.unsorted_gauss_idx()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `globalHistogram`, `partitionHistogram`, `b.unsorted_keys()`, `b.unsorted_gauss_idx()` |
| 2229 | dispatch | `pipeline_radix_histogram_clear` | [0] `globalHistogram` (WRITE)<br>[1] `partitionHistogram` (WRITE) |
| 2251 | barrier (BufferBarrier-built) | `globalHistogram` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `partitionHistogram` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.unsorted_keys() (if wave_barriers_hoisted)` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.unsorted_gauss_idx() (if wave_barriers_hoisted)` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `globalHistogram`, `partitionHistogram`, `b.unsorted_keys() (if wave_barriers_hoisted)`, `b.unsorted_gauss_idx() (if wave_barriers_hoisted)` |
| 2255 | barrier (BufferBarrier) | `count_buffer` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `dispatch_args_buffer` src=COMPUTE_SHADER_WRITE→dst=INDIRECT_DISPATCH_READ | `count_buffer`, `dispatch_args_buffer` |
| 2271 | barrier (pair) | `b.unsorted_keys()` src=COMPUTE_SHADER_WRITE; `b.unsorted_gauss_idx()` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b.unsorted_keys()`, `b.unsorted_gauss_idx()` |
| 2279 | dispatch-indirect | indirect=`dispatch_args_buffer` offset=`indirect::byteOffset(radix_word_offset)` `pipeline_sorting.upsweep` | [0] `b.unsorted_keys()` (READ)<br>[1] `globalHistogram` (READ_WRITE)<br>[2] `partitionHistogram` (READ_WRITE)<br>[3] `count_buffer` (READ) |
| 2294 | barrier (pair) | `globalHistogram` src=COMPUTE_SHADER_READ_WRITE; `partitionHistogram` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `globalHistogram`, `partitionHistogram` |
| 2302 | dispatch | `pipeline_sorting.spine` | [0] `globalHistogram` (READ_WRITE)<br>[1] `partitionHistogram` (READ_WRITE)<br>[2] `count_buffer` (READ) |
| 2315 | barrier (pair) | `globalHistogram` src=COMPUTE_SHADER_READ_WRITE; `partitionHistogram` src=COMPUTE_SHADER_READ_WRITE | **dst=COMPUTE_SHADER_READ** | `globalHistogram`, `partitionHistogram` |
| 2323 | dispatch-indirect | indirect=`dispatch_args_buffer` offset=`indirect::byteOffset(radix_word_offset)` `pipeline_sorting.downsweep` | [0] `globalHistogram` (READ_WRITE)<br>[1] `partitionHistogram` (READ_WRITE)<br>[2] `b.unsorted_keys()` (READ)<br>[3] `b.unsorted_gauss_idx()` (READ)<br>[4] `b.sorted_keys()` (WRITE)<br>[5] `b.sorted_gauss_idx()` (WRITE)<br>[6] `count_buffer` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2210** overload=`BufferBarrier`
- `globalHistogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`
- `partitionHistogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`
- `b.unsorted_keys()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.unsorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2251** overload=`BufferBarrier-built`
- `globalHistogram`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `partitionHistogram`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.unsorted_keys() (if wave_barriers_hoisted)`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.unsorted_gauss_idx() (if wave_barriers_hoisted)`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2255** overload=`BufferBarrier`
- `count_buffer`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `dispatch_args_buffer`: `COMPUTE_SHADER_WRITE` → `INDIRECT_DISPATCH_READ`

**L2271** overload=`pair`
- `b.unsorted_keys()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.unsorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L2294** overload=`pair`
- `globalHistogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `partitionHistogram`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L2315** overload=`pair`
- `globalHistogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ`
- `partitionHistogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeSortPrimitivesByDepth`

_Defined at gs_renderer.cpp:2344_  
**Local counts:** barriers=8, dispatches=3, copies=1, fills/clears=0

**Nested:** `executeCumsum(visible_flags→visible_prefix)`, `executeSortIndirectCount`→Impl, `recordVisibleCountReadback`.

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2384 | barrier (pair) | `b.tiles_touched` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.tiles_touched` |
| 2388 | dispatch | `pipeline_visible_flags` | [0] `b.tiles_touched` (READ)<br>[1] `b.visible_flags` (WRITE) |
| 2415 | barrier (pair) | `b.visible_prefix` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.visible_prefix` |
| 2419 | dispatch | `pipeline_prepare_visible_sort` | [0] `b.visible_prefix` (READ)<br>[1] `b.visible_count` (WRITE)<br>[2] `b.visible_sort_dispatch_args` (WRITE) |
| 2428 | barrier (pair) | `b.visible_count` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_COMPUTE_SHADER_READ** | `b.visible_count` |
| 2439 | barrier (pair) | `b.primitive_depth_keys` src=COMPUTE_SHADER_WRITE; `b.visible_prefix` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.primitive_depth_keys`, `b.visible_prefix` |
| 2444 | dispatch | `pipeline_compact_visible_primitives` | [0] `b.tiles_touched` (READ)<br>[1] `b.visible_prefix` (READ)<br>[2] `b.primitive_depth_keys` (READ)<br>[3] `*unsorted_keys` (WRITE)<br>[4] `*unsorted_idx` (WRITE) |
| 2460 | barrier (pair) | `*unsorted_keys` src=COMPUTE_SHADER_WRITE; `*unsorted_idx` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `*unsorted_keys`, `*unsorted_idx` |
| 2488 | barrier (pair) | `b.sorted_gauss_idx()` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_READ** | `b.sorted_gauss_idx()` |
| 2490 | barrier (pair) | `sort_indices` src=COMPUTE_SHADER_READ | **dst=TRANSFER_WRITE** | `sort_indices` |
| 2504 | copybuffer | vkCmdCopyBuffer | `        vkCmdCopyBuffer(command_buffer,                         buffers.sorted_gauss_idx().deviceBuffer.buffer,         ` |
| 2507 | barrier (pair) | `sort_indices` src=TRANSFER_WRITE | **dst=COMPUTE_SHADER_READ** | `sort_indices` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2384** overload=`pair`
- `b.tiles_touched`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2415** overload=`pair`
- `b.visible_prefix`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2428** overload=`pair`
- `b.visible_count`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_READ`

**L2439** overload=`pair`
- `b.primitive_depth_keys`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.visible_prefix`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2460** overload=`pair`
- `*unsorted_keys`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `*unsorted_idx`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L2488** overload=`pair`
- `b.sorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `TRANSFER_READ`

**L2490** overload=`pair`
- `sort_indices`: `COMPUTE_SHADER_READ` → `TRANSFER_WRITE`

**L2507** overload=`pair`
- `sort_indices`: `TRANSFER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeApplyDepthOrdering`

_Defined at gs_renderer.cpp:2512_  
**Local counts:** barriers=1, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2525 | barrier (pair) | `b.primitive_sort_indices` src=TRANSFER_WRITE; `b.tiles_touched` src=COMPUTE_SHADER_WRITE; `b.visible_count` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.primitive_sort_indices`, `b.tiles_touched`, `b.visible_count` |
| 2535 | dispatch | `pipeline_apply_depth_ordering` | [0] `b.primitive_sort_indices` (READ)<br>[1] `b.tiles_touched` (READ)<br>[2] `tiles_touched_ordered` (WRITE)<br>[3] `b.visible_count` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2525** overload=`pair`
- `b.primitive_sort_indices`: `TRANSFER_WRITE` → `COMPUTE_SHADER_READ`
- `b.tiles_touched`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.visible_count`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeCullSplats`

_Defined at gs_renderer.cpp:2547_  
**Local counts:** barriers=6, dispatches=2, copies=0, fills/clears=1

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2564 | barrier (pair) | `b.xyz_ws` src=TRANSFER_COMPUTE_SHADER_WRITE; `transform_indices` src=TRANSFER_COMPUTE_SHADER_WRITE; `node_mask` src=TRANSFER_COMPUTE_SHADER_WRITE; `overlay_params` src=TRANSFER_COMPUTE_SHADER_WRITE; `model_transforms` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.xyz_ws`, `transform_indices`, `node_mask`, `overlay_params`, `model_transforms` |
| 2574 | fill-clearDeviceBuffer | fill-clearDeviceBuffer | `auto& survivor_state = clearDeviceBuffer(` |
| 2581 | barrier (pair) | `survivor_state` src=TRANSFER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `survivor_state` |
| 2592 | barrier (pair-conditional) | `lod_indices (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE; `lod_logical_indices (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `lod_indices (if valid)`, `lod_logical_indices (if valid)` |
| 2602 | dispatch | `pipeline_cull_splats` | [0] `b.xyz_ws` (READ)<br>[1] `transform_indices` (READ)<br>[2] `node_mask` (READ)<br>[3] `overlay_params` (READ)<br>[4] `model_transforms` (READ)<br>[5] `lod_indices_binding` (READ)<br>[6] `lod_logical_indices_binding` (READ)<br>[7] `lod_counts_binding` (READ)<br>[8] `survivors` (WRITE)<br>[9] `survivor_state` (WRITE) |
| 2619 | barrier (pair) | `survivor_state` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `survivor_state` |
| 2620 | dispatch | `pipeline_cull_prepare` | [0] `survivor_state` (READ_WRITE)<br>[1] `emit_count` (WRITE) |
| 2629 | barrier (pair) | `survivor_state` src=COMPUTE_SHADER_WRITE | **dst=INDIRECT_DISPATCH_READ** | `survivor_state` |
| 2630 | barrier (pair) | `survivors` src=COMPUTE_SHADER_WRITE; `emit_count` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `survivors`, `emit_count` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2564** overload=`pair`
- `b.xyz_ws`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `transform_indices`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `node_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `overlay_params`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `model_transforms`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2581** overload=`pair`
- `survivor_state`: `TRANSFER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L2592** overload=`pair-conditional`
- `lod_indices (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `lod_logical_indices (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2619** overload=`pair`
- `survivor_state`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L2629** overload=`pair`
- `survivor_state`: `COMPUTE_SHADER_WRITE` → `INDIRECT_DISPATCH_READ`

**L2630** overload=`pair`
- `survivors`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `emit_count`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

</details>

## Chain: `executeProjectionForwardSurvivors`

_Defined at gs_renderer.cpp:2637_  
**Local counts:** barriers=2, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2656 | barrier (pair) | `b.sh0` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.shN` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.rotations` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.scaling_raw` src=TRANSFER_COMPUTE_SHADER_WRITE; `b.opacity_raw` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.sh0`, `b.shN`, `b.rotations`, `b.scaling_raw`, `b.opacity_raw` |
| 2673 | barrier (pair-conditional) | `lod_levels (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE; `lod_weights (if valid)` src=TRANSFER_COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `lod_levels (if valid)`, `lod_weights (if valid)` |
| 2735 | dispatch-indirect | indirect=`b.survivor_state` offset=`indirect::byteOffset(indirect::SurvivorState::kProjectionWordOffset)` `buffers.quant_pool ? pipeline_projection_forward_quant_survivors : pipeline_projection_forward_survivors` | [0] `projection_buffers` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2656** overload=`pair`
- `b.sh0`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.shN`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rotations`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.scaling_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.opacity_raw`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2673** overload=`pair-conditional`
- `lod_levels (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `lod_weights (if valid)`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeSortPrimitivesByDepthVisible`

_Defined at gs_renderer.cpp:2744_  
**Local counts:** barriers=6, dispatches=2, copies=0, fills/clears=0

**Nested:** `recordVisibleCountReadback`, `executeSortIndirectCount`→Impl.

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2775 | barrier (pair) | `b.visible_emit_count` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.visible_emit_count` |
| 2779 | dispatch | `pipeline_prepare_visible_chain` | [0] `b.visible_emit_count` (READ)<br>[1] `b.visible_count` (WRITE)<br>[2] `visible_dispatch` (WRITE)<br>[3] `cumsum_counts` (WRITE) |
| 2789 | barrier (pair) | `b.visible_count` src=COMPUTE_SHADER_WRITE; `cumsum_counts` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_COMPUTE_SHADER_READ** | `b.visible_count`, `cumsum_counts` |
| 2794 | barrier (pair) | `visible_dispatch` src=COMPUTE_SHADER_WRITE | **dst=INDIRECT_DISPATCH_READ** | `visible_dispatch` |
| 2803 | barrier (pair) | `b.unsorted_keys()` src=COMPUTE_SHADER_WRITE; `b.unsorted_gauss_idx()` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `b.unsorted_keys()`, `b.unsorted_gauss_idx()` |
| 2825 | barrier (pair) | `b.sorted_gauss_idx()` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.sorted_gauss_idx()` |
| 2827 | dispatch-indirect | indirect=`visible_dispatch` offset=`indirect::byteOffset(indirect::VisibleChainDispatch::kPerElementWordOffset)` `pipeline_copy_visible_indices` | [0] `b.sorted_gauss_idx()` (READ)<br>[1] `sort_indices` (WRITE)<br>[2] `b.visible_count` (READ) |
| 2837 | barrier (pair) | `sort_indices` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `sort_indices` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2775** overload=`pair`
- `b.visible_emit_count`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2789** overload=`pair`
- `b.visible_count`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_READ`
- `cumsum_counts`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_READ`

**L2794** overload=`pair`
- `visible_dispatch`: `COMPUTE_SHADER_WRITE` → `INDIRECT_DISPATCH_READ`

**L2803** overload=`pair`
- `b.unsorted_keys()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.unsorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L2825** overload=`pair`
- `b.sorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L2837** overload=`pair`
- `sort_indices`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeMacroCoverage`

_Defined at gs_renderer.cpp:2842_  
**Local counts:** barriers=1, dispatches=1, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2858 | barrier (pair) | `b.rect_tile_space` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `b.rect_tile_space` |
| 2861 | dispatch-indirect | indirect=`b.visible_dispatch` offset=`indirect::byteOffset(indirect::VisibleChainDispatch::kPerElementWordOffset)` `pipeline_macro_coverage` | [0] `b.primitive_sort_indices` (READ)<br>[1] `b.rect_tile_space` (READ)<br>[2] `macro_counts` (WRITE)<br>[3] `b.visible_count` (READ)<br>[4] `b.xy_vs` (READ)<br>[5] `b.inv_cov_vs_opacity` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2858** overload=`pair`
- `b.rect_tile_space`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeMacroDepthWaves`

_Defined at gs_renderer.cpp:2876_  
**Local counts:** barriers=7, dispatches=5, copies=0, fills/clears=0

**Nested:** per-wave `executeSortIndirectCountImpl` + `executeCumsum(tile_batch_counts→offsets, additional={{tile_ranges,CSW→CSR}})`.

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 2988 | barrier (BufferBarrier) | `b.xy_vs` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.inv_cov_vs_opacity` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.rect_tile_space` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.index_buffer_offset` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.primitive_sort_indices` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.visible_count` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.rgb` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.depths` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `selection_mask` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `preview_mask` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `selection_colors` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `overlay_params` src=TRANSFER_COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.sorting_keys_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_keys_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b._sorting_histogram` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE; `b._sorting_histogram_cumsum` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_WRITE | `b.xy_vs`, `b.inv_cov_vs_opacity`, `b.rect_tile_space`, `b.index_buffer_offset`, `b.primitive_sort_indices`, `b.visible_count`, `b.rgb`, `b.depths`, `selection_mask`, `preview_mask`, `selection_colors`, `overlay_params`, `b.sorting_keys_1`, `b.sorting_keys_2`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `b._sorting_histogram`, `b._sorting_histogram_cumsum` |
| 3034 | barrier (BufferBarrier) | `b.sorting_keys_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_keys_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_1` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b.sorting_gauss_idx_2` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `tile_ranges` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `batch_counts` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `batch_offsets` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `macro_wave_args` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `partials` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `active_mask` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_state` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_depth` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b._sorting_histogram` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `b._sorting_histogram_cumsum` src=COMPUTE_SHADER_READ_WRITE→dst=COMPUTE_SHADER_READ_WRITE | `b.sorting_keys_1`, `b.sorting_keys_2`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `tile_ranges`, `batch_counts`, `batch_offsets`, `macro_wave_args`, `partials`, `active_mask`, `pixel_state`, `pixel_depth`, `b._sorting_histogram`, `b._sorting_histogram_cumsum` |
| 3079 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kKeygenWordOffset)` `pipeline_generate_macro_keys_wave` | [0] `b.xy_vs` (READ)<br>[1] `b.inv_cov_vs_opacity` (READ)<br>[2] `b.rect_tile_space` (READ)<br>[3] `b.index_buffer_offset` (READ)<br>[4] `b.primitive_sort_indices` (READ)<br>[5] `unsorted_keys` (WRITE)<br>[6] `unsorted_indices` (WRITE)<br>[7] `b.visible_count` (READ)<br>[8] `wave_buffer` (READ_WRITE) |
| 3106 | barrier (BufferBarrier) | `b.sorted_keys()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.sorted_gauss_idx()` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `b.sorted_keys()`, `b.sorted_gauss_idx()` |
| 3112 | dispatch-indirect | indirect=`record` offset=`indirect::byteOffset(indirect::DepthWave::kPerTileWordOffset)` `pipeline_compute_macro_ranges[buffers.is_unsorted_1]` | [0] `b.sorted_keys()` (READ)<br>[1] `tile_ranges` (WRITE)<br>[2] `count` (WRITE)<br>[3] `batch_counts` (WRITE) |
| 3125 | barrier (BufferBarrier) | `batch_offsets` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `batch_offsets` |
| 3128 | dispatch | `pipeline_macro_batch_prepare` | [0] `batch_offsets` (READ)<br>[1] `macro_wave_args` (WRITE) |
| 3133 | barrier (BufferBarrier) | `macro_wave_args` src=COMPUTE_SHADER_WRITE→dst=INDIRECT_DISPATCH_READ | `macro_wave_args` |
| 3184 | barrier (BufferBarrier) | `partials` src=COMPUTE_SHADER_READ→dst=COMPUTE_SHADER_READ_WRITE; `pixel_state` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE; `pixel_depth` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ_WRITE | `partials`, `pixel_state`, `pixel_depth` |
| 3190 | dispatch-indirect | indirect=`macro_wave_args` offset=`indirect::byteOffset(indirect::MacroWaveDispatch::rasterWordOffset(batch_wave))` `raster_pipeline[buffers.is_unsorted_1]` | [0] `raster_bindings` (UNKNOWN) |
| 3197 | barrier (BufferBarrier) | `partials` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `active_mask` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `partials`, `active_mask` |
| 3201 | dispatch-indirect | indirect=`macro_wave_args` offset=`indirect::byteOffset(indirect::MacroWaveDispatch::composeWordOffset(batch_wave))` `compose_pipeline[buffers.is_unsorted_1]` | [0] `compose_bindings` (READ) |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L2988** overload=`BufferBarrier`
- `b.xy_vs`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.inv_cov_vs_opacity`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rect_tile_space`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.index_buffer_offset`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.primitive_sort_indices`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.visible_count`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.rgb`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.depths`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `selection_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `preview_mask`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `selection_colors`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `overlay_params`: `TRANSFER_COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.sorting_keys_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_keys_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._sorting_histogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`
- `b._sorting_histogram_cumsum`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_WRITE`

**L3034** overload=`BufferBarrier`
- `b.sorting_keys_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_keys_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_1`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b.sorting_gauss_idx_2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `tile_ranges`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `batch_counts`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `batch_offsets`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `macro_wave_args`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `partials`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `active_mask`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_state`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_depth`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._sorting_histogram`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `b._sorting_histogram_cumsum`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L3106** overload=`BufferBarrier`
- `b.sorted_keys()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.sorted_gauss_idx()`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L3125** overload=`BufferBarrier`
- `batch_offsets`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L3133** overload=`BufferBarrier`
- `macro_wave_args`: `COMPUTE_SHADER_WRITE` → `INDIRECT_DISPATCH_READ`

**L3184** overload=`BufferBarrier`
- `partials`: `COMPUTE_SHADER_READ` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_state`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `pixel_depth`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L3197** overload=`BufferBarrier`
- `partials`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `active_mask`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

</details>

## Chain: `executeCalculateIndexBufferOffsetVisible`

_Defined at gs_renderer.cpp:3212_  
**Local counts:** barriers=7, dispatches=6, copies=0, fills/clears=0

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 3248 | barrier (pair) | `input` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `input` |
| 3256 | dispatch-indirect | indirect=`dispatch` offset=`indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset)` `pipeline_cumsum_indirect.block_scan` | [0] `input` (READ)<br>[1] `output` (WRITE)<br>[2] `block_sums` (READ_WRITE)<br>[3] `counts` (READ_WRITE) |
| 3262 | barrier (pair) | `block_sums` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `block_sums` |
| 3264 | dispatch-indirect | indirect=`dispatch` offset=`indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset)` `pipeline_cumsum_indirect.block_scan` | [0] `block_sums` (READ)<br>[1] `block_sums` (WRITE)<br>[2] `block_sums2` (READ_WRITE)<br>[3] `counts` (READ_WRITE) |
| 3270 | barrier (pair) | `block_sums` src=COMPUTE_SHADER_READ_WRITE; `block_sums2` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `block_sums`, `block_sums2` |
| 3276 | dispatch | `pipeline_cumsum_indirect.scan_block_sums` | [0] `block_sums` (READ)<br>[1] `block_sums` (WRITE)<br>[2] `block_sums2` (READ_WRITE)<br>[3] `counts` (READ_WRITE) |
| 3281 | barrier (pair) | `block_sums2` src=COMPUTE_SHADER_READ_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `block_sums2` |
| 3284 | dispatch-indirect | indirect=`dispatch` offset=`indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel1WordOffset)` `pipeline_cumsum_indirect.add_block_offsets` | [0] `block_sums` (READ)<br>[1] `block_sums` (WRITE)<br>[2] `block_sums2` (READ_WRITE)<br>[3] `counts` (READ_WRITE) |
| 3290 | barrier (pair) | `output` src=COMPUTE_SHADER_WRITE; `block_sums` src=COMPUTE_SHADER_READ_WRITE | **dst=COMPUTE_SHADER_READ_WRITE** | `output`, `block_sums` |
| 3296 | dispatch-indirect | indirect=`dispatch` offset=`indirect::byteOffset(indirect::VisibleChainDispatch::kCumsumLevel0WordOffset)` `pipeline_cumsum_indirect.add_block_offsets` | [0] `input` (READ)<br>[1] `output` (WRITE)<br>[2] `block_sums` (READ_WRITE)<br>[3] `counts` (READ_WRITE) |
| 3320 | barrier (pair) | `output` src=COMPUTE_SHADER_WRITE | **dst=COMPUTE_SHADER_READ** | `output` |
| 3321 | dispatch | `pipeline_prepare_tile_sort_visible` | [0] `output` (READ)<br>[1] `b.tile_sort_count` (WRITE)<br>[2] `b.visible_count` (READ) |
| 3332 | barrier (pair) | `b.tile_sort_count` src=COMPUTE_SHADER_WRITE | **dst=TRANSFER_COMPUTE_SHADER_READ** | `b.tile_sort_count` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L3248** overload=`pair`
- `input`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L3262** overload=`pair`
- `block_sums`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L3270** overload=`pair`
- `block_sums`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `block_sums2`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L3281** overload=`pair`
- `block_sums2`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L3290** overload=`pair`
- `output`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ_WRITE`
- `block_sums`: `COMPUTE_SHADER_READ_WRITE` → `COMPUTE_SHADER_READ_WRITE`

**L3320** overload=`pair`
- `output`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L3332** overload=`pair`
- `b.tile_sort_count`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_READ`

</details>

## Chain: `executeWavePartition`

_Defined at gs_renderer.cpp:3336_  
**Local counts:** barriers=2, dispatches=1, copies=0, fills/clears=0

**Nested:** ends with `recordInstanceCountReadback` (copies + barrier 360).

| line | kind | pipeline or masks | buffers |
|------|------|-------------------|---------|
| 3387 | barrier (BufferBarrier) | `b.index_buffer_offset` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ; `b.tile_sort_count` src=COMPUTE_SHADER_WRITE→dst=COMPUTE_SHADER_READ | `b.index_buffer_offset`, `b.tile_sort_count` |
| 3396 | dispatch | `visible_bounded ? pipeline_wave_partition_visible : pipeline_wave_partition` | [0] `bindings` (READ) |
| 3413 | barrier (BufferBarrier-built) | `wave_dispatch` src=COMPUTE_SHADER_WRITE→dst=TRANSFER_COMPUTE_SHADER_INDIRECT_READ; `b.tile_sort_count` src=COMPUTE_SHADER_WRITE→dst=TRANSFER_COMPUTE_SHADER_READ; `predicates (if conditional rendering)` src=COMPUTE_SHADER_WRITE→dst=CONDITIONAL_RENDERING_READ | `wave_dispatch`, `b.tile_sort_count`, `predicates (if conditional rendering)` |

<details><summary>Barrier detail (src→dst per buffer)</summary>

**L3387** overload=`BufferBarrier`
- `b.index_buffer_offset`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`
- `b.tile_sort_count`: `COMPUTE_SHADER_WRITE` → `COMPUTE_SHADER_READ`

**L3413** overload=`BufferBarrier-built`
- `wave_dispatch`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_INDIRECT_READ`
- `b.tile_sort_count`: `COMPUTE_SHADER_WRITE` → `TRANSFER_COMPUTE_SHADER_READ`
- `predicates (if conditional rendering)`: `COMPUTE_SHADER_WRITE` → `CONDITIONAL_RENDERING_READ`

</details>

## Complete index of all 82 `bufferMemoryBarrier` call sites

| # | line | function | overload | #bufs | dst / notes |
|---|------|----------|----------|-------|-------------|
| 1 | 360 | `recordInstanceCountReadback` | pair | 1 | HOST_READ |
| 2 | 491 | `synchronizeTileInstanceGate` | pair | 1 | HOST_READ |
| 3 | 776 | `recordVisibleCountReadback` | pair | 1 | HOST_READ |
| 4 | 795 | `recordLodSelectionReadback` | pair | 4 | TRANSFER_READ |
| 5 | 820 | `recordLodSelectionReadback` | pair | 1 | HOST_READ |
| 6 | 1023 | `executeMapLodIndices` | pair | 3 | COMPUTE_SHADER_READ_WRITE |
| 7 | 1040 | `executeMapLodIndices` | pair | 1 | COMPUTE_SHADER_READ |
| 8 | 1077 | `executeSelectLodThreshold` | pair | 12 | COMPUTE_SHADER_READ_WRITE |
| 9 | 1112 | `executeSelectLodThreshold` | pair | 5 | COMPUTE_SHADER_READ |
| 10 | 1134 | `executeSelectLodThreshold` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 11 | 1169 | `executeProjectionForward` | pair | 10 | COMPUTE_SHADER_READ |
| 12 | 1191 | `executeProjectionForward` | pair | 1 | TRANSFER_COMPUTE_SHADER_WRITE |
| 13 | 1197 | `executeProjectionForward` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 14 | 1218 | `executeProjectionForward` | pair-conditional | 4 | COMPUTE_SHADER_READ |
| 15 | 1387 | `executeLegacyDepthWaves` | BufferBarrier | 24 | per-entry |
| 16 | 1487 | `executeLegacyDepthWaves` | BufferBarrier-built | 12 | per-entry |
| 17 | 1531 | `executeLegacyDepthWaves` | BufferBarrier | 2 | per-entry |
| 18 | 1555 | `executeLegacyDepthWaves` | BufferBarrier | 1 | per-entry |
| 19 | 1594 | `executeLegacyDepthWaves` | BufferBarrier | 5 | per-entry |
| 20 | 1629 | `executeLegacyDepthWaves` | BufferBarrier | 2 | per-entry |
| 21 | 1664 | `executeLegacyDepthWaves` | BufferBarrier | 2 | per-entry |
| 22 | 1679 | `executeLegacyDepthWaves` | BufferBarrier | 1 | per-entry |
| 23 | 1746 | `executeLegacyDepthWaves` | BufferBarrier | 2 | per-entry |
| 24 | 1771 | `executeSelectionMask` | pair | 11 | COMPUTE_SHADER_READ_WRITE |
| 25 | 1805 | `executeSelectionMask` | pair | 2 | TRANSFER_READ |
| 26 | 1816 | `executeSelectionPolygonRasterize` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 27 | 1833 | `executeSelectionPolygonRasterize` | pair | 1 | COMPUTE_SHADER_READ |
| 28 | 1887 | `executeCumsum` | BufferBarrier-built | 5 | per-entry |
| 29 | 1915 | `executeCumsum` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 30 | 1928 | `executeCumsum` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 31 | 1959 | `executeCumsum` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 32 | 1972 | `executeCumsum` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 33 | 1986 | `executeCumsum` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 34 | 1999 | `executeCumsum` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 35 | 2073 | `executePrepareTileSort` | pair | 1 | COMPUTE_SHADER_READ |
| 36 | 2085 | `executePrepareTileSort` | pair | 1 | TRANSFER_COMPUTE_SHADER_READ |
| 37 | 2210 | `executeSortIndirectCountImpl` | BufferBarrier | 4 | per-entry |
| 38 | 2251 | `executeSortIndirectCountImpl` | BufferBarrier-built | 4 | per-entry |
| 39 | 2255 | `executeSortIndirectCountImpl` | BufferBarrier | 2 | per-entry |
| 40 | 2271 | `executeSortIndirectCountImpl` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 41 | 2294 | `executeSortIndirectCountImpl` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 42 | 2315 | `executeSortIndirectCountImpl` | pair | 2 | COMPUTE_SHADER_READ |
| 43 | 2384 | `executeSortPrimitivesByDepth` | pair | 1 | COMPUTE_SHADER_READ |
| 44 | 2415 | `executeSortPrimitivesByDepth` | pair | 1 | COMPUTE_SHADER_READ |
| 45 | 2428 | `executeSortPrimitivesByDepth` | pair | 1 | TRANSFER_COMPUTE_SHADER_READ |
| 46 | 2439 | `executeSortPrimitivesByDepth` | pair | 2 | COMPUTE_SHADER_READ |
| 47 | 2460 | `executeSortPrimitivesByDepth` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 48 | 2488 | `executeSortPrimitivesByDepth` | pair | 1 | TRANSFER_READ |
| 49 | 2490 | `executeSortPrimitivesByDepth` | pair | 1 | TRANSFER_WRITE |
| 50 | 2507 | `executeSortPrimitivesByDepth` | pair | 1 | COMPUTE_SHADER_READ |
| 51 | 2525 | `executeApplyDepthOrdering` | pair | 3 | COMPUTE_SHADER_READ |
| 52 | 2564 | `executeCullSplats` | pair | 5 | COMPUTE_SHADER_READ |
| 53 | 2581 | `executeCullSplats` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 54 | 2592 | `executeCullSplats` | pair-conditional | 2 | COMPUTE_SHADER_READ |
| 55 | 2619 | `executeCullSplats` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 56 | 2629 | `executeCullSplats` | pair | 1 | INDIRECT_DISPATCH_READ |
| 57 | 2630 | `executeCullSplats` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 58 | 2656 | `executeProjectionForwardSurvivors` | pair | 5 | COMPUTE_SHADER_READ |
| 59 | 2673 | `executeProjectionForwardSurvivors` | pair-conditional | 2 | COMPUTE_SHADER_READ |
| 60 | 2775 | `executeSortPrimitivesByDepthVisible` | pair | 1 | COMPUTE_SHADER_READ |
| 61 | 2789 | `executeSortPrimitivesByDepthVisible` | pair | 2 | TRANSFER_COMPUTE_SHADER_READ |
| 62 | 2794 | `executeSortPrimitivesByDepthVisible` | pair | 1 | INDIRECT_DISPATCH_READ |
| 63 | 2803 | `executeSortPrimitivesByDepthVisible` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 64 | 2825 | `executeSortPrimitivesByDepthVisible` | pair | 1 | COMPUTE_SHADER_READ |
| 65 | 2837 | `executeSortPrimitivesByDepthVisible` | pair | 1 | COMPUTE_SHADER_READ |
| 66 | 2858 | `executeMacroCoverage` | pair | 1 | COMPUTE_SHADER_READ |
| 67 | 2988 | `executeMacroDepthWaves` | BufferBarrier | 18 | per-entry |
| 68 | 3034 | `executeMacroDepthWaves` | BufferBarrier | 14 | per-entry |
| 69 | 3106 | `executeMacroDepthWaves` | BufferBarrier | 2 | per-entry |
| 70 | 3125 | `executeMacroDepthWaves` | BufferBarrier | 1 | per-entry |
| 71 | 3133 | `executeMacroDepthWaves` | BufferBarrier | 1 | per-entry |
| 72 | 3184 | `executeMacroDepthWaves` | BufferBarrier | 3 | per-entry |
| 73 | 3197 | `executeMacroDepthWaves` | BufferBarrier | 2 | per-entry |
| 74 | 3248 | `executeCalculateIndexBufferOffsetVisible` | pair | 1 | COMPUTE_SHADER_READ |
| 75 | 3262 | `executeCalculateIndexBufferOffsetVisible` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 76 | 3270 | `executeCalculateIndexBufferOffsetVisible` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 77 | 3281 | `executeCalculateIndexBufferOffsetVisible` | pair | 1 | COMPUTE_SHADER_READ_WRITE |
| 78 | 3290 | `executeCalculateIndexBufferOffsetVisible` | pair | 2 | COMPUTE_SHADER_READ_WRITE |
| 79 | 3320 | `executeCalculateIndexBufferOffsetVisible` | pair | 1 | COMPUTE_SHADER_READ |
| 80 | 3332 | `executeCalculateIndexBufferOffsetVisible` | pair | 1 | TRANSFER_COMPUTE_SHADER_READ |
| 81 | 3387 | `executeWavePartition` | BufferBarrier | 2 | per-entry |
| 82 | 3413 | `executeWavePartition` | BufferBarrier-built | 3 | per-entry |

## Complete index of all `executeCompute*` call sites

| # | line | kind | function | pipeline | #bufs |
|---|------|------|----------|----------|-------|
| 1 | 1030 | dispatch | `executeMapLodIndices` | `pipeline_lod_map_indices` | 3 |
| 2 | 1093 | dispatch | `executeSelectLodThreshold` | `pipeline_lod_select_threshold` | 12 |
| 3 | 1137 | dispatch | `executeSelectLodThreshold` | `pipeline_lod_compact_touch` | 4 |
| 4 | 1266 | dispatch | `executeProjectionForward` | `buffers.quant_pool ? (use_gut_projection ? pipeline_projection_forward_quant_3dgut : pipeline_projection_forward_quant) : (use_gut_projection ? pipeline_projection_forward_3dgut : pipeline_projection_forward)` | 1 |
| 5 | 1506 | dispatch-indirect | `executeLegacyDepthWaves` | `pipeline_generate_keys_wave` | 8 |
| 6 | 1539 | dispatch-indirect | `executeLegacyDepthWaves` | `pipeline_compute_tile_ranges_and_batch_counts [buffers.is_unsorted_1]` | 4 |
| 7 | 1558 | dispatch-indirect | `executeLegacyDepthWaves` | `pipeline_tile_batch_descriptors` | 4 |
| 8 | 1572 | dispatch-indirect | `executeLegacyDepthWaves` | `light_pipeline[buffers.is_unsorted_1]` | 15 |
| 9 | 1621 | dispatch-indirect | `executeLegacyDepthWaves` | `batch_pipeline[buffers.is_unsorted_1]` | 1 |
| 10 | 1655 | dispatch-indirect | `executeLegacyDepthWaves` | `overlays_active ? pipeline_compose_tile_batches : pipeline_compose_tile_batches_plain` | 1 |
| 11 | 1672 | dispatch-indirect | `executeLegacyDepthWaves` | `pipeline_compute_tile_ranges[buffers.is_unsorted_1]` | 3 |
| 12 | 1689 | dispatch-indirect | `executeLegacyDepthWaves` | `gut_pipeline[buffers.is_unsorted_1]` | 21 |
| 13 | 1720 | dispatch-indirect | `executeLegacyDepthWaves` | `raster_pipeline[buffers.is_unsorted_1]` | 15 |
| 14 | 1751 | dispatch | `executeLegacyDepthWaves` | `pipeline_expected_depth_finalize` | 2 |
| 15 | 1787 | dispatch | `executeSelectionMask` | `pipeline_selection_mask` | 11 |
| 16 | 1823 | dispatch | `executeSelectionPolygonRasterize` | `pipeline_selection_polygon_rasterize` | 2 |
| 17 | 1858 | dispatch | `executeCumsum` | `pipeline` | 1 |
| 18 | 2077 | dispatch | `executePrepareTileSort` | `pipeline_prepare_tile_sort` | 2 |
| 19 | 2229 | dispatch | `executeSortIndirectCountImpl` | `pipeline_radix_histogram_clear` | 2 |
| 20 | 2279 | dispatch-indirect | `executeSortIndirectCountImpl` | `pipeline_sorting.upsweep` | 4 |
| 21 | 2302 | dispatch | `executeSortIndirectCountImpl` | `pipeline_sorting.spine` | 3 |
| 22 | 2323 | dispatch-indirect | `executeSortIndirectCountImpl` | `pipeline_sorting.downsweep` | 7 |
| 23 | 2388 | dispatch | `executeSortPrimitivesByDepth` | `pipeline_visible_flags` | 2 |
| 24 | 2419 | dispatch | `executeSortPrimitivesByDepth` | `pipeline_prepare_visible_sort` | 3 |
| 25 | 2444 | dispatch | `executeSortPrimitivesByDepth` | `pipeline_compact_visible_primitives` | 5 |
| 26 | 2535 | dispatch | `executeApplyDepthOrdering` | `pipeline_apply_depth_ordering` | 4 |
| 27 | 2602 | dispatch | `executeCullSplats` | `pipeline_cull_splats` | 10 |
| 28 | 2620 | dispatch | `executeCullSplats` | `pipeline_cull_prepare` | 2 |
| 29 | 2735 | dispatch-indirect | `executeProjectionForwardSurvivors` | `buffers.quant_pool ? pipeline_projection_forward_quant_survivors : pipeline_projection_forward_survivors` | 1 |
| 30 | 2779 | dispatch | `executeSortPrimitivesByDepthVisible` | `pipeline_prepare_visible_chain` | 4 |
| 31 | 2827 | dispatch-indirect | `executeSortPrimitivesByDepthVisible` | `pipeline_copy_visible_indices` | 3 |
| 32 | 2861 | dispatch-indirect | `executeMacroCoverage` | `pipeline_macro_coverage` | 6 |
| 33 | 3079 | dispatch-indirect | `executeMacroDepthWaves` | `pipeline_generate_macro_keys_wave` | 9 |
| 34 | 3112 | dispatch-indirect | `executeMacroDepthWaves` | `pipeline_compute_macro_ranges[buffers.is_unsorted_1]` | 4 |
| 35 | 3128 | dispatch | `executeMacroDepthWaves` | `pipeline_macro_batch_prepare` | 2 |
| 36 | 3190 | dispatch-indirect | `executeMacroDepthWaves` | `raster_pipeline[buffers.is_unsorted_1]` | 1 |
| 37 | 3201 | dispatch-indirect | `executeMacroDepthWaves` | `compose_pipeline[buffers.is_unsorted_1]` | 1 |
| 38 | 3256 | dispatch-indirect | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.block_scan` | 4 |
| 39 | 3264 | dispatch-indirect | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.block_scan` | 4 |
| 40 | 3276 | dispatch | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.scan_block_sums` | 4 |
| 41 | 3284 | dispatch-indirect | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.add_block_offsets` | 4 |
| 42 | 3296 | dispatch-indirect | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.add_block_offsets` | 4 |
| 43 | 3321 | dispatch | `executeCalculateIndexBufferOffsetVisible` | `pipeline_prepare_tile_sort_visible` | 3 |
| 44 | 3396 | dispatch | `executeWavePartition` | `visible_bounded ? pipeline_wave_partition_visible : pipeline_wave_partition` | 1 |

## vkCmd buffer ops + clearDeviceBuffer (implicit fill)

| line | function | op | notes |
|------|----------|-----|-------|
| 329 | `recordInstanceCountReadback` | `vkCmdCopyBuffer` | see chain |
| 345 | `recordInstanceCountReadback` | `vkCmdCopyBuffer` | see chain |
| 355 | `recordInstanceCountReadback` | `vkCmdUpdateBuffer` | see chain |
| 486 | `synchronizeTileInstanceGate` | `vkCmdCopyBuffer` | see chain |
| 771 | `recordVisibleCountReadback` | `vkCmdCopyBuffer` | see chain |
| 812 | `recordLodSelectionReadback` | `vkCmdCopyBuffer` | see chain |
| 1194 | `executeProjectionForward` | `vkCmdFillBuffer` | see chain |
| 2504 | `executeSortPrimitivesByDepth` | `vkCmdCopyBuffer` | see chain |
| 1064 | `executeSelectLodThreshold` | `clearDeviceBuffer→vkCmdFillBuffer` | zero-fill (impl in buffer.cpp:404) |
| 1072 | `executeSelectLodThreshold` | `clearDeviceBuffer→vkCmdFillBuffer` | zero-fill (impl in buffer.cpp:404) |
| 1123 | `executeSelectLodThreshold` | `clearDeviceBuffer→vkCmdFillBuffer` | zero-fill (impl in buffer.cpp:404) |
| 2574 | `executeCullSplats` | `clearDeviceBuffer→vkCmdFillBuffer` | zero-fill (impl in buffer.cpp:404) |

## Barrier vs following dispatch mismatches

Compared each barrier that is immediately followed (no intervening barrier) by a dispatch in the **same function**. Extra = in barrier but not dispatch bindings; Missing = in dispatch but not barrier.

Note: many “mismatches” are intentional (e.g. barrier only the hazard producers; dispatch also binds long-lived read-only buffers already ordered earlier). Flagged for migration review.

| barrier L | dispatch L | function | pipeline | extra in barrier | missing from barrier |
|-----------|------------|----------|----------|------------------|----------------------|
| 1134 | 1137 | `executeSelectLodThreshold` | `pipeline_lod_compact_touch` | — | `compact_misses`, `compact_protected` |
| 1218 | 1266 | `executeProjectionForward` | `buffers.quant_pool ? (use_gut_projection ? pipeline_projecti` | `lod_indices`, `lod_levels`, `lod_logical_indices`, `lod_weights` | `projection_buffers` |
| 1487 | 1506 | `executeLegacyDepthWaves` | `pipeline_generate_keys_wave` | `b._sorting_histogram`, `b._sorting_histogram_cumsum`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `b.sorting_keys_1`, `b.sorting_keys_2`, `batch_`, `n_contributors`, `pixel_depth`, `pixel_depth_weight`, `pixel_state`, `tile_ranges` | `b.index_buffer_offset`, `b.inv_cov_vs_opacity`, `b.primitive_sort_indices`, `b.rect_tile_space`, `b.xy_vs`, `record`, `unsorted_indices`, `unsorted_keys`, `wave_buffer` |
| 1531 | 1539 | `executeLegacyDepthWaves` | `pipeline_compute_tile_ranges_and_batch_counts [buffers.is_un` | `b.sorted_gauss_idx` | `batch_counts`, `count`, `record`, `tile_ranges` |
| 1555 | 1558 | `executeLegacyDepthWaves` | `pipeline_tile_batch_descriptors` | — | `batch_descriptors`, `batch_dispatch`, `record`, `tile_ranges` |
| 1594 | 1621 | `executeLegacyDepthWaves` | `batch_pipeline[buffers.is_unsorted_1]` | `batch_descriptors`, `n_contributors`, `pixel_depth`, `pixel_state` | `batch_bindings` |
| 1629 | 1655 | `executeLegacyDepthWaves` | `overlays_active ? pipeline_compose_tile_batches : pipeline_c` | `batch_n_contributors`, `batch_pixel_state` | `compose_bindings`, `record` |
| 1664 | 1672 | `executeLegacyDepthWaves` | `pipeline_compute_tile_ranges[buffers.is_unsorted_1]` | `b.sorted_gauss_idx` | `count`, `record`, `tile_ranges` |
| 1679 | 1689 | `executeLegacyDepthWaves` | `gut_pipeline[buffers.is_unsorted_1]` | — | `b.depths`, `b.inv_cov_vs_opacity`, `b.opacity_raw`, `b.rgb`, `b.rotations`, `b.scaling_raw`, `b.sorted_gauss_idx`, `b.xy_vs`, `b.xyz_ws`, `model_transforms`, `n_contributors`, `overlay_flags`, `overlay_params`, `pixel_depth`, `pixel_depth_weight`, `pixel_state`, `preview_mask`, `record`, `selection_colors`, `selection_mask`, `transform_indices` |
| 2073 | 2077 | `executePrepareTileSort` | `pipeline_prepare_tile_sort` | — | `b.tile_sort_count` |
| 2210 | 2229 | `executeSortIndirectCountImpl` | `pipeline_radix_histogram_clear` | `b.unsorted_gauss_idx`, `b.unsorted_keys` | — |
| 2271 | 2279 | `executeSortIndirectCountImpl` | `pipeline_sorting.upsweep` | `b.unsorted_gauss_idx` | `count_buffer`, `dispatch_args_buffer`, `globalHistogram`, `partitionHistogram` |
| 2294 | 2302 | `executeSortIndirectCountImpl` | `pipeline_sorting.spine` | — | `count_buffer` |
| 2315 | 2323 | `executeSortIndirectCountImpl` | `pipeline_sorting.downsweep` | — | `b.sorted_gauss_idx`, `b.sorted_keys`, `b.unsorted_gauss_idx`, `b.unsorted_keys`, `count_buffer`, `dispatch_args_buffer` |
| 2384 | 2388 | `executeSortPrimitivesByDepth` | `pipeline_visible_flags` | — | `b.visible_flags` |
| 2415 | 2419 | `executeSortPrimitivesByDepth` | `pipeline_prepare_visible_sort` | — | `b.visible_count`, `b.visible_sort_dispatch_args` |
| 2439 | 2444 | `executeSortPrimitivesByDepth` | `pipeline_compact_visible_primitives` | — | `b.tiles_touched`, `unsorted_idx`, `unsorted_keys` |
| 2525 | 2535 | `executeApplyDepthOrdering` | `pipeline_apply_depth_ordering` | — | `tiles_touched_ordered` |
| 2592 | 2602 | `executeCullSplats` | `pipeline_cull_splats` | `lod_indices`, `lod_logical_indices` | `b.xyz_ws`, `lod_counts_binding`, `lod_indices_binding`, `lod_logical_indices_binding`, `model_transforms`, `node_mask`, `overlay_params`, `survivor_state`, `survivors`, `transform_indices` |
| 2619 | 2620 | `executeCullSplats` | `pipeline_cull_prepare` | — | `emit_count` |
| 2673 | 2735 | `executeProjectionForwardSurvivors` | `buffers.quant_pool ? pipeline_projection_forward_quant_survi` | `lod_levels`, `lod_weights` | `b.survivor_state`, `projection_buffers` |
| 2775 | 2779 | `executeSortPrimitivesByDepthVisible` | `pipeline_prepare_visible_chain` | — | `b.visible_count`, `cumsum_counts`, `visible_dispatch` |
| 2825 | 2827 | `executeSortPrimitivesByDepthVisible` | `pipeline_copy_visible_indices` | — | `b.visible_count`, `sort_indices`, `visible_dispatch` |
| 2858 | 2861 | `executeMacroCoverage` | `pipeline_macro_coverage` | — | `b.inv_cov_vs_opacity`, `b.primitive_sort_indices`, `b.visible_count`, `b.visible_dispatch`, `b.xy_vs`, `macro_counts` |
| 3034 | 3079 | `executeMacroDepthWaves` | `pipeline_generate_macro_keys_wave` | `active_mask`, `b._sorting_histogram`, `b._sorting_histogram_cumsum`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `b.sorting_keys_1`, `b.sorting_keys_2`, `batch_counts`, `batch_offsets`, `macro_wave_args`, `partials`, `pixel_depth`, `pixel_state`, `tile_ranges` | `b.index_buffer_offset`, `b.inv_cov_vs_opacity`, `b.primitive_sort_indices`, `b.rect_tile_space`, `b.visible_count`, `b.xy_vs`, `record`, `unsorted_indices`, `unsorted_keys`, `wave_buffer` |
| 3106 | 3112 | `executeMacroDepthWaves` | `pipeline_compute_macro_ranges[buffers.is_unsorted_1]` | `b.sorted_gauss_idx` | `batch_counts`, `count`, `record`, `tile_ranges` |
| 3125 | 3128 | `executeMacroDepthWaves` | `pipeline_macro_batch_prepare` | — | `macro_wave_args` |
| 3184 | 3190 | `executeMacroDepthWaves` | `raster_pipeline[buffers.is_unsorted_1]` | `partials`, `pixel_depth`, `pixel_state` | `macro_wave_args`, `raster_bindings` |
| 3197 | 3201 | `executeMacroDepthWaves` | `compose_pipeline[buffers.is_unsorted_1]` | `active_mask`, `partials` | `compose_bindings`, `macro_wave_args` |
| 3248 | 3256 | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.block_scan` | — | `block_sums`, `counts`, `dispatch`, `output` |
| 3262 | 3264 | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.block_scan` | — | `block_sums2`, `counts`, `dispatch` |
| 3270 | 3276 | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.scan_block_sums` | — | `counts` |
| 3281 | 3284 | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.add_block_offsets` | — | `block_sums`, `counts`, `dispatch` |
| 3290 | 3296 | `executeCalculateIndexBufferOffsetVisible` | `pipeline_cumsum_indirect.add_block_offsets` | — | `counts`, `dispatch`, `input` |
| 3320 | 3321 | `executeCalculateIndexBufferOffsetVisible` | `pipeline_prepare_tile_sort_visible` | — | `b.tile_sort_count`, `b.visible_count` |
| 3387 | 3396 | `executeWavePartition` | `visible_bounded ? pipeline_wave_partition_visible : pipeline` | `b.index_buffer_offset`, `b.tile_sort_count` | `bindings` |

### Hand-noted structural mismatches (high signal)

1. **executeSelectLodThreshold L1134 → compact dispatch L1137:** barrier has `chunk_touch`, `compact_counts` only; dispatch also binds `compact_protected`, `compact_misses` (freshly resized) — **missing from barrier**.
2. **executeProjectionForward L1169:** barrier covers 10 **input** attribute buffers only; projection also writes many outputs — outputs ordered via fill on `primitive_depth_keys` (L1191–1197) and resize, not the pre-dispatch barrier.
3. **executeProjectionForward L1218 LOD barrier:** conditional LOD inputs only; does not list outputs.
4. **executeLegacyDepthWaves L1387 mega-barrier (24 entries):** far wider than any single following dispatch; hoists wave-loop + sort workspace hazards. **Extra-heavy by design**.
5. **executeLegacyDepthWaves light raster L1572:** no dedicated pre-barrier after descriptors beyond L1555; relies on cumsum/sort edges.
6. **executeSortIndirectCountImpl L2210:** when `!wave_barriers_hoisted`, barriers histogram + unsorted; clear dispatch only uses histograms — **extra unsorted_***. When hoisted, L2210 skipped; L2251 folds unsorted.
7. **executeSortIndirectCountImpl spine L2302:** barrier L2294 lists global+partition histograms; spine also binds `count_buffer` — **missing count_buffer** (ordered at L2255 or hoisted).
8. **executeSortIndirectCountImpl downsweep L2323:** barrier L2315 only histograms; downsweep binds unsorted/sorted keys/idx + count — **missing keys/idx/count**.
9. **executeSortPrimitivesByDepth compact L2444:** barrier L2439 has primitive_depth_keys + visible_prefix; dispatch also reads tiles_touched and writes unsorted — **missing tiles_touched, unsorted_*** (tiles_touched ordered earlier at L2384).
10. **executeCullSplats L2564:** 5 inputs; dispatch also binds lod_* + survivors + survivor_state — lod optional at L2592; survivors/state ordered by L2581 clear.
11. **executeProjectionForwardSurvivors L2656:** only SH/rot/scale/opacity; dispatch binds xyz, survivors, survivor_state, emit_count, many outputs — **large missing set** (prior cull barriers expected).
12. **executeMacroCoverage L2858:** only `rect_tile_space`; dispatch also needs primitive_sort_indices, macro_counts, visible_count, xy_vs, inv_cov — **missing most**.
13. **executeMacroDepthWaves L2988 mega-barrier (18):** same hoist pattern as legacy.
14. **executeCalculateIndexBufferOffsetVisible:** level barriers often only the producer of the previous phase; count+dispatch args assumed ordered by prepare_visible_chain.
15. **executeWavePartition L3387:** index_buffer_offset + tile_sort_count; dispatch also writes wave_dispatch + predicates and may bind visible_count — **missing outputs and optional visible_count**.
16. **recordLodSelectionReadback L795:** barriers sources for TRANSFER_READ before copy; not a dispatch pairing.

## External / other-file barrier usage

### Callers of `VulkanGSPipeline::bufferMemoryBarrier`

Repo-wide grep of `bufferMemoryBarrier(`:

| File | Role |
|------|------|
| `gs_pipeline.h` | declarations (pair + BufferBarrier overloads) |
| `gs_pipeline.cpp:1646,1696` | **definitions only** (no call sites) |
| `gs_renderer.cpp` | **all 82 call sites** |

**No other translation unit calls `bufferMemoryBarrier`.**

### `additional_begin_barriers` into `executeCumsum`

| Caller | Line | additional_begin_barriers |
|--------|------|---------------------------|
| `executeLegacyDepthWaves` | ~1550 | `{{tile_ranges, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ}}` |
| `executeMacroDepthWaves` | ~3120 | `{{tile_ranges, COMPUTE_SHADER_WRITE, COMPUTE_SHADER_READ}}` |
| `executeSortPrimitivesByDepth` | ~2402 | `{}` (default) |
| `executeCalculateIndexBufferOffset` | ~2044 | `{}` (default) |

These feed into `executeCumsum` L1887 `begin_cumsum` → `bufferMemoryBarrier(barriers)` which concatenates caller barriers + input/output/(optional blockSums).

### Raw Vk buffer barriers elsewhere (not this API)

| File | Notes |
|------|-------|
| `src/visualizer/rendering/passes/vulkan_mesh_pass.cpp:2191` | raw `VkDependencyInfo.bufferMemoryBarrierCount` |
| `src/visualizer/rendering/vksplat_viewport_renderer.cpp:5305,5809` | raw pixel/depth buffer barriers |

## gs_pipeline.cpp notes

- `bufferMemoryBarrier` implementations at L1646 (pair) and L1696 (BufferBarrier).
- `executeCompute` L1884, `executeComputeIndirect` L2016 — no barriers inside.
- `validateFillRange` / alignment checks for `vkCmdFillBuffer` around L596.
- Actual `vkCmdFillBuffer` for `clearDeviceBuffer` lives in `buffer.cpp` (~L404), invoked from renderer via `clearDeviceBuffer` at L1064, L1072, L1123, L2574.

## Summary

- **Total `bufferMemoryBarrier` call sites (gs_renderer.cpp):** 82
- **Total `executeCompute` / `executeComputeIndirect` call sites:** 44 (22 direct, 22 indirect)
- **vkCmdCopyBuffer / Fill / Update in gs_renderer.cpp:** 8
- **clearDeviceBuffer sites (implicit fill):** 4
- **Automated barrier→dispatch mismatches (same-fn, immediate):** 36
- **External `bufferMemoryBarrier` callers outside these two files:** 0

### Top 5 most conservative barriers (widest masks / most buffers)

| rank | line | function | #entries | score | highlights |
|------|------|----------|----------|-------|------------|
| 1 | 1387 | `executeLegacyDepthWaves` | 24 | 300 | `b.xy_vs`, `b.inv_cov_vs_opacity`, `b.rect_tile_space`, `b.index_buffer_offset`, `b.primitive_sort_indices`, `b.rgb`, …(+18) |
| 2 | 2988 | `executeMacroDepthWaves` | 18 | 222 | `b.xy_vs`, `b.inv_cov_vs_opacity`, `b.rect_tile_space`, `b.index_buffer_offset`, `b.primitive_sort_indices`, `b.visible_count`, …(+12) |
| 3 | 1077 | `executeSelectLodThreshold` | 12 | 210 | `node_bounds`, `node_links`, `chunk_to_page`, `counts`, `out_indices`, `out_logical_indices`, …(+6) |
| 4 | 3034 | `executeMacroDepthWaves` | 14 | 210 | `b.sorting_keys_1`, `b.sorting_keys_2`, `b.sorting_gauss_idx_1`, `b.sorting_gauss_idx_2`, `tile_ranges`, `batch_counts`, …(+8) |
| 5 | 1771 | `executeSelectionMask` | 11 | 195 | `b.xyz_ws`, `b.rotations`, `b.scaling_raw`, `b.opacity_raw`, `transform_indices`, `node_mask`, …(+5) |

Narrative top conservative sites (matches score table; L1487 also high-signal):
1. **L1387 `executeLegacyDepthWaves`** — 24-buffer BufferBarrier hoist (geometry + overlays + full sort workspace).
2. **L2988 `executeMacroDepthWaves`** — 18-buffer hoist (macro chain inputs + sort workspace).
3. **L1077 `executeSelectLodThreshold`** — 12 buffers all → `COMPUTE_SHADER_READ_WRITE`.
4. **L3034 `executeMacroDepthWaves` wave>0** — 14× `COMPUTE_SHADER_READ_WRITE` between waves.
5. **L1771 `executeSelectionMask`** — 11 buffers → `COMPUTE_SHADER_READ_WRITE`.
Honorable mention: **L1487 wave>0** (11 base + optional 6 batch, all RW) — not in top-5 score only because built-vector expansion lists 12 summary rows.

### Per-function barrier tally (must sum to 82)

| function | barriers |
|----------|----------|
| `recordInstanceCountReadback` | 1 |
| `synchronizeTileInstanceGate` | 1 |
| `recordVisibleCountReadback` | 1 |
| `recordLodSelectionReadback` | 2 |
| `executeMapLodIndices` | 2 |
| `executeSelectLodThreshold` | 3 |
| `executeProjectionForward` | 4 |
| `executeLegacyDepthWaves` | 9 |
| `executeSelectionMask` | 2 |
| `executeSelectionPolygonRasterize` | 2 |
| `executeCumsum` | 7 |
| `executePrepareTileSort` | 2 |
| `executeSortIndirectCountImpl` | 6 |
| `executeSortPrimitivesByDepth` | 8 |
| `executeApplyDepthOrdering` | 1 |
| `executeCullSplats` | 6 |
| `executeProjectionForwardSurvivors` | 2 |
| `executeSortPrimitivesByDepthVisible` | 6 |
| `executeMacroCoverage` | 1 |
| `executeMacroDepthWaves` | 7 |
| `executeCalculateIndexBufferOffsetVisible` | 7 |
| `executeWavePartition` | 2 |
| **TOTAL** | **82** |

---

## Appendix: vector-variable dispatches (exact binding order)

Some `executeCompute*` calls pass a `std::vector<_VulkanBuffer>` built earlier. The chain tables may collapse those to a single name; exact ordered bindings:

### `executeProjectionForward` L1266 — `projection_buffers`
0. `b.xyz_ws` (READ)
1. `b.sh0` (READ)
2. `b.shN` (READ)
3. `b.rotations` (READ)
4. `b.scaling_raw` (READ)
5. `b.opacity_raw` (READ)
6. `b.tiles_touched` (WRITE)
7. `b.rect_tile_space` (WRITE)
8. `b.radii` (WRITE)
9. `b.xy_vs` (WRITE)
10. `b.depths` (WRITE)
11. `b.inv_cov_vs_opacity` (WRITE)
12. `b.rgb` (WRITE)
13. `b.overlay_flags` (WRITE)
14. `transform_indices` (READ)
15. `node_mask` (READ)
16. `overlay_params` (READ)
17. `model_transforms` (READ)
18. `primitive_depth_keys` (WRITE / sentinel RMW)
19. `lod_indices` or dummy (READ)
20. `lod_logical_indices` or dummy (READ)
21. `lod_levels` or dummy (READ)
22. `lod_weights` or dummy (READ)
23. `lod_counts` or dummy (READ)
24. `b.page_frames` if quant_pool (READ)

Pipeline: `pipeline_projection_forward[_quant][_3dgut]` selected by `use_gut_projection` / `quant_pool`.

### `executeProjectionForwardSurvivors` L2735 — `projection_buffers`
0–5. same attr inputs as above (xyz/sh/rot/scale/opacity) (READ)
6. `unsorted_keys` (WRITE) — placeholder slot for layout
7. `b.rect_tile_space` (WRITE)
8. `unsorted_keys` (placeholder)
9. `b.xy_vs` (WRITE)
10. `b.depths` (WRITE)
11. `b.inv_cov_vs_opacity` (WRITE)
12. `b.rgb` (WRITE)
13. `b.overlay_flags` (WRITE)
14–17. transform/node/overlay/model (READ)
18. `unsorted_keys` (placeholder)
19–23. lod_* bindings or dummies (READ)
24. `b.survivors` (READ)
25. `b.survivor_state` (READ / indirect source)
26. `unsorted_idx` (WRITE)
27. `b.visible_emit_count` (READ_WRITE atomic)
28. `b.orig_ids` (WRITE)
29. `b.page_frames` if quant_pool (READ)

Indirect: `survivor_state` @ `SurvivorState::kProjectionWordOffset`.
Pipeline: `pipeline_projection_forward[_quant]_survivors`.

### `executeLegacyDepthWaves` batched light L1572
`sorted_gauss_idx, tile_ranges, xy_vs, inv_cov, rgb, depths, pixel_state, pixel_depth, n_contributors, pixel_depth_weight, selection_mask, preview_mask, selection_colors, overlay_flags, overlay_params`
— geometry/overlay READ; pixel_* READ_WRITE.

### `executeLegacyDepthWaves` batch raster L1621 — `batch_bindings`
`sorted_gauss_idx, batch_descriptors, xy_vs, inv_cov, rgb, batch_pixel_state, batch_n_contributors` [+ overlays if active]
— batch_pixel_* READ_WRITE; rest mostly READ.

### `executeLegacyDepthWaves` compose L1655 — `compose_bindings`
`sorted_gauss_idx, batch_descriptors, batch_offsets, xy_vs, inv_cov, rgb, depths, batch_pixel_state, batch_n_contributors, pixel_state, pixel_depth, n_contributors` [+ overlays]
— pixel_* READ_WRITE; batch partials READ.

### `executeMacroDepthWaves` raster L3190 — `raster_bindings`
`sorted_gauss_idx, tile_ranges, batch_offsets, xy_vs, inv_cov, rgb, partials, active_mask` [+ overlays: selection_mask, preview_mask, selection_colors, overlay_flags, overlay_params, orig_ids]
— partials/active_mask READ_WRITE.

### `executeMacroDepthWaves` compose L3201 — `compose_bindings`
`sorted_gauss_idx, tile_ranges, batch_offsets, xy_vs, inv_cov, rgb, depths, partials, active_mask, pixel_state, pixel_depth, n_contributors` [+ overlays]
— pixel_* READ_WRITE; partials/active_mask READ.

### `executeCumsum` phases (L1858 lambda)
Not a single dispatch site — expands per size class:
- **≤block:** `single_pass` → `{input, output}`
- **≤block²:** `block_scan` → `{input, output, blockSums}`; `scan_block_sums` → same; `add_block_offsets` → same
- **≤block³:** multi-level with blockSums/blockSums2 ping-pong (see barriers 1959–1999)

### `executeWavePartition` L3396 — `bindings`
`index_buffer_offset` [, `visible_count` if visible_bounded], `tile_sort_count`, `wave_dispatch`, `predicates`
— last two WRITE; tile_sort_count READ_WRITE; others READ.

_Generated for epic1496 barrier migration recon. Precision targets: line numbers and buffer identities._