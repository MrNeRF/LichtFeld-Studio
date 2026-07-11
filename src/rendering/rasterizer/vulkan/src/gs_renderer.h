#pragma once

#include "gs_pipeline.h"

#include "indirect_layout.h"
#include "perf_timer.h"

#include <cstdint>
#include <optional>

PACK_STRUCT(struct VulkanGSRendererUniforms {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t num_splats;
    uint32_t active_sh;
    uint32_t step;
    uint32_t camera_model;
    uint32_t sort_capacity;
    uint32_t shN_layout_slots;
    uint32_t lod_enabled;
    uint32_t lod_count;
    uint32_t mip_filter;
    uint32_t render_origin_x;
    uint32_t render_origin_y;
    uint32_t camera_width;
    uint32_t camera_height;
    uint32_t model_num_splats;
    float fx;
    float fy;
    float cx;
    float cy;
    // HiGS raster/compose wave window start; occupies the former alignment
    // padding before dist_coeffs (match shader).
    uint32_t wave_base;
    // Splats per LOD pool page for the quant-pool projection variants.
    uint32_t lod_page_splats;
    // 0 = median depth; > 0 = alpha-weighted (expected, hole-free) depth, with
    // this value as the far bound (splats beyond it are model junk that 3DGUT
    // projects onto sky pixels — excluded so they can't pollute the average).
    // Honored by the per-pixel rasterizer (alphablend_shader) regardless of backend.
    float expected_far;
    // Explicit padding: dist_coeffs is a float4 on the shader side and must
    // sit on a 16-byte boundary; both layouts pad here by hand so C++ and
    // Slang can never silently disagree.
    uint32_t uniforms_pad0;
    uint32_t uniforms_pad1;
    uint32_t uniforms_pad2;
    float dist_coeffs[4];
    float world_view_transform[16];
});
static_assert(sizeof(VulkanGSRendererUniforms) == 192);

PACK_STRUCT(struct VulkanGSLodCompactUniforms {
    uint32_t chunk_count;
    uint32_t protected_capacity;
    uint32_t miss_capacity;
    uint32_t pad0;
});

PACK_STRUCT(struct VulkanGSLodSelectUniforms {
    uint32_t node_count;
    uint32_t output_capacity;
    uint32_t chunk_splats;
    uint32_t invalid_page;
    float pixel_scale_limit;
    float object_scale;
    float behind_camera_penalty;
    float cone_foveation;
    float cone_dot0;
    float cone_dot;
    float cone_blend_denominator;
    float cone_tail_valid;
    float view_row0[4];
    float view_row1[4];
    float view_row2[4];
    float outside_view_foveation;
    float viewport_half_tan_x;
    float viewport_half_tan_y;
    float ortho_half_width;
    float ortho_half_height;
    uint32_t viewport_foveation;
    uint32_t orthographic;
    uint32_t physical_node_count;
    uint32_t logical_chunk_count;
    // Frame clock + fade window for newly streamed pages (0 disables fading).
    uint32_t current_frame;
    uint32_t fade_frames;
    uint32_t pad4;
});
static_assert(sizeof(VulkanGSLodSelectUniforms) == 144);

PACK_STRUCT(struct VulkanGSSelectionMaskUniforms {
    uint32_t num_splats;
    uint32_t primitive_count;
    uint32_t mode;
    uint32_t transform_indices_enabled;
    uint32_t node_visibility_enabled;
    uint32_t node_visibility_count;
    uint32_t num_model_transforms;
    uint32_t image_height;
    uint32_t image_width;
    uint32_t camera_model;
    uint32_t pad0;
    uint32_t pad1;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float world_view_transform[16];
    uint32_t aabb_x0;
    uint32_t aabb_y0;
    uint32_t aabb_w;
    uint32_t aabb_h;
    float ring_width;
    uint32_t mip_filter;
    uint32_t ring_pick_phase;
    uint32_t pad2;
});
static_assert(sizeof(VulkanGSSelectionMaskUniforms) == 176);

PACK_STRUCT(struct VulkanGSSelectionPolygonRasterizeUniforms {
    uint32_t vertex_count;
    uint32_t aabb_x0;
    uint32_t aabb_y0;
    uint32_t aabb_w;
    uint32_t aabb_h;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
});

inline constexpr uint32_t kLodCompactProtectedCap = 98304;
inline constexpr uint32_t kLodCompactMissCap = 16384;

class VulkanGSRenderer : public VulkanGSPipeline {
public:
    struct PrimitiveVisibilityStats {
        size_t visible_count = 0; // clamped to the frame's visible capacity
        size_t raw_count = 0;     // unclamped emit total; > visible_count means clamping
        size_t num_splats = 0;
    };
    struct TileInstanceStats {
        size_t instance_count = 0; // clamped to the frame's capacity
        size_t raw_count = 0;      // unclamped; > capacity means the frame clamped
        bool count_overflow = false;
    };
    struct LodSelectionStats {
        size_t candidate_count = 0;
        size_t rendered_capacity = 0;
        size_t overflow_count = 0;
        // GPU-compacted traversal interest: chunks the cut renders from, and
        // (chunk, priority) misses with priority = float bits of the touching
        // node's pixel scale (orderable as uint).
        std::vector<uint32_t> protected_chunks;
        std::vector<std::pair<uint32_t, uint32_t>> miss_candidates;
        uint32_t protected_overflow = 0;
        uint32_t miss_overflow = 0;
    };

    VulkanGSRenderer();
    ~VulkanGSRenderer() noexcept;

    void initializeExternal(const std::map<std::string, std::string>& spirv_paths,
                            VkInstance external_instance,
                            VkPhysicalDevice external_physical_device,
                            VkDevice external_device,
                            VkQueue external_queue,
                            uint32_t external_queue_family_index,
                            VmaAllocator external_allocator,
                            VkPipelineCache external_pipeline_cache = VK_NULL_HANDLE);
    void cleanup();

    void tagDeferredVisibleCountReadback(VkSemaphore semaphore, std::uint64_t value);
    void tagDeferredLodSelectionReadback(VkSemaphore semaphore, std::uint64_t value);
    void tagDeferredInstanceCountReadback(VkSemaphore semaphore, std::uint64_t value);
    [[nodiscard]] std::optional<PrimitiveVisibilityStats> pollDeferredPrimitiveVisibilityStats();
    [[nodiscard]] std::optional<LodSelectionStats> pollDeferredLodSelectionStats();
    [[nodiscard]] std::optional<TileInstanceStats> pollDeferredTileInstanceStats();
    [[nodiscard]] bool shrinkSortBuffersForCapacity(VulkanGSPipelineBuffers& buffers,
                                                    size_t target_capacity,
                                                    size_t visible_capacity);

    void executeProjectionForward(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  const _VulkanBuffer& transform_indices,
                                  const _VulkanBuffer& node_mask,
                                  const _VulkanBuffer& overlay_params,
                                  const _VulkanBuffer& model_transforms,
                                  size_t alloc_reserve = 0,
                                  bool use_gut_projection = false,
                                  const _VulkanBuffer& lod_indices = _VulkanBuffer(),
                                  const _VulkanBuffer& lod_logical_indices = _VulkanBuffer(),
                                  const _VulkanBuffer& lod_levels = _VulkanBuffer(),
                                  const _VulkanBuffer& lod_weights = _VulkanBuffer(),
                                  const _VulkanBuffer& lod_counts = _VulkanBuffer());
    // HiGS viewer chain. The cull prepass + survivor projection replace the
    // N-wide projection / visible-flag / compact passes: per-splat outputs are
    // written at wave-appended compact slots and the depth-sort input is
    // appended directly, so every downstream pass is bounded by the visible
    // count (GPU-resident) instead of N.
    void executeCullSplats(const VulkanGSRendererUniforms& uniforms,
                           VulkanGSPipelineBuffers& buffers,
                           const _VulkanBuffer& transform_indices,
                           const _VulkanBuffer& node_mask,
                           const _VulkanBuffer& overlay_params,
                           const _VulkanBuffer& model_transforms,
                           const _VulkanBuffer& lod_indices = _VulkanBuffer(),
                           const _VulkanBuffer& lod_logical_indices = _VulkanBuffer(),
                           const _VulkanBuffer& lod_counts = _VulkanBuffer());
    void executeProjectionForwardSurvivors(const VulkanGSRendererUniforms& uniforms,
                                           VulkanGSPipelineBuffers& buffers,
                                           const _VulkanBuffer& transform_indices,
                                           const _VulkanBuffer& node_mask,
                                           const _VulkanBuffer& overlay_params,
                                           const _VulkanBuffer& model_transforms,
                                           size_t visible_capacity,
                                           const _VulkanBuffer& lod_indices = _VulkanBuffer(),
                                           const _VulkanBuffer& lod_logical_indices = _VulkanBuffer(),
                                           const _VulkanBuffer& lod_levels = _VulkanBuffer(),
                                           const _VulkanBuffer& lod_weights = _VulkanBuffer(),
                                           const _VulkanBuffer& lod_counts = _VulkanBuffer());
    // prepare_visible_chain fan-out + indirect depth sort + sorted-id snapshot.
    void executeSortPrimitivesByDepthVisible(const VulkanGSRendererUniforms& uniforms,
                                             VulkanGSPipelineBuffers& buffers,
                                             size_t visible_capacity);
    // Per depth rank: conservative macro-tile coverage count, written in rank
    // order (combines the legacy apply-depth-ordering reorder with the
    // macro-granularity coverage). Feeds the visible-bounded cumsum.
    void executeMacroCoverage(const VulkanGSRendererUniforms& uniforms,
                              VulkanGSPipelineBuffers& buffers,
                              size_t visible_capacity);
    void executeGenerateMacroKeys(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  size_t visible_capacity,
                                  size_t instance_capacity);
    void executeComputeMacroRanges(const VulkanGSRendererUniforms& uniforms,
                                   VulkanGSPipelineBuffers& buffers,
                                   size_t instance_capacity);
    // Batch counts/offsets per macro tile + wave-chunked indirect args.
    void executeMacroBatches(const VulkanGSRendererUniforms& uniforms,
                             VulkanGSPipelineBuffers& buffers);
    // Wave loop: raster partials + compose per HIGS_RASTER_WAVE_BATCHES batches.
    void executeMacroRasterCompose(const VulkanGSRendererUniforms& uniforms,
                                   VulkanGSPipelineBuffers& buffers,
                                   size_t instance_capacity,
                                   const _VulkanBuffer& selection_mask,
                                   const _VulkanBuffer& preview_mask,
                                   const _VulkanBuffer& selection_colors,
                                   const _VulkanBuffer& overlay_params,
                                   bool overlays_active);
    [[nodiscard]] bool supportsFloat16Storage() const { return supports_float16_storage_; }
    // Indirect visible-bounded cumsum of tiles_touched_depth_ordered followed
    // by GPU count/dispatch preparation and deferred count readback.
    void executeCalculateIndexBufferOffsetVisible(const VulkanGSRendererUniforms& uniforms,
                                                  VulkanGSPipelineBuffers& buffers,
                                                  size_t visible_capacity,
                                                  size_t instance_capacity);

    void executeMapLodIndices(std::uint32_t lod_count,
                              std::uint32_t chunk_splats,
                              std::uint32_t invalid_page,
                              VulkanGSPipelineBuffers& buffers,
                              const _VulkanBuffer& chunk_to_page);
    void executeSelectLodThreshold(const VulkanGSLodSelectUniforms& uniforms,
                                   VulkanGSPipelineBuffers& buffers,
                                   const _VulkanBuffer& node_bounds,
                                   const _VulkanBuffer& node_links,
                                   const _VulkanBuffer& chunk_to_page,
                                   const _VulkanBuffer& page_age,
                                   const _VulkanBuffer& page_frames,
                                   const _VulkanBuffer& page_to_chunk);
    void executeGenerateKeys(const VulkanGSRendererUniforms& uniforms,
                             VulkanGSPipelineBuffers& buffers,
                             size_t instance_capacity);
    void executeComputeTileRanges(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  size_t instance_capacity);
    void executeRasterizeForward(const VulkanGSRendererUniforms& uniforms,
                                 VulkanGSPipelineBuffers& buffers,
                                 const _VulkanBuffer& selection_mask,
                                 const _VulkanBuffer& preview_mask,
                                 const _VulkanBuffer& selection_colors,
                                 const _VulkanBuffer& overlay_flags,
                                 const _VulkanBuffer& overlay_params,
                                 const _VulkanBuffer& transform_indices,
                                 const _VulkanBuffer& model_transforms,
                                 bool use_gut_rasterization = false,
                                 bool overlays_active = true);
    // When set, forward forces the non-batched per-pixel rasterizer: the
    // load-balanced batched compose only covers a subset of pixels, leaving the
    // rest with shared-buffer residue, which corrupts a one-shot depth readback.
    // (Whether that rasterizer writes median or expected depth is carried per
    // render by VulkanGSRendererUniforms::expected_far.)
    void setDepthCapture(bool on) { depth_capture_ = on; }
    void executeSelectionMask(const VulkanGSSelectionMaskUniforms& uniforms,
                              VulkanGSPipelineBuffers& buffers,
                              const _VulkanBuffer& transform_indices,
                              const _VulkanBuffer& node_mask,
                              const _VulkanBuffer& primitives,
                              const _VulkanBuffer& model_transforms,
                              const _VulkanBuffer& selection_out,
                              const _VulkanBuffer& polygon_mask,
                              const _VulkanBuffer& ring_pick_out);

    void executeSelectionPolygonRasterize(const VulkanGSSelectionPolygonRasterizeUniforms& uniforms,
                                          const _VulkanBuffer& polygon_vertices,
                                          const _VulkanBuffer& polygon_mask);

    void executeCalculateIndexBufferOffset(const VulkanGSRendererUniforms& uniforms,
                                           VulkanGSPipelineBuffers& buffers,
                                           size_t instance_capacity);
    void executeSortTileInstances(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  int num_bits,
                                  size_t capacity);

    // Two-stage sort stage 1: sort the N primitives by depth (radial distance
    // squared, written into buffers.primitive_depth_keys by projection_forward).
    // Projection rejects are compacted out on the GPU before the radix sort.
    // Writes the depth-ranked primitive indices into buffers.primitive_sort_indices.
    void executeSortPrimitivesByDepth(const VulkanGSRendererUniforms& uniforms,
                                      VulkanGSPipelineBuffers& buffers);

    // Reorder tiles_touched into depth-rank order so the subsequent cumsum
    // produces offsets matching the depth-ordered walk in generate_keys.
    void executeApplyDepthOrdering(const VulkanGSRendererUniforms& uniforms,
                                   VulkanGSPipelineBuffers& buffers);

protected:
    void executeCumsum(
        VulkanGSPipelineBuffers& buffers,
        Buffer<int32_t>& input_buffer,
        Buffer<int32_t>& output_buffer);

    void executeSortIndirectCount(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  int num_bits,
                                  const _VulkanBuffer& count_buffer,
                                  const _VulkanBuffer& dispatch_args_buffer,
                                  size_t capacity,
                                  const lfs::rendering::vulkan::indirect_layout::Layout& dispatch_layout,
                                  size_t radix_word_offset);
    void executeSortIndirectCountImpl(const VulkanGSRendererUniforms& uniforms,
                                      VulkanGSPipelineBuffers& buffers,
                                      int num_bits,
                                      const _VulkanBuffer& count_buffer,
                                      const _VulkanBuffer& dispatch_args_buffer,
                                      size_t capacity,
                                      const lfs::rendering::vulkan::indirect_layout::Layout& dispatch_layout,
                                      size_t radix_word_offset,
                                      const char* cpu_timer_prefix);
    void executePrepareTileSort(const VulkanGSRendererUniforms& uniforms,
                                VulkanGSPipelineBuffers& buffers,
                                size_t instance_capacity);
    void executeBatchedRasterizeForward(const VulkanGSRendererUniforms& uniforms,
                                        VulkanGSPipelineBuffers& buffers,
                                        const _VulkanBuffer& selection_mask,
                                        const _VulkanBuffer& preview_mask,
                                        const _VulkanBuffer& selection_colors,
                                        const _VulkanBuffer& overlay_flags,
                                        const _VulkanBuffer& overlay_params,
                                        bool overlays_active);

    _ComputePipeline pipeline_projection_forward = _ComputePipeline(24);
    _ComputePipeline pipeline_projection_forward_3dgut = _ComputePipeline(24);
    // Canonical quantized LOD pool variants: same binding sets plus the
    // per-page dequant frames appended last.
    _ComputePipeline pipeline_projection_forward_quant = _ComputePipeline(25);
    _ComputePipeline pipeline_projection_forward_quant_3dgut = _ComputePipeline(25);
    _ComputePipeline pipeline_selection_mask = _ComputePipeline(11);
    _ComputePipeline pipeline_selection_polygon_rasterize = _ComputePipeline(2);
    _ComputePipeline pipeline_generate_keys = _ComputePipeline(7);
    _ComputePipeline pipeline_seed_primitive_indices = _ComputePipeline(1);
    _ComputePipeline pipeline_apply_depth_ordering = _ComputePipeline(4);
    _ComputePipeline pipeline_visible_flags = _ComputePipeline(2);
    _ComputePipeline pipeline_prepare_visible_sort = _ComputePipeline(3);
    _ComputePipeline pipeline_prepare_tile_sort = _ComputePipeline(3);
    _ComputePipeline pipeline_compact_visible_primitives = _ComputePipeline(5);
    _ComputePipeline pipeline_lod_map_indices = _ComputePipeline(3);
    _ComputePipeline pipeline_lod_select_threshold = _ComputePipeline(12);
    _ComputePipeline pipeline_lod_compact_touch = _ComputePipeline(4);
    // HiGS viewer chain
    _ComputePipeline pipeline_cull_splats = _ComputePipeline(10);
    _ComputePipeline pipeline_cull_prepare = _ComputePipeline(2);
    // Bindings 6 (tiles_touched) and 8 (radii) are legacy-chain outputs and
    // absent from the survivor variant.
    _ComputePipeline pipeline_projection_forward_survivors = _ComputePipeline(std::vector<int>{
        0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28});
    _ComputePipeline pipeline_projection_forward_quant_survivors = _ComputePipeline(std::vector<int>{
        0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29});
    _ComputePipeline pipeline_prepare_visible_chain = _ComputePipeline(4);
    _ComputePipeline pipeline_copy_visible_indices = _ComputePipeline(3);
    struct _CumsumIndirectComputePipeline {
        _ComputePipeline block_scan = _ComputePipeline(4);
        _ComputePipeline scan_block_sums = _ComputePipeline(4);
        _ComputePipeline add_block_offsets = _ComputePipeline(4);
    } pipeline_cumsum_indirect;
    _ComputePipeline pipeline_prepare_tile_sort_visible = _ComputePipeline(4);
    // HiGS macro chain
    _ComputePipeline pipeline_macro_coverage = _ComputePipeline(6);
    _ComputePipeline pipeline_generate_macro_keys = _ComputePipeline(8);
    _ComputePipeline pipeline_compute_macro_ranges[2] = {
        _ComputePipeline(3),
        _ComputePipeline(3)};
    _ComputePipeline pipeline_macro_batch_counts = _ComputePipeline(2);
    _ComputePipeline pipeline_macro_batch_prepare = _ComputePipeline(2);
    _ComputePipelinePair pipeline_macro_raster = _ComputePipelinePair(8);
    _ComputePipelinePair pipeline_macro_raster_fp32 = _ComputePipelinePair(8);
    _ComputePipelinePair pipeline_macro_raster_overlays = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_macro_compose = _ComputePipelinePair(12);
    _ComputePipelinePair pipeline_macro_compose_overlays = _ComputePipelinePair(18);
    bool supports_float16_storage_ = false;
    // 3 bindings: sorted_keys, out_tile_ranges, GPU tile-instance count.
    _ComputePipeline pipeline_compute_tile_ranges[2] = {
        _ComputePipeline(3),
        _ComputePipeline(3)};
    _ComputePipelinePair pipeline_rasterize_forward = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_rasterize_forward_3dgut = _ComputePipelinePair(20);
    _ComputePipelinePair pipeline_rasterize_forward_plain = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_rasterize_forward_3dgut_plain = _ComputePipelinePair(20);
    _ComputePipelinePair pipeline_rasterize_forward_light = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_rasterize_forward_light_plain = _ComputePipelinePair(14);
    _ComputePipeline pipeline_tile_batch_counts = _ComputePipeline(2);
    _ComputePipeline pipeline_tile_batch_descriptors = _ComputePipeline(4);
    _ComputePipelinePair pipeline_rasterize_forward_batches = _ComputePipelinePair(12);
    _ComputePipelinePair pipeline_rasterize_forward_batches_plain = _ComputePipelinePair(7);
    _ComputePipeline pipeline_compose_tile_batches = _ComputePipeline(17);
    _ComputePipeline pipeline_compose_tile_batches_plain = _ComputePipeline(12);
    bool depth_capture_ = false;
    struct _CumsumComputePipeline {
        _ComputePipeline single_pass = _ComputePipeline(2);
        _ComputePipeline block_scan = _ComputePipeline(3);
        _ComputePipeline scan_block_sums = _ComputePipeline(3);
        _ComputePipeline add_block_offsets = _ComputePipeline(3);
    } pipeline_cumsum;
    struct _RadixSortIndirectComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(std::vector<int>{0, 1, 2, 3});
        _ComputePipeline spine = _ComputePipeline(std::vector<int>{0, 1, 2});
        _ComputePipeline downsweep = _ComputePipeline(std::vector<int>{0, 1, 2, 3, 4, 5, 6});
    } pipeline_sorting_indirect_1, pipeline_sorting_indirect_2;

    bool invalidateReadbackBuffer(_VulkanBuffer& buffer, VkDeviceSize size);

    // Deferred visible-count readback for diagnostics. The copy is recorded after
    // prepare_visible_sort writes buffers.visible_count and consumed on the next
    // frame, avoiding the synchronous GPU drain this instrumentation is meant to
    // diagnose.
    _VulkanBuffer visible_count_readback_buffer_{};
    uint32_t* visible_count_readback_mapped_ = nullptr;
    bool visible_count_readback_initialized_ = false;
    bool visible_count_readback_pending_ = false;
    VkSemaphore visible_count_readback_signal_ = VK_NULL_HANDLE;
    std::uint64_t visible_count_readback_value_ = 0;
    size_t visible_count_readback_num_splats_ = 0;

    _VulkanBuffer instance_count_readback_buffer_{};
    uint32_t* instance_count_readback_mapped_ = nullptr;
    bool instance_count_readback_initialized_ = false;
    bool instance_count_readback_pending_ = false;
    VkSemaphore instance_count_readback_signal_ = VK_NULL_HANDLE;
    std::uint64_t instance_count_readback_value_ = 0;

    _VulkanBuffer lod_selection_readback_buffer_{};
    uint32_t* lod_selection_readback_mapped_ = nullptr;
    bool lod_selection_readback_initialized_ = false;
    bool lod_selection_readback_pending_ = false;
    VkSemaphore lod_selection_readback_signal_ = VK_NULL_HANDLE;
    std::uint64_t lod_selection_readback_value_ = 0;
    size_t lod_selection_readback_capacity_ = 0;
    size_t lod_selection_readback_chunk_capacity_ = 0;

    void ensureVisibleCountReadback();
    void destroyVisibleCountReadback();
    void recordVisibleCountReadback(VulkanGSPipelineBuffers& buffers, size_t num_splats);
    void ensureInstanceCountReadback();
    void destroyInstanceCountReadback();
    void recordInstanceCountReadback(VulkanGSPipelineBuffers& buffers);
    void ensureLodSelectionReadback(size_t chunk_capacity);
    void destroyLodSelectionReadback();
    void recordLodSelectionReadback(VulkanGSPipelineBuffers& buffers,
                                    size_t rendered_capacity);
};
