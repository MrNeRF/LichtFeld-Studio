#pragma once

#include "gs_pipeline.h"

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
    uint32_t mip_filter;
    uint32_t pad2;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float world_view_transform[16];
});
static_assert(sizeof(VulkanGSRendererUniforms) == 144);

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
});

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

class VulkanGSRenderer : public VulkanGSPipeline {
public:
    struct PrimitiveVisibilityStats {
        size_t visible_count = 0;
        size_t num_splats = 0;
    };

    VulkanGSRenderer();
    ~VulkanGSRenderer();

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
    [[nodiscard]] std::optional<PrimitiveVisibilityStats> pollDeferredPrimitiveVisibilityStats();
    [[nodiscard]] bool shrinkSortBuffersForCapacity(VulkanGSPipelineBuffers& buffers,
                                                    size_t target_capacity);

    void executeProjectionForward(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  const _VulkanBuffer& transform_indices,
                                  const _VulkanBuffer& node_mask,
                                  const _VulkanBuffer& overlay_params,
                                  const _VulkanBuffer& model_transforms,
                                  size_t alloc_reserve = 0,
                                  bool use_gut_projection = false);
    void executeGenerateKeys(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
    void executeComputeTileRanges(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers);
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
    void executeSelectionMask(const VulkanGSSelectionMaskUniforms& uniforms,
                              VulkanGSPipelineBuffers& buffers,
                              const _VulkanBuffer& transform_indices,
                              const _VulkanBuffer& node_mask,
                              const _VulkanBuffer& primitives,
                              const _VulkanBuffer& model_transforms,
                              const _VulkanBuffer& selection_out,
                              const _VulkanBuffer& polygon_mask);

    void executeSelectionPolygonRasterize(const VulkanGSSelectionPolygonRasterizeUniforms& uniforms,
                                          const _VulkanBuffer& polygon_vertices,
                                          const _VulkanBuffer& polygon_mask);

    void executeCalculateIndexBufferOffset(const VulkanGSRendererUniforms& uniforms,
                                           VulkanGSPipelineBuffers& buffers);
    // num_elements_override < 0 → use buffers.unsorted_keys().deviceSize().
    void executeSort(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers,
                     int num_bits, int64_t num_elements_override = -1);
    void executeSortTileInstances(const VulkanGSRendererUniforms& uniforms,
                                  VulkanGSPipelineBuffers& buffers,
                                  int num_bits);

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
                                  size_t capacity);
    void executeSortIndirectCountImpl(const VulkanGSRendererUniforms& uniforms,
                                      VulkanGSPipelineBuffers& buffers,
                                      int num_bits,
                                      const _VulkanBuffer& count_buffer,
                                      const _VulkanBuffer& dispatch_args_buffer,
                                      size_t capacity,
                                      const char* cpu_timer_prefix);
    void executePrepareTileSort(const VulkanGSRendererUniforms& uniforms,
                                VulkanGSPipelineBuffers& buffers);

    _ComputePipeline pipeline_projection_forward = _ComputePipeline(19);
    _ComputePipeline pipeline_projection_forward_3dgut = _ComputePipeline(19);
    _ComputePipeline pipeline_selection_mask = _ComputePipeline(9);
    _ComputePipeline pipeline_selection_polygon_rasterize = _ComputePipeline(2);
    _ComputePipeline pipeline_generate_keys = _ComputePipeline(7);
    _ComputePipeline pipeline_seed_primitive_indices = _ComputePipeline(1);
    _ComputePipeline pipeline_apply_depth_ordering = _ComputePipeline(4);
    _ComputePipeline pipeline_visible_flags = _ComputePipeline(2);
    _ComputePipeline pipeline_prepare_visible_sort = _ComputePipeline(3);
    _ComputePipeline pipeline_prepare_tile_sort = _ComputePipeline(3);
    _ComputePipeline pipeline_compact_visible_primitives = _ComputePipeline(5);
    // 3 bindings: sorted_keys, out_tile_ranges, index_buffer_offset (for num_isects).
    _ComputePipeline pipeline_compute_tile_ranges[2] = {
        _ComputePipeline(3),
        _ComputePipeline(3)};
    _ComputePipelinePair pipeline_rasterize_forward = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_rasterize_forward_3dgut = _ComputePipelinePair(20);
    _ComputePipelinePair pipeline_rasterize_forward_plain = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_rasterize_forward_3dgut_plain = _ComputePipelinePair(20);
    struct _CumsumComputePipeline {
        _ComputePipeline single_pass = _ComputePipeline(2);
        _ComputePipeline block_scan = _ComputePipeline(3);
        _ComputePipeline scan_block_sums = _ComputePipeline(3);
        _ComputePipeline add_block_offsets = _ComputePipeline(3);
    } pipeline_cumsum;
    struct _RadixSortComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(3);
        _ComputePipeline spine = _ComputePipeline(2);
        _ComputePipeline downsweep = _ComputePipeline(6);
    } pipeline_sorting_1, pipeline_sorting_2;
    struct _RadixSortIndirectComputePipeline {
        _ComputePipeline upsweep = _ComputePipeline(std::vector<int>{0, 1, 2, 3});
        _ComputePipeline spine = _ComputePipeline(std::vector<int>{0, 1, 2});
        _ComputePipeline downsweep = _ComputePipeline(std::vector<int>{0, 1, 2, 3, 4, 5, 6});
    } pipeline_sorting_indirect_1, pipeline_sorting_indirect_2;
    _ComputePipeline pipeline_null = _ComputePipeline(0);

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

    void ensureVisibleCountReadback();
    void destroyVisibleCountReadback();
    void recordVisibleCountReadback(VulkanGSPipelineBuffers& buffers, size_t num_splats);
};
