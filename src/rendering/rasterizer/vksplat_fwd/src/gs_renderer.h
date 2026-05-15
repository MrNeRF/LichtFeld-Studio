#pragma once

#include "gs_pipeline.h"

#include "perf_timer.h"

PACK_STRUCT(struct VulkanGSRendererUniforms {
    uint32_t image_height;
    uint32_t image_width;
    uint32_t grid_height;
    uint32_t grid_width;
    uint32_t num_splats;
    uint32_t active_sh;
    uint32_t step;
    uint32_t camera_model;
    float fx;
    float fy;
    float cx;
    float cy;
    float dist_coeffs[4];
    float world_view_transform[16];
});

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
});

class VulkanGSRenderer : public VulkanGSPipeline {
public:
    VulkanGSRenderer();
    ~VulkanGSRenderer();

    void initializeExternal(const std::map<std::string, std::string>& spirv_paths,
                            VkInstance external_instance,
                            VkPhysicalDevice external_physical_device,
                            VkDevice external_device,
                            VkQueue external_queue,
                            uint32_t external_queue_family_index,
                            VmaAllocator external_allocator);
    void cleanup();

    // Drop the cached num_indices estimate; the next executeCalculateIndexBufferOffset
    // will re-seed via a heuristic and reallocate sort buffers as the GPU writes
    // arrive. Call this when the splat model identity changes (different SplatData,
    // densification step, etc.) to keep the deferred readback correct.
    void resetNumIndicesEstimate();

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
                                 bool use_gut_rasterization = false);
    void executeSelectionMask(const VulkanGSSelectionMaskUniforms& uniforms,
                              VulkanGSPipelineBuffers& buffers,
                              const _VulkanBuffer& transform_indices,
                              const _VulkanBuffer& node_mask,
                              const _VulkanBuffer& primitives,
                              const _VulkanBuffer& model_transforms,
                              const _VulkanBuffer& selection_out);

    void executeCalculateIndexBufferOffset(VulkanGSPipelineBuffers& buffers);
    void executeSort(const VulkanGSRendererUniforms& uniforms, VulkanGSPipelineBuffers& buffers, int num_bits);

protected:
    void executeCumsum(
        VulkanGSPipelineBuffers& buffers,
        Buffer<int32_t>& input_buffer,
        Buffer<int32_t>& output_buffer);

    _ComputePipeline pipeline_projection_forward = _ComputePipeline(16);
    _ComputePipeline pipeline_projection_forward_gut = _ComputePipeline(16);
    _ComputePipeline pipeline_selection_mask = _ComputePipeline(8);
    _ComputePipeline pipeline_generate_keys = _ComputePipeline(7);
    // 3 bindings: sorted_keys, out_tile_ranges, index_buffer_offset (for num_isects).
    _ComputePipeline pipeline_compute_tile_ranges[2] = {
        _ComputePipeline(3),
        _ComputePipeline(3)};
    // Indirect-dispatch setup: reads cumsum tail, writes VkDispatchIndirectCommand.
    _ComputePipeline pipeline_setup_dispatch_indirect = _ComputePipeline(2);
    _ComputePipelinePair pipeline_rasterize_forward = _ComputePipelinePair(14);
    _ComputePipelinePair pipeline_rasterize_forward_gut = _ComputePipelinePair(19);
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
    _ComputePipeline pipeline_null = _ComputePipeline(0);

    // Deferred (1-frame-stale) num_indices readback, replacing the synchronous
    // mid-frame readElement that used to drain the queue every frame. The shader
    // setup_dispatch_indirect writes the indirect command directly on GPU, so
    // compute_tile_ranges no longer needs the CPU-side value at all. The mapped
    // pointer is read at the start of the next frame's executeCalculateIndexBufferOffset
    // (after the prior frame's submit fence has signaled, guaranteeing the host
    // copy is observable).
    _VulkanBuffer num_indices_readback_buffer_{};
    int32_t* num_indices_readback_mapped_ = nullptr;
    bool num_indices_readback_initialized_ = false;
    bool num_indices_readback_pending_ = false;
    size_t num_indices_estimate_ = 0;

    void ensureNumIndicesReadback();
    void destroyNumIndicesReadback();
    size_t pollDeferredNumIndices();
};
