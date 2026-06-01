#include "gs_renderer.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

VulkanGSRenderer::VulkanGSRenderer()
    : VulkanGSPipeline() {
}

VulkanGSRenderer::~VulkanGSRenderer() {
    if (commandBatchInProgress)
        endCommandBatch(false);
    destroyVisibleCountReadback();
    cleanup();
}

void VulkanGSRenderer::cleanup() {
    destroyVisibleCountReadback();
    VulkanGSPipeline::cleanup();
}

void VulkanGSRenderer::tagDeferredVisibleCountReadback(const VkSemaphore semaphore,
                                                       const std::uint64_t value) {
    if (visible_count_readback_pending_) {
        visible_count_readback_signal_ = semaphore;
        visible_count_readback_value_ = value;
    }
}

bool VulkanGSRenderer::shrinkSortBuffersForCapacity(VulkanGSPipelineBuffers& buffers,
                                                    size_t target_capacity) {
    if (commandBatchInProgress)
        _THROW_ERROR("shrinkSortBuffersForCapacity called while command batch is active");
    if (buffers.num_splats == 0)
        return false;

    target_capacity = std::max(target_capacity, buffers.num_splats);
    if (target_capacity == 0)
        return false;

    bool changed = false;
    const auto shrink_i32 = [&](Buffer<int32_t>& buffer) {
        const size_t target_bytes = target_capacity * sizeof(int32_t);
        if (buffer.deviceBuffer.allocSize > target_bytes * 2) {
            resizeDeviceBuffer(buffer, target_capacity, false);
            changed = true;
        }
    };
    const auto shrink_sort_key = [&](Buffer<sortingKey_t>& buffer) {
        const size_t target_bytes = target_capacity * sizeof(sortingKey_t);
        if (buffer.deviceBuffer.allocSize > target_bytes * 2) {
            resizeDeviceBuffer(buffer, target_capacity, false);
            changed = true;
        }
    };

    shrink_sort_key(buffers.sorting_keys_1);
    shrink_sort_key(buffers.sorting_keys_2);
    shrink_i32(buffers.sorting_gauss_idx_1);
    shrink_i32(buffers.sorting_gauss_idx_2);
    if (changed)
        buffers.num_indices_high_water = std::min(buffers.num_indices_high_water, target_capacity);
    return changed;
}

void VulkanGSRenderer::ensureVisibleCountReadback() {
    if (visible_count_readback_initialized_)
        return;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(uint32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    if (vmaCreateBuffer(allocator, &info, &aci,
                        &visible_count_readback_buffer_.buffer,
                        &visible_count_readback_buffer_.allocation,
                        &alloc_info) != VK_SUCCESS) {
        visible_count_readback_buffer_.buffer = VK_NULL_HANDLE;
        visible_count_readback_buffer_.allocation = VK_NULL_HANDLE;
        _CHECK_FATAL("Failed to allocate visible_count readback buffer");
    }
    visible_count_readback_buffer_.allocSize = sizeof(uint32_t);
    visible_count_readback_buffer_.size = sizeof(uint32_t);
    visible_count_readback_mapped_ = static_cast<uint32_t*>(alloc_info.pMappedData);
    if (visible_count_readback_mapped_)
        *visible_count_readback_mapped_ = 0;
    visible_count_readback_initialized_ = true;
    visible_count_readback_pending_ = false;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    visible_count_readback_num_splats_ = 0;
}

void VulkanGSRenderer::destroyVisibleCountReadback() {
    if (!visible_count_readback_initialized_)
        return;
    if (visible_count_readback_buffer_.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator,
                         visible_count_readback_buffer_.buffer,
                         visible_count_readback_buffer_.allocation);
    }
    visible_count_readback_buffer_ = {};
    visible_count_readback_mapped_ = nullptr;
    visible_count_readback_initialized_ = false;
    visible_count_readback_pending_ = false;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    visible_count_readback_num_splats_ = 0;
}

std::optional<VulkanGSRenderer::PrimitiveVisibilityStats>
VulkanGSRenderer::pollDeferredPrimitiveVisibilityStats() {
    // Consume only after the tagged render-completion timeline has signaled;
    // otherwise keep the previous stats and avoid a CPU-side GPU drain.
    if (!visible_count_readback_pending_ || !visible_count_readback_mapped_)
        return std::nullopt;
    if (visible_count_readback_signal_ == VK_NULL_HANDLE || visible_count_readback_value_ == 0)
        return std::nullopt;
    if (!timelineValueComplete(visible_count_readback_signal_, visible_count_readback_value_))
        return std::nullopt;
    if (!invalidateReadbackBuffer(visible_count_readback_buffer_, sizeof(uint32_t)))
        return std::nullopt;

    PrimitiveVisibilityStats stats{};
    stats.num_splats = visible_count_readback_num_splats_;
    stats.visible_count = std::min<size_t>(*visible_count_readback_mapped_, stats.num_splats);
    visible_count_readback_pending_ = false;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    return stats;
}

void VulkanGSRenderer::recordVisibleCountReadback(VulkanGSPipelineBuffers& buffers,
                                                  const size_t num_splats) {
    ensureVisibleCountReadback();
    if (buffers.visible_count.deviceBuffer.buffer == VK_NULL_HANDLE)
        return;

    VkBufferCopy copy{};
    copy.srcOffset = buffers.visible_count.deviceBuffer.offset;
    copy.dstOffset = 0;
    copy.size = sizeof(uint32_t);
    vkCmdCopyBuffer(command_buffer,
                    buffers.visible_count.deviceBuffer.buffer,
                    visible_count_readback_buffer_.buffer,
                    1,
                    &copy);
    bufferMemoryBarrier({{visible_count_readback_buffer_, TRANSFER_WRITE}},
                        HOST_READ);
    visible_count_readback_pending_ = true;
    visible_count_readback_signal_ = VK_NULL_HANDLE;
    visible_count_readback_value_ = 0;
    visible_count_readback_num_splats_ = num_splats;
}

bool VulkanGSRenderer::invalidateReadbackBuffer(_VulkanBuffer& buffer, VkDeviceSize size) {
    if (buffer.allocation == VK_NULL_HANDLE)
        return false;
    return vmaInvalidateAllocation(allocator, buffer.allocation, 0, size) == VK_SUCCESS;
}

void VulkanGSRenderer::initializeExternal(const std::map<std::string, std::string>& spirv_paths,
                                          VkInstance external_instance,
                                          VkPhysicalDevice external_physical_device,
                                          VkDevice external_device,
                                          VkQueue external_queue,
                                          uint32_t external_queue_family_index,
                                          VmaAllocator external_allocator,
                                          VkPipelineCache external_pipeline_cache) {
    destroyVisibleCountReadback();
    VulkanGSPipeline::initializeExternal(
        external_instance,
        external_physical_device,
        external_device,
        external_queue,
        external_queue_family_index,
        external_allocator,
        external_pipeline_cache);

    createComputePipeline(pipeline_projection_forward, spirv_paths.at("projection_forward"));
    createComputePipeline(pipeline_projection_forward_3dgut, spirv_paths.at("projection_forward_3dgut"));
    createComputePipeline(pipeline_selection_mask, spirv_paths.at("selection_mask"));
    createComputePipeline(pipeline_selection_polygon_rasterize, spirv_paths.at("selection_polygon_rasterize"));
    createComputePipeline(pipeline_generate_keys, spirv_paths.at("generate_keys"));
    for (int i = 0; i < 2; ++i) {
        createComputePipeline(pipeline_compute_tile_ranges[i], spirv_paths.at("compute_tile_ranges"));
        createComputePipeline(pipeline_rasterize_forward[i], spirv_paths.at("rasterize_forward"));
        createComputePipeline(pipeline_rasterize_forward_3dgut[i], spirv_paths.at("rasterize_forward_3dgut"));
        createComputePipeline(pipeline_rasterize_forward_plain[i], spirv_paths.at("rasterize_forward_plain"));
        createComputePipeline(pipeline_rasterize_forward_3dgut_plain[i], spirv_paths.at("rasterize_forward_3dgut_plain"));
    }
    createComputePipeline(pipeline_cumsum.single_pass, spirv_paths.at("cumsum_single_pass"));
    createComputePipeline(pipeline_cumsum.block_scan, spirv_paths.at("cumsum_block_scan"));
    createComputePipeline(pipeline_cumsum.scan_block_sums, spirv_paths.at("cumsum_scan_block_sums"));
    createComputePipeline(pipeline_cumsum.add_block_offsets, spirv_paths.at("cumsum_add_block_offsets"));
    createComputePipeline(pipeline_sorting_1.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_1.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_1.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sorting_2.upsweep, spirv_paths.at("radix_sort/upsweep"));
    createComputePipeline(pipeline_sorting_2.spine, spirv_paths.at("radix_sort/spine"));
    createComputePipeline(pipeline_sorting_2.downsweep, spirv_paths.at("radix_sort/downsweep"));
    createComputePipeline(pipeline_sorting_indirect_1.upsweep, spirv_paths.at("radix_sort/upsweep_indirect"));
    createComputePipeline(pipeline_sorting_indirect_1.spine, spirv_paths.at("radix_sort/spine_indirect"));
    createComputePipeline(pipeline_sorting_indirect_1.downsweep, spirv_paths.at("radix_sort/downsweep_indirect"));
    createComputePipeline(pipeline_sorting_indirect_2.upsweep, spirv_paths.at("radix_sort/upsweep_indirect"));
    createComputePipeline(pipeline_sorting_indirect_2.spine, spirv_paths.at("radix_sort/spine_indirect"));
    createComputePipeline(pipeline_sorting_indirect_2.downsweep, spirv_paths.at("radix_sort/downsweep_indirect"));
    createComputePipeline(pipeline_seed_primitive_indices, spirv_paths.at("seed_primitive_indices"));
    createComputePipeline(pipeline_apply_depth_ordering, spirv_paths.at("apply_depth_ordering"));
    createComputePipeline(pipeline_visible_flags, spirv_paths.at("visible_flags"));
    createComputePipeline(pipeline_prepare_visible_sort, spirv_paths.at("prepare_visible_sort"));
    createComputePipeline(pipeline_prepare_tile_sort, spirv_paths.at("prepare_tile_sort"));
    createComputePipeline(pipeline_compact_visible_primitives, spirv_paths.at("compact_visible_primitives"));
}

void VulkanGSRenderer::executeProjectionForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& node_mask,
    const _VulkanBuffer& overlay_params,
    const _VulkanBuffer& model_transforms,
    size_t alloc_reserve,
    bool use_gut_projection) {
    PerfTimer::Timer<PerfTimer::ProjectionForward> timer(this);
    DEVICE_GUARD;

    size_t num_splats = buffers.num_splats;

    bufferMemoryBarrier({
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.sh0.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.shN.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.opacity_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {node_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {overlay_params, TRANSFER_COMPUTE_SHADER_WRITE},
                            {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    size_t alloc_size = std::max(num_splats, alloc_reserve);

    // Two-stage sort: pre-fill primitive_depth_keys with 0xFFFFFFFFu so any
    // primitive that hits an early-return path inside the projection shader
    // (z-near reject, opacity below threshold, degenerate covariance, zero
    // tiles touched) keeps the max-key sentinel and sorts to the tail.
    auto& primitive_depth_keys =
        resizeDeviceBuffer(buffers.primitive_depth_keys, alloc_size);
    bufferMemoryBarrier({{primitive_depth_keys, COMPUTE_SHADER_READ_WRITE}},
                        TRANSFER_COMPUTE_SHADER_WRITE);
    vkCmdFillBuffer(command_buffer, primitive_depth_keys.buffer,
                    primitive_depth_keys.offset, primitive_depth_keys.size,
                    0xFFFFFFFFu);
    bufferMemoryBarrier({{primitive_depth_keys, TRANSFER_COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{num_splats, SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        use_gut_projection ? pipeline_projection_forward_3dgut : pipeline_projection_forward,
        {
            // inputs
            buffers.xyz_ws.deviceBuffer,
            buffers.sh0.deviceBuffer,
            buffers.shN.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.scaling_raw.deviceBuffer,
            buffers.opacity_raw.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.tiles_touched, alloc_size),
            resizeDeviceBuffer(buffers.rect_tile_space, alloc_size),
            resizeDeviceBuffer(buffers.radii, alloc_size),
            resizeDeviceBuffer(buffers.xy_vs, 2 * alloc_size),
            resizeDeviceBuffer(buffers.depths, alloc_size),
            resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4 * alloc_size),
            resizeDeviceBuffer(buffers.rgb, 3 * alloc_size),
            resizeDeviceBuffer(buffers.overlay_flags, alloc_size),
            transform_indices,
            node_mask,
            overlay_params,
            model_transforms,
            primitive_depth_keys,
        });
}

void VulkanGSRenderer::executeGenerateKeys(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::GenerateKeys> timer(this);
    DEVICE_GUARD;

    const size_t num_elements = buffers.num_splats;
    // executeCalculateIndexBufferOffset has synchronously read the cumsum tail,
    // so num_indices is the exact tile-instance count for this frame.
    const size_t capacity = buffers.num_indices;

    auto& unsorted_keys = resizeDeviceBuffer(buffers.unsorted_keys(), capacity);
    auto& unsorted_idx = resizeDeviceBuffer(buffers.unsorted_gauss_idx(), capacity);

    // Pre-fill with the max sentinel; this keeps any untouched entries harmless
    // if a shader path emits fewer keys than the exact cumsum count.
    bufferMemoryBarrier({{unsorted_keys, COMPUTE_SHADER_READ_WRITE}},
                        TRANSFER_COMPUTE_SHADER_WRITE);
    vkCmdFillBuffer(command_buffer, unsorted_keys.buffer, unsorted_keys.offset, unsorted_keys.size,
                    0xFFFFFFFFu);
    bufferMemoryBarrier({{unsorted_keys, TRANSFER_COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ_WRITE);

    executeCompute(
        {{num_elements, 64}},
        &uniforms, sizeof(uniforms),
        pipeline_generate_keys,
        {
            // inputs
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rect_tile_space.deviceBuffer,
            buffers.index_buffer_offset.deviceBuffer,
            buffers.primitive_sort_indices.deviceBuffer,
            // outputs
            unsorted_keys,
            unsorted_idx,
        });
}

void VulkanGSRenderer::executeComputeTileRanges(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::ComputeTileRanges> timer(this);
    DEVICE_GUARD;

    const size_t num_tiles = (size_t)(uniforms.grid_height * uniforms.grid_width);

    bufferMemoryBarrier({
                            {buffers.sorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    // Dispatch over the exact CPU-known tile-instance count. The shader still
    // clamps to uniforms.sort_capacity as a defensive bounds check.
    executeCompute(
        {{buffers.num_indices + 1, 256}},
        &uniforms, sizeof(uniforms),
        pipeline_compute_tile_ranges[buffers.is_unsorted_1],
        {
            buffers.sorted_keys().deviceBuffer,
            resizeDeviceBuffer(buffers.tile_ranges, num_tiles + 1),
            buffers.index_buffer_offset.deviceBuffer,
        });
}

void VulkanGSRenderer::executeRasterizeForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& selection_mask,
    const _VulkanBuffer& preview_mask,
    const _VulkanBuffer& selection_colors,
    const _VulkanBuffer& overlay_flags,
    const _VulkanBuffer& overlay_params,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& model_transforms,
    bool use_gut_rasterization,
    bool overlays_active) {
    if (buffers.num_indices == 0)
        return;

    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);
    DEVICE_GUARD;

    size_t num_pixels = uniforms.image_height * uniforms.image_width;

    bufferMemoryBarrier({
                            {buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE},
                            {buffers.tile_ranges.deviceBuffer, COMPUTE_SHADER_WRITE},
                            {buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE},
                            {buffers.depths.deviceBuffer, COMPUTE_SHADER_WRITE},
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.opacity_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {selection_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {preview_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {selection_colors, TRANSFER_COMPUTE_SHADER_WRITE},
                            {overlay_flags, COMPUTE_SHADER_WRITE},
                            {overlay_params, TRANSFER_COMPUTE_SHADER_WRITE},
                            {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    if (use_gut_rasterization) {
        auto& gut_pipeline = overlays_active
                                 ? pipeline_rasterize_forward_3dgut
                                 : pipeline_rasterize_forward_3dgut_plain;
        executeCompute(
            {{uniforms.image_width, TILE_WIDTH}, {uniforms.image_height, TILE_HEIGHT}},
            &uniforms, sizeof(uniforms),
            gut_pipeline[buffers.is_unsorted_1],
            std::vector<_VulkanBuffer>({
                // inputs
                buffers.sorted_gauss_idx().deviceBuffer,
                buffers.tile_ranges.deviceBuffer,
                buffers.xy_vs.deviceBuffer,
                buffers.inv_cov_vs_opacity.deviceBuffer,
                buffers.rgb.deviceBuffer,
                buffers.depths.deviceBuffer,
                buffers.xyz_ws.deviceBuffer,
                buffers.rotations.deviceBuffer,
                buffers.scaling_raw.deviceBuffer,
                buffers.opacity_raw.deviceBuffer,
                // outputs
                resizeDeviceBuffer(buffers.pixel_state, 4 * num_pixels),
                resizeDeviceBuffer(buffers.pixel_depth, num_pixels),
                resizeDeviceBuffer(buffers.n_contributors, num_pixels),
                // selection overlay inputs
                selection_mask,
                preview_mask,
                selection_colors,
                overlay_flags,
                overlay_params,
                transform_indices,
                model_transforms,
            }));
    } else {
        auto& pipeline = overlays_active
                             ? pipeline_rasterize_forward
                             : pipeline_rasterize_forward_plain;
        executeCompute(
            {{uniforms.image_width, TILE_WIDTH}, {uniforms.image_height, TILE_HEIGHT}},
            &uniforms, sizeof(uniforms),
            pipeline[buffers.is_unsorted_1],
            std::vector<_VulkanBuffer>({
                // inputs
                buffers.sorted_gauss_idx().deviceBuffer,
                buffers.tile_ranges.deviceBuffer,
                buffers.xy_vs.deviceBuffer,
                buffers.inv_cov_vs_opacity.deviceBuffer,
                buffers.rgb.deviceBuffer,
                buffers.depths.deviceBuffer,
                // outputs
                resizeDeviceBuffer(buffers.pixel_state, 4 * num_pixels),
                resizeDeviceBuffer(buffers.pixel_depth, num_pixels),
                resizeDeviceBuffer(buffers.n_contributors, num_pixels),
                // selection overlay inputs
                selection_mask,
                preview_mask,
                selection_colors,
                overlay_flags,
                overlay_params,
            }));
    }
}

void VulkanGSRenderer::executeSelectionMask(
    const VulkanGSSelectionMaskUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const _VulkanBuffer& transform_indices,
    const _VulkanBuffer& node_mask,
    const _VulkanBuffer& primitives,
    const _VulkanBuffer& model_transforms,
    const _VulkanBuffer& selection_out,
    const _VulkanBuffer& polygon_mask) {
    DEVICE_GUARD;

    bufferMemoryBarrier({
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scaling_raw.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {transform_indices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {node_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                            {primitives, TRANSFER_COMPUTE_SHADER_WRITE},
                            {model_transforms, TRANSFER_COMPUTE_SHADER_WRITE},
                            {selection_out, TRANSFER_COMPUTE_SHADER_WRITE},
                            {polygon_mask, COMPUTE_SHADER_READ_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    const size_t num_words = _CEIL_DIV(static_cast<size_t>(uniforms.num_splats), 4);
    executeCompute(
        {{num_words, SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_selection_mask,
        {
            buffers.xyz_ws.deviceBuffer,
            transform_indices,
            node_mask,
            primitives,
            model_transforms,
            buffers.rotations.deviceBuffer,
            buffers.scaling_raw.deviceBuffer,
            selection_out,
            polygon_mask,
        });

    bufferMemoryBarrier({{selection_out, COMPUTE_SHADER_WRITE}}, TRANSFER_READ);
}

void VulkanGSRenderer::executeSelectionPolygonRasterize(
    const VulkanGSSelectionPolygonRasterizeUniforms& uniforms,
    const _VulkanBuffer& polygon_vertices,
    const _VulkanBuffer& polygon_mask) {
    DEVICE_GUARD;

    bufferMemoryBarrier({
                            {polygon_vertices, TRANSFER_COMPUTE_SHADER_WRITE},
                            {polygon_mask, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    constexpr size_t kBlockXY = 8;
    executeCompute(
        {{static_cast<size_t>(uniforms.aabb_w), kBlockXY},
         {static_cast<size_t>(uniforms.aabb_h), kBlockXY}},
        &uniforms, sizeof(uniforms),
        pipeline_selection_polygon_rasterize,
        {
            polygon_vertices,
            polygon_mask,
        });

    bufferMemoryBarrier({{polygon_mask, COMPUTE_SHADER_WRITE}}, COMPUTE_SHADER_READ);
}

void VulkanGSRenderer::executeCumsum(
    VulkanGSPipelineBuffers& buffers,
    Buffer<int32_t>& input_buffer,
    Buffer<int32_t>& output_buffer) {
    PerfTimer::Timer<PerfTimer::_Cumsum> timer(this);
    DEVICE_GUARD;

    size_t num_elements = input_buffer.deviceSize();
    const size_t block_0 = 1024;
    const size_t block_limit = deviceInfo.subgroupSize * deviceInfo.subgroupSize * deviceInfo.subgroupSize;
    const size_t block = std::min(block_0, block_limit);

    auto execute_cumsum_phase = [&](size_t active_elements,
                                    size_t threads_per_group,
                                    _ComputePipeline& pipeline,
                                    const std::vector<_VulkanBuffer>& phase_buffers) {
        uint32_t phase_uniforms[1] = {static_cast<uint32_t>(active_elements)};
        executeCompute(
            {{active_elements, threads_per_group}},
            phase_uniforms,
            sizeof(uint32_t),
            pipeline,
            phase_buffers);
    };

    bufferMemoryBarrier({
                            {input_buffer.deviceBuffer, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    resizeDeviceBuffer(output_buffer, num_elements);

    if (num_elements <= block_0) {
        execute_cumsum_phase(
            num_elements, block_0,
            pipeline_cumsum.single_pass,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
            });
    }

    else if (num_elements <= block * block) {
        const size_t num_blocks = _CEIL_DIV(num_elements, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_blocks, true);

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_blocks, block,
            pipeline_cumsum.scan_block_sums,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });
    }

    else if (num_elements <= block * block * block) {
        const size_t num_elements_1 = _CEIL_DIV(num_elements, block);
        const size_t num_elements_2 = _CEIL_DIV(num_elements_1, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_elements_1, true);
        resizeDeviceBuffer(buffers._cumsum_blockSums2, num_elements_2, true);

        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.block_scan,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements_1, block,
            pipeline_cumsum.block_scan,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                {buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements_2, block,
            pipeline_cumsum.scan_block_sums,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {buffers._cumsum_blockSums2.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements_1, block,
            pipeline_cumsum.add_block_offsets,
            {
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
                buffers._cumsum_blockSums2.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {output_buffer.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers._cumsum_blockSums.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        execute_cumsum_phase(
            num_elements, block,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });
    }

    // can't reasonably expect more than 1G splats
    // although there may be more than 1G sorting indices
    else {
        _THROW_ERROR("Too many numbers for cumsum");
    }
}

void VulkanGSRenderer::executeCalculateIndexBufferOffset(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::CalculateIndexBufferOffset> timer(this);

    const size_t num_elements = buffers.num_splats;
    if (num_elements == 0) {
        buffers.num_indices = 0;
        return;
    }

    // Cumsum on tiles_touched_depth_ordered (output of executeApplyDepthOrdering)
    // so index_buffer_offset[depth_rank] gives the contiguous offset interval
    // for the primitive at depth rank `depth_rank`. Matches the gsplat_fwd CUDA
    // reference (cub::DeviceScan::ExclusiveSum on the reordered offsets array).
    executeCumsum(
        buffers,
        buffers.tiles_touched_depth_ordered,
        buffers.index_buffer_offset);

    DEVICE_GUARD;

    bufferMemoryBarrier({
                            {buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                        },
                        TRANSFER_COMPUTE_SHADER_READ);

    const int32_t num_indices =
        readElement<int32_t>(buffers.index_buffer_offset.deviceBuffer, num_elements - 1);
    buffers.num_indices = num_indices < 0 ? 0u : static_cast<size_t>(num_indices);
    buffers.num_indices_high_water =
        std::max(buffers.num_indices_high_water, buffers.num_indices);

    executePrepareTileSort(uniforms, buffers);
}

void VulkanGSRenderer::executePrepareTileSort(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::PrepareTileSort> timer(this);
    [[maybe_unused]] auto cpu_timer =
        timeCpuStage("vksplat.render.record.executePrepareTileSort");
    DEVICE_GUARD;

    resizeDeviceBuffer(buffers.tile_sort_count, 1);
    resizeDeviceBuffer(buffers.tile_sort_dispatch_args, 3);

    struct PrepareTileSortUniforms {
        uint32_t num_splats;
        uint32_t sort_capacity;
        uint32_t sort_partition_size;
        uint32_t pad0;
    } prepare_uniforms{
        static_cast<uint32_t>(buffers.num_splats),
        static_cast<uint32_t>(
            std::min<std::size_t>(buffers.num_indices,
                                  static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()))),
        512u * 8u,
        0u};

    bufferMemoryBarrier({
                            {buffers.index_buffer_offset.deviceBuffer, TRANSFER_COMPUTE_SHADER_READ_WRITE},
                        },
                        COMPUTE_SHADER_READ);
    executeCompute(
        {{1, 1}},
        &prepare_uniforms, sizeof(prepare_uniforms),
        pipeline_prepare_tile_sort,
        {
            buffers.index_buffer_offset.deviceBuffer,
            buffers.tile_sort_count.deviceBuffer,
            buffers.tile_sort_dispatch_args.deviceBuffer,
        });
}

void VulkanGSRenderer::executeSort(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits,
    int64_t num_elements_override) {
    PerfTimer::Timer<PerfTimer::SortRTS> timer(this);

    size_t buffer_capacity = buffers.unsorted_keys().deviceSize();
    if (buffer_capacity != buffers.unsorted_gauss_idx().deviceSize())
        _THROW_ERROR("number of elements don't match in executeSort");
    size_t num_elements = num_elements_override < 0
                              ? buffer_capacity
                              : std::min<size_t>(buffer_capacity,
                                                 static_cast<size_t>(num_elements_override));

    const int RADIX = 256;
    const int WORKGROUP_SIZE = 512;
    const int PARTITION_DIVISION = 8;
    const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

    auto& globalHistogram = buffers._sorting_histogram;
    auto& partitionHistogram = buffers._sorting_histogram_cumsum;

    const size_t num_parts = _CEIL_DIV(num_elements, PARTITION_SIZE);

    int max_nonzero_bit = 8 * sizeof(sortingKey_t);
    if (num_bits == -1 && sizeof(sortingKey_t) == 8) {
        int32_t num_tiles = (int32_t)(uniforms.grid_height * uniforms.grid_width);
        max_nonzero_bit = 23; // float fraction bits
        int32_t temp = num_tiles;
        while (temp)
            temp >>= 1, max_nonzero_bit++;
    } else if (num_bits >= 0)
        max_nonzero_bit = num_bits;
    int num_passes = _CEIL_DIV(max_nonzero_bit, 8);

    resizeDeviceBuffer(partitionHistogram, num_parts * RADIX);
    resizeDeviceBuffer(buffers.sorted_keys(), num_elements);
    resizeDeviceBuffer(buffers.sorted_gauss_idx(), num_elements);

    DEVICE_GUARD;
    clearDeviceBuffer(globalHistogram, num_passes * RADIX);
    bufferMemoryBarrier({
                            {globalHistogram.deviceBuffer, TRANSFER_WRITE},
                        },
                        COMPUTE_SHADER_READ_WRITE);

    for (int pass = 0; 8 * pass < max_nonzero_bit; pass++) {

        auto& pipeline_sorting = buffers.is_unsorted_1 ? pipeline_sorting_1 : pipeline_sorting_2;

        uint32_t uniforms[2];
        uniforms[0] = (uint32_t)pass;
        uniforms[1] = (uint32_t)num_elements;

        if (pass)
            bufferMemoryBarrier({
                                    {buffers.unsorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE},
                                    {buffers.unsorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE},
                                },
                                COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{num_parts, 1}},
            uniforms, 2 * sizeof(int32_t),
            pipeline_sorting.upsweep,
            {
                buffers.unsorted_keys().deviceBuffer,
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                {partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
        executeCompute(
            {{RADIX, 1}},
            uniforms, 2 * sizeof(int32_t),
            pipeline_sorting.spine,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
            });

        bufferMemoryBarrier({
                                {globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                {partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{num_parts, 1}},
            uniforms, 2 * sizeof(int32_t),
            pipeline_sorting.downsweep,
            {
                globalHistogram.deviceBuffer,
                partitionHistogram.deviceBuffer,
                buffers.unsorted_keys().deviceBuffer,
                buffers.unsorted_gauss_idx().deviceBuffer,
                buffers.sorted_keys().deviceBuffer,
                buffers.sorted_gauss_idx().deviceBuffer,
            });

        buffers.is_unsorted_1 = !buffers.is_unsorted_1;
    }
    buffers.is_unsorted_1 = !buffers.is_unsorted_1;
}

void VulkanGSRenderer::executeSortIndirectCount(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits,
    const _VulkanBuffer& count_buffer,
    const _VulkanBuffer& dispatch_args_buffer,
    size_t capacity) {
    PerfTimer::Timer<PerfTimer::SortVisiblePrimitives> timer(this);
    executeSortIndirectCountImpl(uniforms,
                                 buffers,
                                 num_bits,
                                 count_buffer,
                                 dispatch_args_buffer,
                                 capacity,
                                 "vksplat.render.record.sort_primitive_indirect");
}

void VulkanGSRenderer::executeSortTileInstances(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    const int num_bits) {
    PerfTimer::Timer<PerfTimer::SortRTS> timer(this);
    executeSortIndirectCountImpl(uniforms,
                                 buffers,
                                 num_bits,
                                 buffers.tile_sort_count.deviceBuffer,
                                 buffers.tile_sort_dispatch_args.deviceBuffer,
                                 buffers.num_indices,
                                 "vksplat.render.record.sort_tile_indirect");
}

void VulkanGSRenderer::executeSortIndirectCountImpl(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits,
    const _VulkanBuffer& count_buffer,
    const _VulkanBuffer& dispatch_args_buffer,
    size_t capacity,
    const char* cpu_timer_prefix) {
    if (capacity == 0)
        return;
    if (capacity != buffers.unsorted_keys().deviceSize() ||
        capacity != buffers.unsorted_gauss_idx().deviceSize())
        _THROW_ERROR("indirect sort capacity does not match input buffer sizes");

    const auto timer_name = [cpu_timer_prefix](const char* suffix) {
        return std::string(cpu_timer_prefix) + suffix;
    };

    const int RADIX = 256;
    const int WORKGROUP_SIZE = 512;
    const int PARTITION_DIVISION = 8;
    const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

    auto& globalHistogram = buffers._sorting_histogram;
    auto& partitionHistogram = buffers._sorting_histogram_cumsum;

    const size_t num_parts_capacity = _CEIL_DIV(capacity, PARTITION_SIZE);

    int max_nonzero_bit = 8 * sizeof(sortingKey_t);
    if (num_bits == -1 && sizeof(sortingKey_t) == 8) {
        int32_t num_tiles = (int32_t)(uniforms.grid_height * uniforms.grid_width);
        max_nonzero_bit = 23;
        int32_t temp = num_tiles;
        while (temp)
            temp >>= 1, max_nonzero_bit++;
    } else if (num_bits >= 0)
        max_nonzero_bit = num_bits;
    int num_passes = _CEIL_DIV(max_nonzero_bit, 8);

    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".resize_buffers"));
        resizeDeviceBuffer(partitionHistogram, num_parts_capacity * RADIX);
        resizeDeviceBuffer(buffers.sorted_keys(), capacity);
        resizeDeviceBuffer(buffers.sorted_gauss_idx(), capacity);
    }

    DEVICE_GUARD;
    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".clear_histogram"));
        clearDeviceBuffer(globalHistogram, num_passes * RADIX);
        bufferMemoryBarrier({
                                {globalHistogram.deviceBuffer, TRANSFER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);
    }
    {
        [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".prepare_count_and_dispatch"));
        bufferMemoryBarrier({
                                {count_buffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        bufferMemoryBarrier({
                                {dispatch_args_buffer, COMPUTE_SHADER_WRITE},
                            },
                            INDIRECT_DISPATCH_READ);
    }

    for (int pass = 0; 8 * pass < max_nonzero_bit; pass++) {
        auto& pipeline_sorting = buffers.is_unsorted_1 ? pipeline_sorting_indirect_1
                                                       : pipeline_sorting_indirect_2;

        uint32_t sort_uniforms[2];
        sort_uniforms[0] = static_cast<uint32_t>(pass);
        sort_uniforms[1] = 0;

        if (pass) {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_pingpong_barrier"));
            bufferMemoryBarrier({
                                    {buffers.unsorted_keys().deviceBuffer, COMPUTE_SHADER_WRITE},
                                    {buffers.unsorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE},
                                },
                                COMPUTE_SHADER_READ_WRITE);
        }
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_upsweep"));
            executeComputeIndirect(
                dispatch_args_buffer,
                0,
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.upsweep,
                {
                    buffers.unsorted_keys().deviceBuffer,
                    globalHistogram.deviceBuffer,
                    partitionHistogram.deviceBuffer,
                    count_buffer,
                });
        }

        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_upsweep_to_spine_barrier"));
            bufferMemoryBarrier({
                                    {globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                    {partitionHistogram.deviceBuffer, COMPUTE_SHADER_WRITE},
                                },
                                COMPUTE_SHADER_READ_WRITE);
        }
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_spine"));
            executeCompute(
                {{RADIX, 1}},
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.spine,
                {
                    globalHistogram.deviceBuffer,
                    partitionHistogram.deviceBuffer,
                    count_buffer,
                });
        }

        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_spine_to_downsweep_barrier"));
            bufferMemoryBarrier({
                                    {globalHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                    {partitionHistogram.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                                },
                                COMPUTE_SHADER_READ);
        }
        {
            [[maybe_unused]] auto cpu_timer = timeCpuStage(timer_name(".pass_downsweep"));
            executeComputeIndirect(
                dispatch_args_buffer,
                0,
                sort_uniforms, 2 * sizeof(int32_t),
                pipeline_sorting.downsweep,
                {
                    globalHistogram.deviceBuffer,
                    partitionHistogram.deviceBuffer,
                    buffers.unsorted_keys().deviceBuffer,
                    buffers.unsorted_gauss_idx().deviceBuffer,
                    buffers.sorted_keys().deviceBuffer,
                    buffers.sorted_gauss_idx().deviceBuffer,
                    count_buffer,
                });
        }

        buffers.is_unsorted_1 = !buffers.is_unsorted_1;
    }
    buffers.is_unsorted_1 = !buffers.is_unsorted_1;
}

void VulkanGSRenderer::executeSortPrimitivesByDepth(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::SortPrimitivesByDepth> timer(this);

    const size_t num_splats = buffers.num_splats;
    if (num_splats == 0)
        return;

    DEVICE_GUARD;

    // Stage 1 follows the old CUDA path: reject/projection work stays N-wide,
    // but the expensive depth radix sort only sees compact visible primitives.
    // The ping-pong sort buffers still have N capacity so the GPU scatter cannot
    // overflow; the indirect sort count comes from the visible-prefix tail.
    _VulkanBuffer* unsorted_keys = nullptr;
    _VulkanBuffer* unsorted_idx = nullptr;
    {
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.ensure_buffers");
        unsorted_keys = &resizeDeviceBuffer(buffers.unsorted_keys(), num_splats);
        unsorted_idx = &resizeDeviceBuffer(buffers.unsorted_gauss_idx(), num_splats);
        resizeDeviceBuffer(buffers.visible_flags, num_splats);
        resizeDeviceBuffer(buffers.visible_count, 1);
        resizeDeviceBuffer(buffers.visible_sort_dispatch_args, 3);
    }

    struct VisibleUniforms {
        uint32_t num_splats;
        uint32_t pad0, pad1, pad2;
    } visible_uniforms{static_cast<uint32_t>(num_splats), 0, 0, 0};

    {
        PerfTimer::Timer<PerfTimer::BuildVisibleFlags> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.build_visible_flags");
        bufferMemoryBarrier({
                                {buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{num_splats, 64}},
            &visible_uniforms, sizeof(visible_uniforms),
            pipeline_visible_flags,
            {
                buffers.tiles_touched.deviceBuffer,
                buffers.visible_flags.deviceBuffer,
            });
    }

    {
        PerfTimer::Timer<PerfTimer::VisiblePrefix> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.visible_prefix");
        executeCumsum(buffers, buffers.visible_flags, buffers.visible_prefix);
    }

    struct PrepareUniforms {
        uint32_t num_splats;
        uint32_t sort_partition_size;
        uint32_t pad0, pad1;
    } prepare_uniforms{static_cast<uint32_t>(num_splats), 512u * 8u, 0, 0};

    {
        PerfTimer::Timer<PerfTimer::PrepareVisibleSort> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.prepare_visible_sort");
        bufferMemoryBarrier({
                                {buffers.visible_prefix.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{1, 1}},
            &prepare_uniforms, sizeof(prepare_uniforms),
            pipeline_prepare_visible_sort,
            {
                buffers.visible_prefix.deviceBuffer,
                buffers.visible_count.deviceBuffer,
                buffers.visible_sort_dispatch_args.deviceBuffer,
            });
        bufferMemoryBarrier({
                                {buffers.visible_count.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            TRANSFER_COMPUTE_SHADER_READ);
        recordVisibleCountReadback(buffers, num_splats);
    }

    {
        PerfTimer::Timer<PerfTimer::CompactVisiblePrimitives> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.compact_visible_primitives");
        bufferMemoryBarrier({
                                {buffers.primitive_depth_keys.deviceBuffer, COMPUTE_SHADER_WRITE},
                                {buffers.visible_prefix.deviceBuffer, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ);
        executeCompute(
            {{num_splats, 64}},
            &visible_uniforms, sizeof(visible_uniforms),
            pipeline_compact_visible_primitives,
            {
                buffers.tiles_touched.deviceBuffer,
                buffers.visible_prefix.deviceBuffer,
                buffers.primitive_depth_keys.deviceBuffer,
                *unsorted_keys,
                *unsorted_idx,
            });
    }

    {
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.sort_visible_primitives");
        bufferMemoryBarrier({
                                {*unsorted_keys, COMPUTE_SHADER_WRITE},
                                {*unsorted_idx, COMPUTE_SHADER_WRITE},
                            },
                            COMPUTE_SHADER_READ_WRITE);

        // Stage 1 sort: full 32-bit depth keys, but only for compact visible
        // primitives. The dispatch group count and element count are GPU-resident.
        executeSortIndirectCount(uniforms,
                                 buffers,
                                 32,
                                 buffers.visible_count.deviceBuffer,
                                 buffers.visible_sort_dispatch_args.deviceBuffer,
                                 num_splats);
    }

    // Snapshot depth-ranked primitive indices into a stable buffer so stage 2
    // is free to reuse the ping-pong without clobbering the ordering. Matches
    // the CUDA reference's `primitive_indices_sorted` view.
    {
        PerfTimer::Timer<PerfTimer::CopyPrimitiveSortIndices> gpu_timer(this);
        [[maybe_unused]] auto cpu_timer =
            timeCpuStage("vksplat.render.record.executeSortPrimitivesByDepth.copy_primitive_sort_indices");
        auto& sort_indices = resizeDeviceBuffer(buffers.primitive_sort_indices, num_splats);
        bufferMemoryBarrier({{buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE}},
                            TRANSFER_READ);
        bufferMemoryBarrier({{sort_indices, COMPUTE_SHADER_READ}},
                            TRANSFER_WRITE);
        VkBufferCopy copy{};
        copy.srcOffset = buffers.sorted_gauss_idx().deviceBuffer.offset;
        copy.dstOffset = sort_indices.offset;
        copy.size = num_splats * sizeof(int32_t);
        vkCmdCopyBuffer(command_buffer,
                        buffers.sorted_gauss_idx().deviceBuffer.buffer,
                        sort_indices.buffer, 1, &copy);
        bufferMemoryBarrier({{sort_indices, TRANSFER_WRITE}},
                            COMPUTE_SHADER_READ);
    }
}

void VulkanGSRenderer::executeApplyDepthOrdering(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::ApplyDepthOrdering> timer(this);
    DEVICE_GUARD;

    const size_t num_splats = buffers.num_splats;
    if (num_splats == 0)
        return;

    auto& tiles_touched_ordered =
        resizeDeviceBuffer(buffers.tiles_touched_depth_ordered, num_splats);

    bufferMemoryBarrier({{buffers.primitive_sort_indices.deviceBuffer, TRANSFER_WRITE},
                         {buffers.tiles_touched.deviceBuffer, COMPUTE_SHADER_WRITE},
                         {buffers.visible_count.deviceBuffer, COMPUTE_SHADER_WRITE}},
                        COMPUTE_SHADER_READ);

    struct ApplyUniforms {
        uint32_t num_splats;
        uint32_t pad0, pad1, pad2;
    } apply_uniforms{static_cast<uint32_t>(num_splats), 0, 0, 0};

    executeCompute(
        {{num_splats, 64}},
        &apply_uniforms, sizeof(apply_uniforms),
        pipeline_apply_depth_ordering,
        {
            buffers.primitive_sort_indices.deviceBuffer,
            buffers.tiles_touched.deviceBuffer,
            tiles_touched_ordered,
            buffers.visible_count.deviceBuffer,
        });
}
