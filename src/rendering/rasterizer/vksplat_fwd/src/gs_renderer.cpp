#include "gs_renderer.h"

#include <csignal>
#include <fstream>
#include <memory>

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
    destroyNumIndicesReadback();
    cleanup();
}

void VulkanGSRenderer::cleanup() {
    destroyNumIndicesReadback();
    VulkanGSPipeline::cleanup();
}

void VulkanGSRenderer::resetNumIndicesEstimate() {
    num_indices_estimate_ = 0;
    num_indices_readback_pending_ = false;
}

void VulkanGSRenderer::ensureNumIndicesReadback() {
    if (num_indices_readback_initialized_)
        return;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = sizeof(int32_t);
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo alloc_info{};
    if (vmaCreateBuffer(allocator, &info, &aci,
                        &num_indices_readback_buffer_.buffer,
                        &num_indices_readback_buffer_.allocation,
                        &alloc_info) != VK_SUCCESS) {
        num_indices_readback_buffer_.buffer = VK_NULL_HANDLE;
        num_indices_readback_buffer_.allocation = VK_NULL_HANDLE;
        _THROW_ERROR("Failed to allocate num_indices readback buffer");
    }
    num_indices_readback_buffer_.allocSize = sizeof(int32_t);
    num_indices_readback_buffer_.size = sizeof(int32_t);
    num_indices_readback_mapped_ = static_cast<int32_t*>(alloc_info.pMappedData);
    if (num_indices_readback_mapped_)
        *num_indices_readback_mapped_ = 0;
    num_indices_readback_initialized_ = true;
    num_indices_readback_pending_ = false;
}

void VulkanGSRenderer::destroyNumIndicesReadback() {
    if (!num_indices_readback_initialized_)
        return;
    if (num_indices_readback_buffer_.buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, num_indices_readback_buffer_.buffer,
                         num_indices_readback_buffer_.allocation);
    }
    num_indices_readback_buffer_ = {};
    num_indices_readback_mapped_ = nullptr;
    num_indices_readback_initialized_ = false;
    num_indices_readback_pending_ = false;
}

size_t VulkanGSRenderer::pollDeferredNumIndices() {
    // The previous frame's submit ended with endCommandBatch(true) → fence-wait
    // (gs_pipeline.cpp). So by the time the next frame enters this function,
    // any prior vkCmdCopyBuffer into num_indices_readback_buffer_ has retired
    // and the mapped value is observable.
    if (!num_indices_readback_pending_ || !num_indices_readback_mapped_)
        return 0;
    const int32_t value = *num_indices_readback_mapped_;
    num_indices_readback_pending_ = false;
    return value < 0 ? 0u : static_cast<size_t>(value);
}

void VulkanGSRenderer::initializeExternal(const std::map<std::string, std::string>& spirv_paths,
                                          VkInstance external_instance,
                                          VkPhysicalDevice external_physical_device,
                                          VkDevice external_device,
                                          VkQueue external_queue,
                                          uint32_t external_queue_family_index,
                                          VmaAllocator external_allocator) {
    VulkanGSPipeline::initializeExternal(
        external_instance,
        external_physical_device,
        external_device,
        external_queue,
        external_queue_family_index,
        external_allocator);

    createComputePipeline(pipeline_projection_forward, spirv_paths.at("projection_forward"));
    createComputePipeline(pipeline_generate_keys, spirv_paths.at("generate_keys"));
    for (int i = 0; i < 2; ++i) {
        createComputePipeline(pipeline_compute_tile_ranges[i], spirv_paths.at("compute_tile_ranges"));
        createComputePipeline(pipeline_rasterize_forward[i], spirv_paths.at("rasterize_forward"));
    }
    createComputePipeline(pipeline_setup_dispatch_indirect,
                          spirv_paths.at("setup_dispatch_indirect"));
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
}

void VulkanGSRenderer::executeProjectionForward(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    size_t alloc_reserve) {
    PerfTimer::Timer<PerfTimer::ProjectionForward> timer(this);
    DEVICE_GUARD;

    size_t num_splats = buffers.num_splats;

    bufferMemoryBarrier({
                            {buffers.xyz_ws.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.sh_coeffs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.rotations.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                            {buffers.scales_opacs.deviceBuffer, TRANSFER_COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    size_t alloc_size = std::max(num_splats, alloc_reserve);
    executeCompute(
        {{num_splats, SUBGROUP_SIZE}},
        &uniforms, sizeof(uniforms),
        pipeline_projection_forward,
        {
            // inputs
            buffers.xyz_ws.deviceBuffer,
            buffers.sh_coeffs.deviceBuffer,
            buffers.rotations.deviceBuffer,
            buffers.scales_opacs.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.tiles_touched, alloc_size),
            resizeDeviceBuffer(buffers.rect_tile_space, alloc_size),
            resizeDeviceBuffer(buffers.radii, alloc_size),
            resizeDeviceBuffer(buffers.xy_vs, 2 * alloc_size),
            resizeDeviceBuffer(buffers.depths, alloc_size),
            resizeDeviceBuffer(buffers.inv_cov_vs_opacity, 4 * alloc_size),
            resizeDeviceBuffer(buffers.rgb, 3 * alloc_size),
        });
}

void VulkanGSRenderer::executeGenerateKeys(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::GenerateKeys> timer(this);
    DEVICE_GUARD;

    const size_t num_elements = buffers.num_splats;
    // num_indices here is the deferred-readback high-water-mark estimate, not the
    // exact GPU value. unsorted_keys/unsorted_gauss_idx are sized to it (grow-only
    // via no_shrink default). The shader's actual writes are still bounded by
    // index_buffer_offset[i] on the GPU — never past the true num_indices.
    const size_t capacity = buffers.num_indices;

    auto& unsorted_keys = resizeDeviceBuffer(buffers.unsorted_keys(), capacity);
    auto& unsorted_idx = resizeDeviceBuffer(buffers.unsorted_gauss_idx(), capacity);

    // Pre-fill unsorted_keys with the max sentinel (0xFFFFFFFF) so that any tail
    // entries [actual_num_indices, capacity) sort to the end of the radix sort and
    // produce empty tile ranges in compute_tile_ranges (whose num_isects is read
    // from the GPU-resident cumsum tail, not the CPU estimate). This gives us
    // capacity-bounded buffers with correct sort output regardless of overestimate.
    bufferMemoryBarrier({{unsorted_keys, COMPUTE_SHADER_READ_WRITE}},
                        TRANSFER_COMPUTE_SHADER_WRITE);
    vkCmdFillBuffer(command_buffer, unsorted_keys.buffer, 0, unsorted_keys.size,
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
            buffers.depths.deviceBuffer,
            buffers.rect_tile_space.deviceBuffer,
            buffers.index_buffer_offset.deviceBuffer,
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

    // Indirect dispatch — group count is filled on GPU by setup_dispatch_indirect.
    // The shader reads num_isects from index_buffer_offset[num_splats-1] directly,
    // so no CPU readback is needed here.
    executeComputeIndirect(
        buffers.dispatch_indirect_compute_tile_ranges.deviceBuffer,
        0,
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
    VulkanGSPipelineBuffers& buffers) {
    if (buffers.num_indices == 0)
        return;

    PerfTimer::Timer<PerfTimer::RasterizeForward> timer(this);
    DEVICE_GUARD;

    size_t num_pixels = uniforms.image_height * uniforms.image_width;

    bufferMemoryBarrier({
                            {buffers.sorted_gauss_idx().deviceBuffer, COMPUTE_SHADER_WRITE},
                            {buffers.tile_ranges.deviceBuffer, COMPUTE_SHADER_WRITE},
                            {buffers.rgb.deviceBuffer, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    executeCompute(
        {{uniforms.image_width, TILE_WIDTH}, {uniforms.image_height, TILE_HEIGHT}},
        &uniforms, sizeof(uniforms),
        pipeline_rasterize_forward[buffers.is_unsorted_1],
        std::vector<_VulkanBuffer>({
            // inputs
            buffers.sorted_gauss_idx().deviceBuffer,
            buffers.tile_ranges.deviceBuffer,
            buffers.xy_vs.deviceBuffer,
            buffers.inv_cov_vs_opacity.deviceBuffer,
            buffers.rgb.deviceBuffer,
            // outputs
            resizeDeviceBuffer(buffers.pixel_state, 4 * num_pixels),
            resizeDeviceBuffer(buffers.n_contributors, num_pixels),
        }));
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

    uint32_t uniforms[2] = {
        (uint32_t)num_elements, 1};
    // int uniform_size = 2*sizeof(uint32_t);
    int uniform_size = 1 * sizeof(uint32_t);

    bufferMemoryBarrier({
                            {input_buffer.deviceBuffer, COMPUTE_SHADER_WRITE},
                        },
                        COMPUTE_SHADER_READ);

    resizeDeviceBuffer(output_buffer, num_elements);

    if (num_elements <= block_0) {
        executeCompute(
            {{num_elements, block_0}},
            uniforms, uniform_size,
            pipeline_cumsum.single_pass,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
            });
    }

    else if (num_elements <= block * block) {
        resizeDeviceBuffer(buffers._cumsum_blockSums, _CEIL_DIV(num_elements, block), true);

        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
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
        executeCompute(
            {{num_elements / block, block}},
            uniforms, uniform_size,
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
        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
            pipeline_cumsum.add_block_offsets,
            {
                input_buffer.deviceBuffer,
                output_buffer.deviceBuffer,
                buffers._cumsum_blockSums.deviceBuffer,
            });
    }

    else if (num_elements <= block * block * block) {
        size_t num_elements_1 = _CEIL_DIV(num_elements, block);
        resizeDeviceBuffer(buffers._cumsum_blockSums, num_elements_1, true);
        resizeDeviceBuffer(buffers._cumsum_blockSums2, _CEIL_DIV(num_elements_1, block), true);

        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
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
        executeCompute(
            {{num_elements / block, block}},
            uniforms, uniform_size,
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
        executeCompute(
            {{num_elements_1 / block, block}},
            uniforms, uniform_size,
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
        executeCompute(
            {{num_elements / block, block}},
            uniforms, uniform_size,
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
        executeCompute(
            {{num_elements, block}},
            uniforms, uniform_size,
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
    VulkanGSPipelineBuffers& buffers) {
    PerfTimer::Timer<PerfTimer::CalculateIndexBufferOffset> timer(this);

    const size_t num_elements = buffers.num_splats;
    if (num_elements == 0) {
        buffers.num_indices = 0;
        return;
    }

    ensureNumIndicesReadback();

    // Read the previous frame's deferred num_indices (now safe — last frame ended
    // with endCommandBatch(true) → fence-wait, so the host copy is observable).
    const size_t observed = pollDeferredNumIndices();
    if (observed > 0)
        num_indices_estimate_ = std::max(num_indices_estimate_, observed);

    // Cumsum populates index_buffer_offset on GPU.
    executeCumsum(
        buffers,
        buffers.tiles_touched,
        buffers.index_buffer_offset);

    DEVICE_GUARD;

    bufferMemoryBarrier({
                            {buffers.index_buffer_offset.deviceBuffer, COMPUTE_SHADER_READ_WRITE},
                        },
                        TRANSFER_COMPUTE_SHADER_READ);

    // Async copy of the cumsum tail into the host-visible buffer for the NEXT
    // frame's poll. No queue wait — the value is consumed one frame later.
    {
        VkBufferCopy copy{};
        copy.srcOffset = (num_elements - 1) * sizeof(int32_t);
        copy.dstOffset = 0;
        copy.size = sizeof(int32_t);
        vkCmdCopyBuffer(command_buffer,
                        buffers.index_buffer_offset.deviceBuffer.buffer,
                        num_indices_readback_buffer_.buffer, 1, &copy);
        num_indices_readback_pending_ = true;
    }

    // GPU-resident VkDispatchIndirectCommand for compute_tile_ranges. setup_dispatch_indirect
    // reads index_buffer_offset[num_splats-1] and computes ceil((n+1)/256) groups.
    auto& indirect_buf =
        resizeDeviceBuffer(buffers.dispatch_indirect_compute_tile_ranges, 3);
    bufferMemoryBarrier({{indirect_buf, COMPUTE_SHADER_READ_WRITE}}, COMPUTE_SHADER_WRITE);

    struct {
        uint32_t num_splats;
    } setup_uniforms{static_cast<uint32_t>(num_elements)};
    executeCompute(
        {{1, 1}},
        &setup_uniforms, sizeof(setup_uniforms),
        pipeline_setup_dispatch_indirect,
        {
            buffers.index_buffer_offset.deviceBuffer,
            indirect_buf,
        });

    bufferMemoryBarrier({{indirect_buf, COMPUTE_SHADER_WRITE}}, INDIRECT_DISPATCH_READ);

    // CPU-side high-water-mark estimate for sort-buffer sizing this frame.
    // Apply a 2x safety factor so a frame whose actual num_indices grows up to
    // 2x the prior peak does not overflow generate_keys' writes.
    constexpr size_t kSafetyFactor = 2;
    constexpr size_t kInitialIndicesPerSplat = 8; // first-frame heuristic
    size_t estimate = num_indices_estimate_ * kSafetyFactor;
    if (estimate == 0)
        estimate = num_elements * kInitialIndicesPerSplat;
    if (estimate < num_elements)
        estimate = num_elements;
    buffers.num_indices = estimate;
    buffers.num_indices_high_water = std::max(buffers.num_indices_high_water, estimate);
}

void VulkanGSRenderer::executeSort(
    const VulkanGSRendererUniforms& uniforms,
    VulkanGSPipelineBuffers& buffers,
    int num_bits) {
    PerfTimer::Timer<PerfTimer::SortRTS> timer(this);

    size_t num_elements = buffers.unsorted_keys().deviceSize();
    if (num_elements != buffers.unsorted_gauss_idx().deviceSize())
        _THROW_ERROR("number of elements don't match in executeSort");

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
    clearDeviceBuffer(globalHistogram, num_passes * sizeof(sortingKey_t) * RADIX);
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
