#include "diagnostics/vram_profiler.hpp"
#include "gs_renderer.h"
#include <limits>
#include <string>

size_t VulkanGSPipelineBuffers::getTotalOwnedAllocSize() const {
    size_t total = 0;
#define ADD_OWNED(name)                                           \
    do {                                                          \
        if (this->name.deviceBuffer.allocation != VK_NULL_HANDLE) \
            total += this->name.deviceBuffer.allocSize;           \
    } while (false)

    ADD_OWNED(xyz_ws);
    ADD_OWNED(sh_coeffs);
    ADD_OWNED(rotations);
    ADD_OWNED(scales_opacs);
    ADD_OWNED(sh0);
    ADD_OWNED(shN);
    ADD_OWNED(scaling_raw);
    ADD_OWNED(opacity_raw);
    ADD_OWNED(page_frames);
    ADD_OWNED(tiles_touched);
    ADD_OWNED(rect_tile_space);
    ADD_OWNED(radii);
    ADD_OWNED(xy_vs);
    ADD_OWNED(depths);
    ADD_OWNED(inv_cov_vs_opacity);
    ADD_OWNED(rgb);
    ADD_OWNED(overlay_flags);
    ADD_OWNED(primitive_depth_keys);
    ADD_OWNED(lod_indices);
    ADD_OWNED(lod_logical_indices);
    ADD_OWNED(lod_levels);
    ADD_OWNED(lod_weights);
    ADD_OWNED(lod_gpu_indices);
    ADD_OWNED(lod_gpu_logical_indices);
    ADD_OWNED(lod_gpu_weights);
    ADD_OWNED(lod_gpu_counts);
    ADD_OWNED(lod_chunk_touch);
    ADD_OWNED(lod_compact_counts);
    ADD_OWNED(lod_compact_protected);
    ADD_OWNED(lod_compact_misses);
    ADD_OWNED(lod_gpu_levels);
    ADD_OWNED(primitive_sort_indices);
    ADD_OWNED(tiles_touched_depth_ordered);
    ADD_OWNED(visible_flags);
    ADD_OWNED(visible_prefix);
    ADD_OWNED(visible_count);
    ADD_OWNED(visible_sort_dispatch_args);
    ADD_OWNED(survivors);
    ADD_OWNED(survivor_state);
    ADD_OWNED(visible_emit_count);
    ADD_OWNED(orig_ids);
    ADD_OWNED(cumsum_counts);
    ADD_OWNED(visible_dispatch);
    ADD_OWNED(macro_partials);
    ADD_OWNED(macro_active_mask);
    ADD_OWNED(macro_wave_args);
    ADD_OWNED(index_buffer_offset);
    ADD_OWNED(sorting_keys_1);
    ADD_OWNED(sorting_keys_2);
    ADD_OWNED(sorting_gauss_idx_1);
    ADD_OWNED(sorting_gauss_idx_2);
    ADD_OWNED(tile_sort_count);
    ADD_OWNED(tile_sort_dispatch_args);
    ADD_OWNED(tile_ranges);
    ADD_OWNED(tile_batch_counts);
    ADD_OWNED(tile_batch_offsets);
    ADD_OWNED(tile_batch_dispatch_args);
    ADD_OWNED(tile_batch_descriptors);
    ADD_OWNED(tile_batch_pixel_state);
    ADD_OWNED(tile_batch_n_contributors);
    ADD_OWNED(pixel_state);
    ADD_OWNED(pixel_depth);
    ADD_OWNED(n_contributors);
    ADD_OWNED(_cumsum_blockSums);
    ADD_OWNED(_cumsum_blockSums2);
    ADD_OWNED(_sorting_histogram);
    ADD_OWNED(_sorting_histogram_cumsum);

#undef ADD_OWNED
    return total;
}

std::map<std::string, size_t> VulkanGSPipelineBuffers::getOwnedVramBreakdown() const {
    std::map<std::string, size_t> breakdown;
#define ADD_OWNED(name)                                            \
    do {                                                           \
        if (this->name.deviceBuffer.allocation != VK_NULL_HANDLE)  \
            breakdown[#name] += this->name.deviceBuffer.allocSize; \
    } while (false)

    ADD_OWNED(xyz_ws);
    ADD_OWNED(sh_coeffs);
    ADD_OWNED(rotations);
    ADD_OWNED(scales_opacs);
    ADD_OWNED(sh0);
    ADD_OWNED(shN);
    ADD_OWNED(scaling_raw);
    ADD_OWNED(opacity_raw);
    ADD_OWNED(page_frames);
    ADD_OWNED(tiles_touched);
    ADD_OWNED(rect_tile_space);
    ADD_OWNED(radii);
    ADD_OWNED(xy_vs);
    ADD_OWNED(depths);
    ADD_OWNED(inv_cov_vs_opacity);
    ADD_OWNED(rgb);
    ADD_OWNED(overlay_flags);
    ADD_OWNED(primitive_depth_keys);
    ADD_OWNED(lod_indices);
    ADD_OWNED(lod_logical_indices);
    ADD_OWNED(lod_levels);
    ADD_OWNED(lod_weights);
    ADD_OWNED(lod_gpu_indices);
    ADD_OWNED(lod_gpu_logical_indices);
    ADD_OWNED(lod_gpu_weights);
    ADD_OWNED(lod_gpu_counts);
    ADD_OWNED(lod_chunk_touch);
    ADD_OWNED(lod_compact_counts);
    ADD_OWNED(lod_compact_protected);
    ADD_OWNED(lod_compact_misses);
    ADD_OWNED(lod_gpu_levels);
    ADD_OWNED(primitive_sort_indices);
    ADD_OWNED(tiles_touched_depth_ordered);
    ADD_OWNED(visible_flags);
    ADD_OWNED(visible_prefix);
    ADD_OWNED(visible_count);
    ADD_OWNED(visible_sort_dispatch_args);
    ADD_OWNED(survivors);
    ADD_OWNED(survivor_state);
    ADD_OWNED(visible_emit_count);
    ADD_OWNED(orig_ids);
    ADD_OWNED(cumsum_counts);
    ADD_OWNED(visible_dispatch);
    ADD_OWNED(macro_partials);
    ADD_OWNED(macro_active_mask);
    ADD_OWNED(macro_wave_args);
    ADD_OWNED(index_buffer_offset);
    ADD_OWNED(sorting_keys_1);
    ADD_OWNED(sorting_keys_2);
    ADD_OWNED(sorting_gauss_idx_1);
    ADD_OWNED(sorting_gauss_idx_2);
    ADD_OWNED(tile_sort_count);
    ADD_OWNED(tile_sort_dispatch_args);
    ADD_OWNED(tile_ranges);
    ADD_OWNED(tile_batch_counts);
    ADD_OWNED(tile_batch_offsets);
    ADD_OWNED(tile_batch_dispatch_args);
    ADD_OWNED(tile_batch_descriptors);
    ADD_OWNED(tile_batch_pixel_state);
    ADD_OWNED(tile_batch_n_contributors);
    ADD_OWNED(pixel_state);
    ADD_OWNED(pixel_depth);
    ADD_OWNED(n_contributors);
    ADD_OWNED(_cumsum_blockSums);
    ADD_OWNED(_cumsum_blockSums2);
    ADD_OWNED(_sorting_histogram);
    ADD_OWNED(_sorting_histogram_cumsum);

#undef ADD_OWNED
    return breakdown;
}

void VulkanGSPipeline::allocStagingBuffer(size_t size) {
    if (size == 0) {
        _THROW_ERROR(std::format(
            "allocStagingBuffer requires a non-zero allocation (requested_bytes={}, existing_buffer={:#x}, existing_bytes={})",
            size,
            lfs::rendering::vkHandleValue(stager.buffer),
            stager.allocSize));
    }
    if (stager.buffer != VK_NULL_HANDLE && stager.allocSize >= size)
        return;

    std::lock_guard<std::mutex> lock(stager.mutex);

    if (stager.allocSize < size) {
        HOST_GUARD;
        waitForPendingBatch();
        if (stager.buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator, stager.buffer, stager.allocation);
        }
        stager.buffer = VK_NULL_HANDLE;
        stager.allocation = VK_NULL_HANDLE;
        stager.allocSize = 0;
    }

    VkBufferCreateInfo staging_info = {};
    staging_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    staging_info.size = size;
    staging_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    staging_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci = {};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    const VkResult result = vmaCreateBuffer(
        allocator, &staging_info, &aci, &stager.buffer, &stager.allocation, nullptr);
    if (result != VK_SUCCESS) {
        stager.buffer = VK_NULL_HANDLE;
        stager.allocation = VK_NULL_HANDLE;
        _THROW_ERROR(std::format(
            "VkSplat staging-buffer allocation failed (requested_bytes={}, allocator={:#x}, usage={:#x}, result={}({}))",
            size,
            lfs::rendering::vkHandleValue(allocator),
            static_cast<std::uint32_t>(staging_info.usage),
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)));
    }

    stager.allocSize = size;
    setDebugObjectName(VK_OBJECT_TYPE_BUFFER, stager.buffer, "vksplat.staging");
}

void VulkanGSPipeline::createBuffer(size_t size, _VulkanBuffer& buffer) {
    if (size == 0 || buffer.buffer != VK_NULL_HANDLE || buffer.allocation != VK_NULL_HANDLE) {
        _THROW_ERROR(std::format(
            "createBuffer requires a non-zero size and an empty destination (requested_bytes={}, existing_buffer={:#x}, existing_allocation={:#x}, label='{}')",
            size,
            lfs::rendering::vkHandleValue(buffer.buffer),
            lfs::rendering::vkHandleValue(buffer.allocation),
            buffer.label ? buffer.label : "<unlabeled>"));
    }
    buffer.allocSize = size;
    buffer.capacity = size;
    buffer.size = size;
    buffer.offset = 0;

    VkBufferCreateInfo buffer_info = {};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci = {};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    const VkResult result =
        vmaCreateBuffer(allocator, &buffer_info, &aci, &buffer.buffer, &buffer.allocation, nullptr);
    if (result != VK_SUCCESS) {
        buffer.buffer = VK_NULL_HANDLE;
        buffer.allocation = VK_NULL_HANDLE;
        buffer.allocSize = 0;
        buffer.capacity = 0;
        buffer.size = 0;
        buffer.offset = 0;
        _THROW_ERROR(std::format(
            "VkSplat device-buffer allocation failed (requested_bytes={}, allocator={:#x}, usage={:#x}, label='{}', result={}({}))",
            size,
            lfs::rendering::vkHandleValue(allocator),
            static_cast<std::uint32_t>(buffer_info.usage),
            buffer.label ? buffer.label : "<unlabeled>",
            lfs::rendering::vkResultToString(result),
            static_cast<int>(result)));
    }

    if (buffer.label && debug_name_writer_.enabled()) {
        setDebugObjectName(VK_OBJECT_TYPE_BUFFER,
                           buffer.buffer,
                           std::format("vksplat.buffer.{}", buffer.label));
    }

    current_vram += size;
    if (current_vram > peak_vram)
        peak_vram = current_vram;

    // Publish per-buffer live bytes so the HUD can split the Vulkan footprint into
    // named rows (xyz_ws / shN / sorting_keys / ...). nullptr label = no instrumentation.
    if (buffer.label) {
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
            std::string("vksplat.") + buffer.label, "device_buffer", size);
    }
}

void VulkanGSPipeline::destroyBuffer(_VulkanBuffer& buffer) {
    if (commandBatchInProgress) {
        _THROW_ERROR(std::format(
            "destroyBuffer cannot destroy an allocation referenced by an active command batch (batch_active={}, buffer={:#x}, allocation={:#x}, bytes={}, label='{}')",
            commandBatchInProgress,
            lfs::rendering::vkHandleValue(buffer.buffer),
            lfs::rendering::vkHandleValue(buffer.allocation),
            buffer.allocSize,
            buffer.label ? buffer.label : "<unlabeled>"));
    }
    if (buffer.buffer != VK_NULL_HANDLE && buffer.allocation != VK_NULL_HANDLE) {
        waitForPendingBatch();
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
        if (current_vram < buffer.allocSize) {
            _THROW_ERROR(std::format(
                "VkSplat VRAM accounting underflowed while destroying a buffer (tracked_bytes={}, buffer_bytes={}, buffer={:#x}, label='{}')",
                current_vram,
                buffer.allocSize,
                lfs::rendering::vkHandleValue(buffer.buffer),
                buffer.label ? buffer.label : "<unlabeled>"));
        }
        current_vram -= buffer.allocSize;
        if (buffer.label) {
            lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                std::string("vksplat.") + buffer.label, "device_buffer", 0);
        }
    }
    buffer.buffer = VK_NULL_HANDLE;
    buffer.allocation = VK_NULL_HANDLE;
    buffer.allocSize = 0;
    buffer.capacity = 0;
    buffer.size = 0;
    buffer.offset = 0;
    // Keep buffer.label intact so a subsequent resize re-establishes the recording.
}

void VulkanGSPipeline::resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink) {
    if (deviceBuffer.capacity < new_byte_size || (!no_shrink && deviceBuffer.capacity > new_byte_size)) {
        HOST_GUARD;
        destroyBuffer(deviceBuffer);
        try {
            createBuffer(new_byte_size, deviceBuffer);
        } catch (const std::runtime_error& err) {
            _THROW_ERROR(std::string(err.what()) + ". createBuffer failed inside resizeDeviceBuffer");
        }
    }
    deviceBuffer.size = new_byte_size;
}

template <typename T>
_VulkanBuffer& VulkanGSPipeline::resizeDeviceBuffer(Buffer<T>& buffer, size_t new_size, bool no_shrink) {
    auto& deviceBuffer = buffer.deviceBuffer;
    if (new_size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        _THROW_ERROR(std::format(
            "resizeDeviceBuffer element count overflows byte sizing (elements={}, element_bytes={}, max_elements={}, label='{}')",
            new_size,
            sizeof(T),
            std::numeric_limits<size_t>::max() / sizeof(T),
            deviceBuffer.label ? deviceBuffer.label : "<unlabeled>"));
    }
    size_t new_byte_size = new_size * sizeof(T);
    resizeDeviceBuffer(deviceBuffer, new_byte_size, no_shrink);
    return deviceBuffer;
}

template <typename T>
_VulkanBuffer& VulkanGSPipeline::clearDeviceBuffer(Buffer<T>& buffer, size_t new_size) {
    auto& deviceBuffer = buffer.deviceBuffer;
    const size_t new_byte_size = new_size * sizeof(T);
    // Clearing is a GPU operation; changing the active view size must not force a
    // host-side submit/wait when the existing allocation is already large enough.
    if (deviceBuffer.capacity < new_byte_size) {
        resizeDeviceBuffer(buffer, new_size, true);
    } else {
        deviceBuffer.size = new_byte_size;
    }

    if (deviceBuffer.size == 0)
        return deviceBuffer;

    {
        DEVICE_GUARD;
        validateFillRange(deviceBuffer, 0, deviceBuffer.size, "clearDeviceBuffer");
        vkCmdFillBuffer(command_buffer, deviceBuffer.buffer, deviceBuffer.offset, deviceBuffer.size, 0);
    }

    return deviceBuffer;
}

template <typename T>
_VulkanBuffer& VulkanGSPipeline::resizeAndCopyDeviceBuffer(
    Buffer<T>& buffer,
    size_t new_size,
    bool clear) {
    auto& deviceBuffer = buffer.deviceBuffer;

    if (new_size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        _THROW_ERROR(std::format(
            "resizeAndCopyDeviceBuffer element count overflows byte sizing (elements={}, element_bytes={}, max_elements={}, label='{}')",
            new_size,
            sizeof(T),
            std::numeric_limits<size_t>::max() / sizeof(T),
            deviceBuffer.label ? deviceBuffer.label : "<unlabeled>"));
    }
    size_t new_byte_size = new_size * sizeof(T);
    size_t old_byte_size = deviceBuffer.size;

    if (new_size <= deviceBuffer.capacity / sizeof(T)) {
        deviceBuffer.size = new_byte_size;

        if (clear && new_byte_size > old_byte_size) {
            VkDeviceSize offset = old_byte_size;
            VkDeviceSize size = new_byte_size - old_byte_size;

            VkDeviceSize alignedOffset = (offset + 3) & ~3ULL;
            VkDeviceSize prefix = alignedOffset - offset;
            if (prefix < size) {
                offset = alignedOffset;
                size -= prefix;
                DEVICE_GUARD;
                validateFillRange(deviceBuffer, offset, size, "resizeAndCopyDeviceBuffer tail clear");
                vkCmdFillBuffer(command_buffer, deviceBuffer.buffer, deviceBuffer.offset + offset, size, 0u);
                HOST_GUARD; // will apply fence
            }
        }

        return deviceBuffer;
    }

    _VulkanBuffer newBuffer;
    newBuffer.label = deviceBuffer.label;
    try {
        createBuffer(new_byte_size, newBuffer);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) +
                     ". createBuffer failed inside resizeAndCopyDeviceBuffer");
    }

    {
        DEVICE_GUARD;

        if (deviceBuffer.buffer != VK_NULL_HANDLE && old_byte_size > 0) {
            validateBufferRange(deviceBuffer, 0, old_byte_size, "resizeAndCopyDeviceBuffer source copy");
            validateBufferRange(newBuffer, 0, old_byte_size, "resizeAndCopyDeviceBuffer destination copy");
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = deviceBuffer.offset;
            copyRegion.dstOffset = 0;
            copyRegion.size = old_byte_size;

            vkCmdCopyBuffer(
                command_buffer,
                deviceBuffer.buffer,
                newBuffer.buffer,
                1,
                &copyRegion);
        }

        if (clear && old_byte_size < new_byte_size) {
            VkDeviceSize offset = old_byte_size;
            VkDeviceSize size = new_byte_size - old_byte_size;

            VkDeviceSize alignedOffset = (offset + 3) & ~3ULL;
            VkDeviceSize prefix = alignedOffset - offset;
            if (prefix < size) {
                offset = alignedOffset;
                size -= prefix;

                validateFillRange(newBuffer, offset, size, "resizeAndCopyDeviceBuffer new tail clear");
                vkCmdFillBuffer(
                    command_buffer,
                    newBuffer.buffer,
                    newBuffer.offset + offset,
                    size,
                    0u);
            }
        }
    }

    HOST_GUARD;
    destroyBuffer(deviceBuffer);
    deviceBuffer = newBuffer;
    deviceBuffer.size = new_byte_size;

    return deviceBuffer;
}

template <typename T>
void VulkanGSPipelineBuffers::reorderSH(Buffer<T>& coeffs) {
    if (SH_REORDER_SIZE <= 1)
        return;

    static constexpr size_t SH_DIM = 12;

    coeffs.resize(_CEIL_ROUND(coeffs.size(), 4 * SH_DIM * SH_REORDER_SIZE), T(0.0));

    auto forwardIndex = [=](size_t i) {
        size_t group_idx = i / (SH_DIM * SH_REORDER_SIZE);
        size_t gauss_idx = (i / SH_DIM) % SH_REORDER_SIZE;
        size_t sh_idx = i % SH_DIM;
        return (group_idx * SH_DIM + sh_idx) * SH_REORDER_SIZE + gauss_idx;
    };

    typedef struct {
        T _[4];
    } __m128;
    __m128* sh = reinterpret_cast<__m128*>(coeffs.data());

    size_t n = coeffs.size() / 4;

    // TODO: do this in O(1) additional memory
    std::vector<__m128> sh_copy(sh, sh + n);
    for (size_t i = 0; i < n; i++) {
        LFS_VK_DEBUG_ASSERT(
            forwardIndex(i) < n,
            "SH reorder index must stay inside the packed coefficient array (source_index={}, destination_index={}, packed_count={}, sh_dimension={}, reorder_width={})",
            i,
            forwardIndex(i),
            n,
            SH_DIM,
            SH_REORDER_SIZE);
        sh[forwardIndex(i)] = sh_copy[i];
    }
}

template <typename T>
void VulkanGSPipelineBuffers::undoReorderSH(Buffer<T>& coeffs, size_t num_splats) {
    if (SH_REORDER_SIZE <= 1)
        return;

    static constexpr size_t SH_DIM = 12;

    coeffs.resize(4 * SH_DIM * _CEIL_ROUND(num_splats, SH_REORDER_SIZE), T(0.0));

    auto forwardIndex = [=](size_t i) {
        size_t group_idx = i / (SH_DIM * SH_REORDER_SIZE);
        size_t gauss_idx = (i / SH_DIM) % SH_REORDER_SIZE;
        size_t sh_idx = i % SH_DIM;
        return (group_idx * SH_DIM + sh_idx) * SH_REORDER_SIZE + gauss_idx;
    };

    typedef struct {
        T _[4];
    } __m128;
    __m128* sh = reinterpret_cast<__m128*>(coeffs.data());

    size_t n = coeffs.size() / 4;

    // TODO: do this in O(1) additional memory
    std::vector<__m128> sh_copy(sh, sh + n);
    for (size_t i = 0; i < n; i++) {
        LFS_VK_DEBUG_ASSERT(
            forwardIndex(i) < n,
            "SH inverse-reorder index must stay inside the packed coefficient array (destination_index={}, source_index={}, packed_count={}, sh_dimension={}, reorder_width={})",
            i,
            forwardIndex(i),
            n,
            SH_DIM,
            SH_REORDER_SIZE);
        sh[i] = sh_copy[forwardIndex(i)];
    }

    coeffs.resize(4 * SH_DIM * num_splats);
}

void VulkanGSPipelineBuffers::assignScalesOpacs(
    Buffer<float>& scales_opacs,
    size_t n, const float* scales, const float* opacs) {
    scales_opacs.resize(4 * n);
    for (size_t i = 0; i < n; i++) {
        float* so = &scales_opacs[4 * i];
        so[0] = scales[3 * i];
        so[1] = scales[3 * i + 1];
        so[2] = scales[3 * i + 2];
        so[3] = opacs[i];
    }
}

#define _INSTANTIATE_BUFFER(dtype)                                                                                           \
    template _VulkanBuffer& VulkanGSPipeline::resizeDeviceBuffer(Buffer<dtype>& buffer, size_t new_size, bool no_shrink);    \
    template _VulkanBuffer& VulkanGSPipeline::clearDeviceBuffer(Buffer<dtype>& buffer, size_t new_size);                     \
    template _VulkanBuffer& VulkanGSPipeline::resizeAndCopyDeviceBuffer(Buffer<dtype>& buffer, size_t new_size, bool clear); \
    template void VulkanGSPipelineBuffers::reorderSH(Buffer<dtype>& coeffs);                                                 \
    template void VulkanGSPipelineBuffers::undoReorderSH(Buffer<dtype>& coeffs, size_t num_splats);

_INSTANTIATE_BUFFER(uint8_t)
_INSTANTIATE_BUFFER(uint16_t)
_INSTANTIATE_BUFFER(float)
_INSTANTIATE_BUFFER(int32_t)
_INSTANTIATE_BUFFER(int64_t)
_INSTANTIATE_BUFFER(uint32_t)

#undef _INSTANTIATE_BUFFER
