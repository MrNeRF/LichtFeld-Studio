#include "gs_renderer.h"

void VulkanGSPipeline::allocStagingBuffer(size_t size) {
    if (stager.buffer != VK_NULL_HANDLE && stager.allocSize >= size)
        return;

    std::lock_guard<std::mutex> lock(stager.mutex);

    if (stager.allocSize < size) {
        HOST_GUARD;
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

    if (vmaCreateBuffer(allocator, &staging_info, &aci, &stager.buffer, &stager.allocation, nullptr) != VK_SUCCESS) {
        stager.buffer = VK_NULL_HANDLE;
        stager.allocation = VK_NULL_HANDLE;
        throw std::runtime_error("Failed to allocate staging buffer memory. You are likely running out of RAM.");
    }

    stager.allocSize = size;
}

void VulkanGSPipeline::createBuffer(size_t size, _VulkanBuffer& buffer) {
    buffer.allocSize = size;
    buffer.size = size;

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

    if (vmaCreateBuffer(allocator, &buffer_info, &aci, &buffer.buffer, &buffer.allocation, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory. You are likely running out of VRAM.");
    }

    current_vram += size;
    if (current_vram > peak_vram)
        peak_vram = current_vram;
}

void VulkanGSPipeline::destroyBuffer(_VulkanBuffer& buffer) {
    if (commandBatchInProgress)
        _THROW_ERROR("destroyBuffer called when command batch in progress");
    if (buffer.buffer != VK_NULL_HANDLE && buffer.allocation != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
        buffer.buffer = VK_NULL_HANDLE;
        buffer.allocation = VK_NULL_HANDLE;
        if (current_vram < buffer.allocSize)
            _THROW_ERROR("Negative VRAM");
        current_vram -= buffer.allocSize;
    }
}

void VulkanGSPipeline::resizeDeviceBuffer(_VulkanBuffer& deviceBuffer, size_t new_byte_size, bool no_shrink) {
    if (deviceBuffer.allocSize < new_byte_size || (!no_shrink && deviceBuffer.allocSize > new_byte_size)) {
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
    size_t new_byte_size = new_size * sizeof(T);
    resizeDeviceBuffer(deviceBuffer, new_byte_size, no_shrink);
    return deviceBuffer;
}

template <typename T>
_VulkanBuffer& VulkanGSPipeline::clearDeviceBuffer(Buffer<T>& buffer, size_t new_size) {
    auto& deviceBuffer = buffer.deviceBuffer;
    if (deviceBuffer.size != new_size * sizeof(T)) {
        HOST_GUARD;
        resizeDeviceBuffer(buffer, new_size);
    }

    {
        DEVICE_GUARD;
        vkCmdFillBuffer(command_buffer, deviceBuffer.buffer, 0, deviceBuffer.size, 0);
    }

    return deviceBuffer;
}

template <typename T>
_VulkanBuffer& VulkanGSPipeline::resizeAndCopyDeviceBuffer(
    Buffer<T>& buffer,
    size_t new_size,
    bool clear) {
    auto& deviceBuffer = buffer.deviceBuffer;

    size_t new_byte_size = new_size * sizeof(T);
    size_t old_byte_size = deviceBuffer.size;

    if (new_size <= deviceBuffer.allocSize / sizeof(T)) {
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
                vkCmdFillBuffer(command_buffer, deviceBuffer.buffer, offset, size, 0u);
                HOST_GUARD; // will apply fence
            }
        }

        return deviceBuffer;
    }

    _VulkanBuffer newBuffer;
    try {
        createBuffer(new_byte_size, newBuffer);
    } catch (const std::runtime_error& err) {
        _THROW_ERROR(std::string(err.what()) +
                     ". createBuffer failed inside resizeAndCopyDeviceBuffer");
    }

    {
        DEVICE_GUARD;

        if (deviceBuffer.buffer != VK_NULL_HANDLE && old_byte_size > 0) {
            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
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

                vkCmdFillBuffer(
                    command_buffer,
                    newBuffer.buffer,
                    offset,
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
T VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index) {

    const size_t elementSize = sizeof(T);
    const size_t offset = index * elementSize;

    // Validate bounds
    if (offset + elementSize > buffer.size)
        _THROW_ERROR("Index out of bound while reading buffer element");

    T outValue;

    // Only need elementSize bytes; sizing the staging buffer to the full source buffer
    // (num_splats * 4 in the hot readback path) wasted a one-time MB-scale allocation.
    allocStagingBuffer(elementSize);
    {
        // std::lock_guard<std::mutex> lock(stager.mutex);
        {
            DEVICE_GUARD;

            // Copy only the specific element from device buffer to staging buffer.
            VkBufferCopy copyRegion = {};
            copyRegion.srcOffset = offset;
            copyRegion.dstOffset = 0;
            copyRegion.size = elementSize;

            vkCmdCopyBuffer(command_buffer, buffer.buffer, stager.buffer, 1, &copyRegion);
        }
        HOST_GUARD; // will apply fence

        // Map the staging buffer and read the specific element
        void* base;
        if (vmaMapMemory(allocator, stager.allocation, &base) != VK_SUCCESS) {
            _THROW_ERROR("Failed to map memory while reading buffer element");
        }

        memcpy(&outValue, base, elementSize);

        vmaUnmapMemory(allocator, stager.allocation);
    }

    return outValue;
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
    template dtype VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index);                                 \
    template void VulkanGSPipelineBuffers::reorderSH(Buffer<dtype>& coeffs);                                                 \
    template void VulkanGSPipelineBuffers::undoReorderSH(Buffer<dtype>& coeffs, size_t num_splats);

_INSTANTIATE_BUFFER(uint8_t)
_INSTANTIATE_BUFFER(float)
_INSTANTIATE_BUFFER(int32_t)
_INSTANTIATE_BUFFER(int64_t)

#undef _INSTANTIATE_BUFFER
