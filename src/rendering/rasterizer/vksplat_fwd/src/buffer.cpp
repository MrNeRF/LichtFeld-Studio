#include "gs_renderer.h"

size_t VulkanGSPipelineBuffers::getTotalAllocSize() {
    size_t total = 0;
#define _(name)                                     \
    {                                               \
        total += this->name.deviceBuffer.allocSize; \
    }
    _(xyz_ws)
    _(sh_coeffs)
    _(rotations)
    _(scales_opacs)
    _(tiles_touched)
    _(rect_tile_space)
    _(radii) _(xy_vs) _(depths) _(inv_cov_vs_opacity) _(rgb)
        _(index_buffer_offset) _(sorting_keys_1) _(sorting_keys_2) _(sorting_gauss_idx_1) _(sorting_gauss_idx_2) _(tile_ranges)
            _(pixel_state) _(n_contributors)
                _(_cumsum_blockSums) _(_cumsum_blockSums2) _(_sorting_histogram) _(_sorting_histogram_cumsum)
#undef _
                    return total;
}

void VulkanGSPipelineBuffers::updateTotalAllocSize() {
    size_t sz = getTotalAllocSize();
    if (sz > maxTotalAllocSize)
        maxTotalAllocSize = sz;
}

std::map<std::string, size_t> VulkanGSPipelineBuffers::getVramBreakdown() {
    std::map<std::string, size_t> breakdown;
#define _(name)                                                \
    {                                                          \
        breakdown[#name] += this->name.deviceBuffer.allocSize; \
    }
    _(xyz_ws)
    _(sh_coeffs)
    _(rotations)
    _(scales_opacs)
    _(tiles_touched)
    _(rect_tile_space)
    _(radii) _(xy_vs) _(depths) _(inv_cov_vs_opacity) _(rgb)
        _(index_buffer_offset) _(sorting_keys_1) _(sorting_keys_2) _(sorting_gauss_idx_1) _(sorting_gauss_idx_2) _(tile_ranges)
            _(pixel_state) _(n_contributors)
                _(_cumsum_blockSums) _(_cumsum_blockSums2) _(_sorting_histogram) _(_sorting_histogram_cumsum)
#undef _
                    return breakdown;
}

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
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
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
_VulkanBuffer& VulkanGSPipeline::copyToDevice(Buffer<T>& buffer) {

    resizeDeviceBuffer(buffer, buffer.size());
    auto& deviceBuffer = buffer.deviceBuffer;

    allocStagingBuffer(buffer.byteLength());
    {
        std::lock_guard<std::mutex> lock(stager.mutex);

        // Use staging buffer for device-local memory
        {
            HOST_GUARD;
            void* temp;
            if (vmaMapMemory(allocator, stager.allocation, &temp) != VK_SUCCESS)
                _THROW_ERROR("Failed to map staging buffer for upload");
            memcpy(temp, buffer.data(), buffer.byteLength());
            vmaUnmapMemory(allocator, stager.allocation);
        }

        // Copy from staging to device buffer
        DEVICE_GUARD;
        VkBufferCopy copy_region = {};
        copy_region.size = deviceBuffer.size;
        vkCmdCopyBuffer(command_buffer, stager.buffer, deviceBuffer.buffer, 1, &copy_region);
    }

    return deviceBuffer;
}

template <typename T>
void VulkanGSPipeline::copyFromDevice(Buffer<T>& buffer) {

    auto& deviceBuffer = buffer.deviceBuffer;
    buffer.resize(deviceBuffer.size / sizeof(T));
    if (buffer.empty())
        return;

    allocStagingBuffer(buffer.byteLength());
    {
        std::lock_guard<std::mutex> lock(stager.mutex);

        // Copy from device to staging buffer

        {
            DEVICE_GUARD;
            VkBufferCopy copy_region = {};
            copy_region.size = deviceBuffer.size;
            vkCmdCopyBuffer(command_buffer, deviceBuffer.buffer, stager.buffer, 1, &copy_region);
        }
        HOST_GUARD; // will apply fence

        // Read from staging buffer
        void* temp;
        if (vmaMapMemory(allocator, stager.allocation, &temp) != VK_SUCCESS)
            _THROW_ERROR("Failed to map staging buffer for readback");
        memcpy(reinterpret_cast<void*>(buffer.data()), temp, deviceBuffer.size);
        vmaUnmapMemory(allocator, stager.allocation);
    }
}

void VulkanGSPipeline::copyFromDeviceToDevice(const _VulkanBuffer& srcBuffer, _VulkanBuffer& dstBuffer) {

    if (srcBuffer.buffer == VK_NULL_HANDLE || srcBuffer.size == 0)
        _THROW_ERROR("Attempt to copy from empty buffer");

    const size_t elementCount = srcBuffer.size;
    resizeDeviceBuffer(dstBuffer, elementCount);
    if (dstBuffer.buffer == VK_NULL_HANDLE)
        _THROW_ERROR("Attempt to copy to empty buffer");

    DEVICE_GUARD;

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = srcBuffer.size;

    vkCmdCopyBuffer(command_buffer, srcBuffer.buffer, dstBuffer.buffer, 1, &copyRegion);
}

template <typename T>
void VulkanGSPipeline::copyFromDeviceToDevice(const Buffer<T>& src, Buffer<T>& dst) {
    const auto& srcBuffer = src.deviceBuffer;

    if (srcBuffer.buffer == VK_NULL_HANDLE || srcBuffer.size == 0)
        _THROW_ERROR("Attempt to copy from empty buffer");

    const size_t elementCount = srcBuffer.size / sizeof(T);
    resizeDeviceBuffer(dst, elementCount);
    auto& dstBuffer = dst.deviceBuffer;
    if (dstBuffer.buffer == VK_NULL_HANDLE)
        _THROW_ERROR("Attempt to copy to empty buffer");

    DEVICE_GUARD;

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = srcBuffer.size;

    vkCmdCopyBuffer(command_buffer, srcBuffer.buffer, dstBuffer.buffer, 1, &copyRegion);
}

template <typename T>
T VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index) {

    const size_t elementSize = sizeof(T);
    const size_t offset = index * elementSize;

    // Validate bounds
    if (offset + elementSize > buffer.size)
        _THROW_ERROR("Index out of bound while reading buffer element");

    T outValue;

    allocStagingBuffer(buffer.size);
    {
        // std::lock_guard<std::mutex> lock(stager.mutex);
        {
            DEVICE_GUARD;

            // Copy only the specific element from device buffer to staging buffer
            VkBufferCopy copyRegion = {};
            copyRegion.srcOffset = offset;
            copyRegion.dstOffset = offset; // Keep same offset in staging buffer
            copyRegion.size = elementSize;

            vkCmdCopyBuffer(command_buffer, buffer.buffer, stager.buffer, 1, &copyRegion);
        }
        HOST_GUARD; // will apply fence

        // Map the staging buffer and read the specific element
        void* base;
        if (vmaMapMemory(allocator, stager.allocation, &base) != VK_SUCCESS) {
            _THROW_ERROR("Failed to map memory while reading buffer element");
        }

        memcpy(&outValue, static_cast<const uint8_t*>(base) + offset, elementSize);

        vmaUnmapMemory(allocator, stager.allocation);
    }

    return outValue;
}

template <typename T>
void VulkanGSPipeline::dumpRawBytesToFile(Buffer<T>& buffer, std::string filename) {
    copyFromDevice(buffer);
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp)
        throw std::runtime_error("Error opening file `" + filename + "`");
    fwrite(buffer.data(), sizeof(T), buffer.size(), fp);
    fclose(fp);
}

#ifdef WIN32
void VulkanGSPipeline::_displayImage(Buffer<float>& pixels, int width, bool reverse_alpha) {
    _THROW_ERROR("Displaying image in terminal is not supported on Windows");
}
#else
#include <sys/ioctl.h>
#include <unistd.h>
void VulkanGSPipeline::_displayImage(Buffer<float>& pixels, int width, bool reverse_alpha) {

    copyFromDevice(pixels);

    const int aspect = 2;

    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    int window_width = std::max((int)std::min((int)w.ws_col, (int)w.ws_row * aspect) - 1, 0);
    window_width = std::min(window_width, width);
    int height = (int)pixels.size() / (4 * width);
    int window_height = (window_width * height) / (width * aspect);
    printf("%d x %d (display: %d x %d)\n", width, height, window_width, window_height);

    auto f2b = [](float x) -> uint8_t {
        return (uint8_t)(255.0 * fmin(fmax(x, 0.0f), 1.0f));
    };

    char buffer[256];
    std::string image;

    for (int y = 0; y < window_height; y++) {
        for (int x = 0; x < window_width; x++) {
            int x1 = (x * width) / window_width;
            int y1 = (y * height) / window_height;
            int idx = 4 * (y1 * width + x1);
            float* rgba = pixels.data() + idx;
            uint8_t alpha = reverse_alpha ? f2b(1.0f - rgba[3]) : f2b(rgba[3]);
            // printf("%d,", (int)alpha);
            sprintf(buffer,
                    "\033[38;2;%d;%d;%d;48;2;%d;%d;%dm%c\033[0m",
                    alpha, alpha, alpha,
                    f2b(rgba[0]), f2b(rgba[1]), f2b(rgba[2]),
                    std::isfinite(rgba[0] + rgba[1] + rgba[2] + rgba[3]) ? ' ' : 'X');
            image += buffer;
        }
        image += "\n";
    }
    // printf("\n");

    printf("%s", image.data());
    fflush(stdout);
}
#endif // WIN32, display image in terminal

float halfToFloat(uint16_t h) {
    // Extract sign, exponent, and fraction from half-precision
    uint16_t sign = (h >> 15) & 0x0001;
    uint16_t exp = (h >> 10) & 0x001F;
    uint16_t frac = h & 0x03FF;

    uint32_t sign32 = sign << 31;
    uint32_t exp32;
    uint32_t frac32;

    if (exp == 0) {
        if (frac == 0) {
            // Zero (+/-0)
            exp32 = 0;
            frac32 = 0;
        } else {
            // Subnormal half -> normalize it
            int shift = 0;
            while ((frac & 0x0400) == 0) { // until leading 1 appears
                frac <<= 1;
                shift++;
            }
            frac &= 0x03FF;               // remove the leading 1
            exp32 = 127 - 15 - shift + 1; // bias adjust (float32 bias=127, half bias=15)
            frac32 = frac << 13;
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        exp32 = 0xFF;
        frac32 = frac << 13;
        if (frac32) {
            frac32 |= 0x00000001; // make sure NaNs are quiet NaNs
        }
    } else {
        // Normalized number
        exp32 = exp - 15 + 127; // bias adjust
        frac32 = frac << 13;
    }

    uint32_t f32 = sign32 | (exp32 << 23) | frac32;
    float result;
    std::memcpy(&result, &f32, sizeof(result));
    return result;
}

uint16_t floatToHalf(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 0x0001;
    int32_t exp = (bits >> 23) & 0x00FF;
    uint32_t frac = bits & 0x007FFFFF;

    uint16_t sign16 = sign << 15;
    uint16_t exp16;
    uint16_t frac16;

    if (exp == 0xFF) {
        // Infinity or NaN
        exp16 = 0x1F;
        frac16 = (frac ? (frac >> 13) | 0x0001 : 0); // preserve signaling as quiet NaN
    } else if (exp == 0) {
        // Zero or subnormal in float32 -> zero in half (no gradual underflow from subnormals in float32 to half)
        exp16 = 0;
        frac16 = 0;
    } else {
        // Normalized float32 -> may map to normalized or subnormal half
        int32_t newExp = exp - 127 + 15;

        if (newExp >= 0x1F) {
            // Overflow -> Inf
            exp16 = 0x1F;
            frac16 = 0;
        } else if (newExp <= 0) {
            // Underflow -> subnormal half or zero
            if (newExp < -10) {
                // Too small -> zero
                exp16 = 0;
                frac16 = 0;
            } else {
                // Subnormal half (shift mantissa)
                frac = (frac | 0x00800000) >> (1 - newExp);
                frac16 = frac >> 13;
                exp16 = 0;
            }
        } else {
            // Normal case
            exp16 = newExp;
            frac16 = frac >> 13;
        }
    }

    return sign16 | (exp16 << 10) | frac16;
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
    template _VulkanBuffer& VulkanGSPipeline::copyToDevice(Buffer<dtype>& buffer);                                           \
    template void VulkanGSPipeline::copyFromDevice(Buffer<dtype>& buffer);                                                   \
    template void VulkanGSPipeline::copyFromDeviceToDevice(const Buffer<dtype>& src, Buffer<dtype>& dst);                    \
    template dtype VulkanGSPipeline::readElement(const _VulkanBuffer& buffer, size_t index);                                 \
    template void VulkanGSPipeline::dumpRawBytesToFile(Buffer<dtype>& buffer, std::string filename);                         \
    template void VulkanGSPipelineBuffers::reorderSH(Buffer<dtype>& coeffs);                                                 \
    template void VulkanGSPipelineBuffers::undoReorderSH(Buffer<dtype>& coeffs, size_t num_splats);

_INSTANTIATE_BUFFER(uint8_t)
_INSTANTIATE_BUFFER(float)
_INSTANTIATE_BUFFER(int32_t)
_INSTANTIATE_BUFFER(int64_t)

#undef _INSTANTIATE_BUFFER
