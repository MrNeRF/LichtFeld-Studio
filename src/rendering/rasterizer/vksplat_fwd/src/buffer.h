#pragma once

#include <cmath>
#include <cstring> // memcpy
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "config.h"

// https://stackoverflow.com/a/3312896
#ifdef __GNUC__
#define PACK_STRUCT(__Declaration__) __Declaration__ __attribute__((__packed__))
#endif
#ifdef _MSC_VER
#define PACK_STRUCT(__Declaration__) __pragma(pack(push, 1)) __Declaration__ __pragma(pack(pop))
#endif

// Buffers
struct _VulkanBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    size_t allocSize;    // allocated size in bytes
    size_t size;         // actual size in bytes (within the [offset, offset+size) view)
    VkDeviceSize offset; // descriptor binding offset (0 for owned buffers; non-zero for views into a coalesced parent allocation)

    _VulkanBuffer()
        : buffer(VK_NULL_HANDLE),
          allocation(VK_NULL_HANDLE),
          allocSize(0),
          size(0),
          offset(0) {}

    _VulkanBuffer(const _VulkanBuffer& other)
        : buffer(other.buffer),
          allocation(other.allocation),
          allocSize(other.allocSize),
          size(other.size),
          offset(other.offset) {}

    _VulkanBuffer& operator=(const _VulkanBuffer& other) {
        buffer = other.buffer;
        allocation = other.allocation;
        allocSize = other.allocSize;
        size = other.size;
        offset = other.offset;
        return *this;
    }

    // used to test if descriptor needs to be updated
    bool operator==(const _VulkanBuffer& other) const {
        return buffer == other.buffer && allocation == other.allocation &&
               allocSize == other.allocSize && offset == other.offset;
    }
};

template <typename T>
class Buffer : public std::vector<T> {
public:
    _VulkanBuffer deviceBuffer;

    Buffer() : std::vector<T>(),
               deviceBuffer() {}
    Buffer(const Buffer& other) : std::vector<T>(other),
                                  deviceBuffer(other.deviceBuffer) {}
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            std::vector<T>::operator=(other);
            deviceBuffer = other.deviceBuffer;
        }
        return *this;
    }

    size_t byteLength() const { return this->size() * sizeof(T); }
    size_t deviceSize() const { return deviceBuffer.size / sizeof(T); }
};

#define _DECL_DEVICE_BUFFER(dtype, name) Buffer<dtype> name = Buffer<dtype>(false)
// #define _DECL_DEVICE_BUFFER(dtype, name) Buffer<dtype> name

struct VulkanGSPipelineBuffers {
    size_t num_splats = 0;
    size_t num_indices = 0;

    // projection inputs
    Buffer<float> xyz_ws;       // (N, 3)
    Buffer<float> sh_coeffs;    // (N, 16, 3)
    Buffer<float> rotations;    // (N, 4)
    Buffer<float> scales_opacs; // (N, 4)

    // projection outputs
    Buffer<int32_t> tiles_touched;    // (N,)
    Buffer<int64_t> rect_tile_space;  // (N,)
    Buffer<int32_t> radii;            // (N,)
    Buffer<float> xy_vs;              // (N, 2)
    Buffer<float> depths;             // (N, 1)
    Buffer<float> inv_cov_vs_opacity; // (N, 4)
    Buffer<float> rgb;                // (N, 3)
    Buffer<int32_t> overlay_flags;    // (N, 1), selection/filter classification

    // tiles
    Buffer<int32_t> index_buffer_offset; // N
    Buffer<sortingKey_t> sorting_keys_1; // NInt [no_shrink]
    Buffer<sortingKey_t> sorting_keys_2; // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_1; // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_2; // NInt [no_shrink]
    Buffer<int32_t> tile_ranges;         // (Gh*Gw, 2)
    bool is_unsorted_1 = true;
    Buffer<sortingKey_t>& unsorted_keys() { return is_unsorted_1 ? sorting_keys_1 : sorting_keys_2; }
    Buffer<sortingKey_t>& sorted_keys() { return is_unsorted_1 ? sorting_keys_2 : sorting_keys_1; }
    Buffer<sortingKey_t>& unsorted_gauss_idx() { return is_unsorted_1 ? sorting_gauss_idx_1 : sorting_gauss_idx_2; }
    Buffer<sortingKey_t>& sorted_gauss_idx() { return is_unsorted_1 ? sorting_gauss_idx_2 : sorting_gauss_idx_1; }

    // pixels
    Buffer<float> pixel_state;      // (H, W, 4)
    Buffer<float> pixel_depth;      // (H, W, 1), median view-space depth
    Buffer<int32_t> n_contributors; // (H, W, 1)

    // intermediate buffers
    Buffer<int32_t> _cumsum_blockSums;
    Buffer<int32_t> _cumsum_blockSums2;
    Buffer<int32_t> _sorting_histogram;
    Buffer<int32_t> _sorting_histogram_cumsum;

    // GPU-resident VkDispatchIndirectCommand for compute_tile_ranges. Filled by
    // setup_dispatch_indirect; consumed by vkCmdDispatchIndirect. Stored as
    // Buffer<int32_t> only because the rasterizer only instantiates the buffer
    // helpers for {uint8_t, float, int32_t, int64_t}; the 12-byte layout (three
    // 32-bit unsigned counts) is identical to a Buffer<uint32_t>.
    Buffer<int32_t> dispatch_indirect_compute_tile_ranges;

    // Per-session high-water-mark for unsorted_keys / unsorted_gauss_idx capacity.
    // Driven by the deferred (1-frame-stale) num_indices readback so generate_keys
    // never writes past allocated bytes.
    size_t num_indices_high_water = 0;

    template <typename T>
    static void reorderSH(Buffer<T>& coeffs);
    template <typename T>
    static void undoReorderSH(Buffer<T>& coeffs, size_t num_splats);

    static void assignScalesOpacs(Buffer<float>& scales_opacs, size_t n, const float* scales, const float* opacs);
};
