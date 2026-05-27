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
    const char* label;   // diagnostics label; nullptr = untracked

    _VulkanBuffer()
        : buffer(VK_NULL_HANDLE),
          allocation(VK_NULL_HANDLE),
          allocSize(0),
          size(0),
          offset(0),
          label(nullptr) {}

    _VulkanBuffer(const _VulkanBuffer& other)
        : buffer(other.buffer),
          allocation(other.allocation),
          allocSize(other.allocSize),
          size(other.size),
          offset(other.offset),
          label(other.label) {}

    _VulkanBuffer& operator=(const _VulkanBuffer& other) {
        buffer = other.buffer;
        allocation = other.allocation;
        allocSize = other.allocSize;
        size = other.size;
        offset = other.offset;
        label = other.label;
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

    size_t deviceSize() const { return deviceBuffer.size / sizeof(T); }
};

struct VulkanGSPipelineBuffers {
    size_t num_splats = 0;
    size_t num_indices = 0;

    // projection inputs
    Buffer<float> xyz_ws;       // (N, 3)
    Buffer<float> sh_coeffs;    // legacy packed full SH: (N, 16, 3)
    Buffer<float> rotations;    // (N, 4)
    Buffer<float> scales_opacs; // legacy activated [scale.xyz, opacity]

    // Raw split SplatData projection inputs used by the Vulkan viewer. These can
    // directly alias Vulkan-external tensor storage during training.
    Buffer<float> sh0;         // (N, 1, 3) flattened
    Buffer<float> shN;         // swizzled rest-only SH: [ceil(N/32), active slots, 32] float4
    Buffer<float> scaling_raw; // (N, 3), log-scale
    Buffer<float> opacity_raw; // (N, 1), logits

    // projection outputs
    Buffer<int32_t> tiles_touched;    // (N,)
    Buffer<int64_t> rect_tile_space;  // (N,)
    Buffer<int32_t> radii;            // (N,)
    Buffer<float> xy_vs;              // (N, 2)
    Buffer<float> depths;             // (N, 1)
    Buffer<float> inv_cov_vs_opacity; // (N, 4)
    Buffer<float> rgb;                // (N, 3)
    Buffer<int32_t> overlay_flags;    // (N, 1), selection/filter classification

    // Two-stage sort (Splatshop): visible primitives are compacted and sorted by
    // radial distance (depth), then tile instances are emitted in depth order and
    // sorted by tile id with a stable radix. Matches the gsplat_fwd CUDA
    // reference (kernels_forward.cuh: primitive_depth_keys → SortPairs →
    // apply_depth_ordering → create_instances → SortPairs) without sorting
    // projection rejects.
    Buffer<uint32_t> primitive_depth_keys;       // (N,) float-as-uint of ‖mean − cam‖²
    Buffer<int32_t> primitive_sort_indices;      // (N,) depth-ranked primitive idx
    Buffer<int32_t> tiles_touched_depth_ordered; // (N,) reordered tiles_touched
    Buffer<int32_t> visible_flags;               // (N,) projection-visible primitive flag
    Buffer<int32_t> visible_prefix;              // (N,) inclusive scan of visible_flags
    Buffer<uint32_t> visible_count;              // (1,) visible primitive count
    Buffer<uint32_t> visible_sort_dispatch_args; // VkDispatchIndirectCommand for visible primitive radix sort

    // tiles
    Buffer<int32_t> index_buffer_offset;      // N
    Buffer<sortingKey_t> sorting_keys_1;      // NInt [no_shrink]
    Buffer<sortingKey_t> sorting_keys_2;      // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_1;      // NInt [no_shrink]
    Buffer<int32_t> sorting_gauss_idx_2;      // NInt [no_shrink]
    Buffer<uint32_t> tile_sort_count;         // (1,) actual tile instance count
    Buffer<uint32_t> tile_sort_dispatch_args; // VkDispatchIndirectCommand for tile-instance radix sort
    Buffer<int32_t> tile_ranges;              // (Gh*Gw, 2)
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

    // Per-session high-water-mark for unsorted_keys / unsorted_gauss_idx capacity.
    // Driven by the deferred (1-frame-stale) num_indices readback so generate_keys
    // can size buffers without a synchronous cumsum readback.
    size_t num_indices_high_water = 0;

    [[nodiscard]] size_t getTotalOwnedAllocSize() const;
    [[nodiscard]] std::map<std::string, size_t> getOwnedVramBreakdown() const;

    template <typename T>
    static void reorderSH(Buffer<T>& coeffs);
    template <typename T>
    static void undoReorderSH(Buffer<T>& coeffs, size_t num_splats);

    static void assignScalesOpacs(Buffer<float>& scales_opacs, size_t n, const float* scales, const float* opacs);
};
