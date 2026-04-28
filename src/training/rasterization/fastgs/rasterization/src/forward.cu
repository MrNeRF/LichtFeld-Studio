/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <algorithm>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    int checked_to_int(uint64_t value, const char* message) {
        if (value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error(message);
        }
        return static_cast<int>(value);
    }

    class StreamOrderedDeviceBuffer {
    public:
        StreamOrderedDeviceBuffer() = default;
        explicit StreamOrderedDeviceBuffer(size_t size) {
            allocate(size);
        }

        StreamOrderedDeviceBuffer(const StreamOrderedDeviceBuffer&) = delete;
        StreamOrderedDeviceBuffer& operator=(const StreamOrderedDeviceBuffer&) = delete;

        StreamOrderedDeviceBuffer(StreamOrderedDeviceBuffer&& other) noexcept
            : ptr_(other.ptr_), size_(other.size_) {
            other.ptr_ = nullptr;
            other.size_ = 0;
        }

        ~StreamOrderedDeviceBuffer() {
            reset();
        }

        void allocate(size_t size) {
            reset();
            if (size == 0) {
                return;
            }

            void* ptr = nullptr;
#if CUDART_VERSION >= 11020
            const cudaError_t err = cudaMallocAsync(&ptr, size, nullptr);
#else
            const cudaError_t err = cudaMalloc(&ptr, size);
#endif
            if (err != cudaSuccess) {
                throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS sort buffer (" +
                                         std::to_string(size) + " bytes): " + cudaGetErrorString(err));
            }
            ptr_ = ptr;
            size_ = size;
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
#if CUDART_VERSION >= 11020
            cudaFreeAsync(ptr_, nullptr);
#else
            cudaFree(ptr_);
#endif
            ptr_ = nullptr;
            size_ = 0;
        }

        void* release() noexcept {
            void* ptr = ptr_;
            ptr_ = nullptr;
            size_ = 0;
            return ptr;
        }

        template <typename T>
        T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }

        size_t size() const noexcept {
            return size_;
        }

    private:
        void* ptr_ = nullptr;
        size_t size_ = 0;
    };
} // namespace

fast_lfs::rasterization::ForwardResult fast_lfs::rasterization::forward(
    std::function<char*(size_t)> per_primitive_buffers_func,
    std::function<char*(size_t)> per_tile_buffers_func,
    const float3* means,
    const float3* scales_raw,
    const float4* rotations_raw,
    const float* opacities_raw,
    const float3* sh_coefficients_0,
    const float3* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    float* image,
    float* alpha,
    const int n_primitives,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_, // near and far are macros in windows
    const float far_,
    bool mip_filter) {

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");
    const uint n_tiles_u32 = static_cast<uint>(n_tiles);
    const uint depth_bits = static_cast<uint>(packed_instance_depth_bits(n_tiles_u32));
    const int key_end_bit = packed_instance_key_end_bit(n_tiles_u32);

    // Allocate per-tile buffers through arena
    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    // Initialize tile instance ranges
    static cudaStream_t memset_stream = 0;
    static cudaEvent_t memset_event = 0;
    if constexpr (!config::debug) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            cudaEventCreate(&memset_event);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
        cudaEventRecord(memset_event, memset_stream); // Record event when memset completes
    } else {
        cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);
    }

    // Allocate per-primitive buffers through arena
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    // Preprocess primitives
    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.depth_keys,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        total_bases_sh_rest,
        static_cast<float>(width),
        static_cast<float>(height),
        fx,
        fy,
        cx,
        cy,
        near_,
        far_,
        depth_bits,
        mip_filter);
    CHECK_CUDA(config::debug, "preprocess")

    cub::DeviceScan::InclusiveSum(
        per_primitive_buffers.cub_workspace,
        per_primitive_buffers.cub_workspace_size,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.offset,
        n_primitives);
    CHECK_CUDA(config::debug, "cub::DeviceScan::InclusiveSum (Primitive Offsets)")

    uint32_t n_instances_u32;
    cudaMemcpy(&n_instances_u32, per_primitive_buffers.offset + n_primitives - 1, sizeof(n_instances_u32), cudaMemcpyDeviceToHost);
    CHECK_CUDA(config::debug, "cudaMemcpy(n_instances)")
    const int n_instances = checked_to_int(n_instances_u32, "n_instances exceeds int range");

    StreamOrderedDeviceBuffer keys_current;
    StreamOrderedDeviceBuffer keys_alternate;
    StreamOrderedDeviceBuffer primitive_indices_current;
    StreamOrderedDeviceBuffer primitive_indices_alternate;
    StreamOrderedDeviceBuffer cub_workspace;

    cub::DoubleBuffer<InstanceKey> keys;
    cub::DoubleBuffer<uint> primitive_indices;
    size_t cub_workspace_size = 0;
    size_t per_instance_sort_total_size = 0;
    uint* sorted_primitive_indices = nullptr;

    if (n_instances > 0) {
        const size_t n_instances_size = static_cast<size_t>(n_instances);
        keys_current.allocate(n_instances_size * sizeof(InstanceKey));
        keys_alternate.allocate(n_instances_size * sizeof(InstanceKey));
        primitive_indices_current.allocate(n_instances_size * sizeof(uint));
        primitive_indices_alternate.allocate(n_instances_size * sizeof(uint));

        keys = cub::DoubleBuffer<InstanceKey>(keys_current.as<InstanceKey>(), keys_alternate.as<InstanceKey>());
        primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current.as<uint>(), primitive_indices_alternate.as<uint>());

        cub::DeviceRadixSort::SortPairs(
            nullptr,
            cub_workspace_size,
            keys,
            primitive_indices,
            n_instances,
            0,
            key_end_bit);
        cub_workspace.allocate(cub_workspace_size);

        per_instance_sort_total_size =
            keys_current.size() +
            keys_alternate.size() +
            primitive_indices_current.size() +
            primitive_indices_alternate.size() +
            cub_workspace.size();

        kernels::forward::create_instances_cu<<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances>>>(
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            per_primitive_buffers.depth_keys,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            keys.Current(),
            primitive_indices.Current(),
            grid.x,
            depth_bits,
            n_primitives);
        CHECK_CUDA(config::debug, "create_instances")

        cub::DeviceRadixSort::SortPairs(
            cub_workspace.as<char>(),
            cub_workspace_size,
            keys,
            primitive_indices,
            n_instances, 0, key_end_bit);
        CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Tile/Depth)")

        sorted_primitive_indices = primitive_indices.Current();
    }

    // Wait for memset to complete (GPU-side wait, doesn't block CPU)
    if constexpr (!config::debug) {
        cudaStreamWaitEvent(nullptr, memset_event, 0); // Default stream waits for memset
    }

    // Extract instance ranges
    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            keys.Current(),
            per_tile_buffers.instance_ranges,
            depth_bits,
            n_instances);
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    // Perform blending
    kernels::forward::blend_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        sorted_primitive_indices,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        image,
        alpha,
        per_tile_buffers.n_contributions,
        per_tile_buffers.final_transmittance,
        width,
        height,
        grid.x);
    CHECK_CUDA(config::debug, "blend")

    if (n_instances > 0) {
        if (sorted_primitive_indices == primitive_indices_current.as<uint>()) {
            primitive_indices_current.release();
        } else if (sorted_primitive_indices == primitive_indices_alternate.as<uint>()) {
            primitive_indices_alternate.release();
        } else {
            throw std::runtime_error("FastGS radix sort returned an unexpected sorted index buffer");
        }
    }

    ForwardResult result;
    result.n_instances = n_instances;
    result.sorted_primitive_indices = sorted_primitive_indices;
    result.sorted_primitive_indices_size = static_cast<size_t>(std::max(n_instances, 0)) * sizeof(uint);
    result.per_instance_sort_total_size = per_instance_sort_total_size;
    result.per_instance_sort_scratch_size = per_instance_sort_total_size > result.sorted_primitive_indices_size
                                                ? per_instance_sort_total_size - result.sorted_primitive_indices_size
                                                : 0;
    return result;
}
