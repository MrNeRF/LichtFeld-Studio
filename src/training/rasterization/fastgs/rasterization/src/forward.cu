/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "diagnostics/vram_profiler.hpp"
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
    namespace raster = fast_lfs::rasterization;

    class StreamOrderedDeviceBuffer {
    public:
        StreamOrderedDeviceBuffer() = default;
        explicit StreamOrderedDeviceBuffer(const char* label, cudaStream_t stream = nullptr)
            : label_(label),
              stream_(stream) {}

        StreamOrderedDeviceBuffer(const StreamOrderedDeviceBuffer&) = delete;
        StreamOrderedDeviceBuffer& operator=(const StreamOrderedDeviceBuffer&) = delete;

        StreamOrderedDeviceBuffer(StreamOrderedDeviceBuffer&& other) noexcept
            : ptr_(other.ptr_),
              size_(other.size_),
              label_(other.label_),
              stream_(other.stream_) {
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
            const cudaError_t err = cudaMallocAsync(&ptr, size, stream_);
#else
            const cudaError_t err = cudaMalloc(&ptr, size);
#endif
            if (err != cudaSuccess) {
                LFS_ENSURE_CUDA_SUCCESS_MSG(
                    err, "FastGS sort-buffer allocation",
                    lfs::core::detail::format_cuda_safe(
                        "requested_bytes={}, label={}", size,
                        label_ ? label_ : "rasterizer.fastgs.scratch"));
            }
            ptr_ = ptr;
            size_ = size;
            lfs::diagnostics::VramProfiler::instance().recordAllocation(
                ptr_, size_,
                lfs::diagnostics::VramAllocationMethod::Async,
                label_ ? label_ : "rasterizer.fastgs.scratch");
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
            lfs::diagnostics::VramProfiler::instance().recordDeallocation(ptr_);
#if CUDART_VERSION >= 11020
            // Free on the stream that used the buffer — a nullptr free would be
            // unordered with the sort kernels once they run on a real stream.
            const cudaError_t status = cudaFreeAsync(ptr_, stream_);
#else
            const cudaError_t status = cudaFree(ptr_);
#endif
            if (status != cudaSuccess) {
                lfs::core::ensure_cuda_success(
                    status, "FastGS sort-buffer free",
                    lfs::core::detail::format_cuda_safe(
                        "ptr={}, bytes={}, label={}", ptr_, size_,
                        label_ ? label_ : "rasterizer.fastgs.scratch"),
                    LFS_SOURCE_SITE_CURRENT(),
                    lfs::core::CudaFailureDisposition::LogOnly);
            }
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
        const char* label_ = "rasterizer.fastgs.scratch";
        cudaStream_t stream_ = nullptr;
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
    const float4* sh_coefficients_rest,
    const float4* w2c,
    const float3* cam_position,
    float* image,
    float* alpha,
    float* depth,
    float* normal,
    float3* primitive_normals,
    const int n_primitives,
    const int active_sh_bases,
    const int sh_layout_bases,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_, // near and far are macros in windows
    const float far_,
    bool mip_filter,
    cudaStream_t stream) {

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");
    const uint n_tiles_u32 = static_cast<uint>(n_tiles);
    const uint depth_bits = static_cast<uint>(packed_instance_depth_bits(n_tiles_u32));
    const int key_end_bit = packed_instance_key_end_bit(n_tiles_u32);
    const uint sh_layout_slots = kernels::shSlotsForBases(static_cast<uint>(sh_layout_bases));

    // Allocate per-tile buffers through arena
    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    // Initialize tile instance ranges on the main stream. The old side-stream
    // overlap trick (~64KB memset) relied on legacy-stream implicit ordering
    // with the previous frame's reads of this same arena memory — gone once
    // the kernels run on an explicit stream.
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, stream),
                         "cudaMemsetAsync(tile instance ranges)");

    // Allocate per-primitive buffers through arena
    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, n_primitives);

    auto* forward_status = per_primitive_buffers.forward_status;
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(forward_status, 0, sizeof(raster::FastGSForwardStatus), stream),
                         "cudaMemsetAsync(FastGS forward status)");

    // Preprocess primitives
    kernels::forward::preprocess_cu<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, stream>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.depth_keys,
        per_primitive_buffers.depths,
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        normal != nullptr ? primitive_normals : nullptr,
        n_primitives,
        grid.x,
        grid.y,
        active_sh_bases,
        sh_layout_slots,
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
    check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    if constexpr (config::debug) {
        check_cuda_with_fastgs_status(cudaDeviceSynchronize(), "preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        throw_if_fastgs_forward_status(forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    } else {
        sync_fastgs_phase_if_requested("preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }

    check_cuda_with_fastgs_status(
        cub::DeviceScan::InclusiveSum(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            n_primitives,
            stream),
        "cub::DeviceScan::InclusiveSum (Primitive Offsets)",
        forward_status,
        "primitive offset scan",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);
    LFS_FASTGS_PHASE_CHECK(config::debug, "cub::DeviceScan::InclusiveSum (Primitive Offsets)");
    if constexpr (!config::debug) {
        sync_fastgs_phase_if_requested(
            "cub::DeviceScan::InclusiveSum (Primitive Offsets)",
            forward_status,
            "primitive offset scan",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
    }

    // Sizing readback: host-blocking by necessity (buffer sizes depend on it),
    // but scoped to this stream instead of relying on legacy-stream ordering.
    std::uint64_t n_instances_u64 = 0;
    check_cuda_with_fastgs_status(
        [&] {
            const cudaError_t copy_err = cudaMemcpyAsync(
                &n_instances_u64, per_primitive_buffers.offset + n_primitives - 1,
                sizeof(n_instances_u64), cudaMemcpyDeviceToHost, stream);
            if (copy_err != cudaSuccess) {
                return copy_err;
            }
            return cudaStreamSynchronize(stream);
        }(),
        "cudaMemcpy(n_instances)",
        forward_status,
        "primitive offset scan",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);
    LFS_FASTGS_PHASE_CHECK(config::debug, "cudaMemcpy(n_instances)");
    const int n_instances = checked_fastgs_instance_count(n_instances_u64, static_cast<uint64_t>(n_primitives), n_tiles_u64);

    StreamOrderedDeviceBuffer keys_current("rasterizer.fastgs.sort_keys", stream);
    StreamOrderedDeviceBuffer keys_alternate("rasterizer.fastgs.sort_keys_alt", stream);
    StreamOrderedDeviceBuffer primitive_indices_current("rasterizer.fastgs.sort_indices", stream);
    StreamOrderedDeviceBuffer primitive_indices_alternate("rasterizer.fastgs.sort_indices_alt", stream);
    StreamOrderedDeviceBuffer cub_workspace("rasterizer.fastgs.cub_workspace", stream);

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

        check_cuda_with_fastgs_status(
            cub::DeviceRadixSort::SortPairs(
                nullptr,
                cub_workspace_size,
                keys,
                primitive_indices,
                n_instances,
                0,
                key_end_bit),
            "cub::DeviceRadixSort::SortPairs workspace query",
            forward_status,
            "radix sort workspace query",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        LFS_ASSERT_MSG(
            cub_workspace_size > 0,
            "FastGS CUB radix sort returned an empty workspace for nonempty instance input");
        cub_workspace.allocate(cub_workspace_size);
        LFS_ASSERT_MSG(cub_workspace.as<char>() != nullptr,
                       "FastGS CUB radix sort cannot execute with null workspace");

        per_instance_sort_total_size =
            keys_current.size() +
            keys_alternate.size() +
            primitive_indices_current.size() +
            primitive_indices_alternate.size() +
            cub_workspace.size();

        kernels::forward::create_instances_cu<<<div_round_up(n_primitives, config::block_size_create_instances), config::block_size_create_instances, 0, stream>>>(
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            per_primitive_buffers.depth_keys,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            keys.Current(),
            primitive_indices.Current(),
            forward_status,
            grid.x,
            depth_bits,
            n_primitives);
        check_cuda_with_fastgs_status(cudaGetLastError(), "create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        if constexpr (config::debug) {
            check_cuda_with_fastgs_status(cudaDeviceSynchronize(), "create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
            throw_if_fastgs_forward_status(forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        } else {
            sync_fastgs_phase_if_requested("create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        }

        check_cuda_with_fastgs_status(
            cub::DeviceRadixSort::SortPairs(
                cub_workspace.as<char>(),
                cub_workspace_size,
                keys,
                primitive_indices,
                n_instances, 0, key_end_bit,
                stream),
            "cub::DeviceRadixSort::SortPairs (Tile/Depth)",
            forward_status,
            "radix sort",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        LFS_FASTGS_PHASE_CHECK(config::debug, "cub::DeviceRadixSort::SortPairs (Tile/Depth)");
        if constexpr (!config::debug) {
            sync_fastgs_phase_if_requested(
                "cub::DeviceRadixSort::SortPairs (Tile/Depth)",
                forward_status,
                "radix sort",
                static_cast<uint64_t>(n_primitives),
                n_tiles_u64);
        }

        sorted_primitive_indices = primitive_indices.Current();
    }

    // Extract instance ranges
    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
            keys.Current(),
            per_tile_buffers.instance_ranges,
            forward_status,
            depth_bits,
            n_tiles_u32,
            n_instances);
        check_cuda_with_fastgs_status(cudaGetLastError(), "extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        if constexpr (config::debug) {
            check_cuda_with_fastgs_status(cudaDeviceSynchronize(), "extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
            throw_if_fastgs_forward_status(forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        } else {
            sync_fastgs_phase_if_requested("extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        }
    }

    // Perform blending
    auto launch_blend = [&]<bool RENDER_NORMAL>() {
        kernels::forward::blend_cu<RENDER_NORMAL><<<grid, block, 0, stream>>>(
            per_tile_buffers.instance_ranges,
            sorted_primitive_indices,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_primitive_buffers.color,
            per_primitive_buffers.depths,
            primitive_normals,
            image,
            alpha,
            depth,
            normal,
            per_tile_buffers.n_contributions,
            per_tile_buffers.final_transmittance,
            width,
            height,
            grid.x);
    };
    if (normal != nullptr) {
        launch_blend.template operator()<true>();
    } else {
        launch_blend.template operator()<false>();
    }
    check_cuda_with_fastgs_status(cudaGetLastError(), "blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    if constexpr (config::debug) {
        check_cuda_with_fastgs_status(cudaDeviceSynchronize(), "blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        throw_if_fastgs_forward_status(forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    } else {
        sync_fastgs_phase_if_requested("blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }

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
