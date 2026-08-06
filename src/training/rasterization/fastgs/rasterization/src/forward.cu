/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "core/alloc_counter.hpp"
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

    // Grow-only high-water sort scratch (Phase 1.1). Never shrinks; frees only
    // on destruction / explicit reset. Matches spirulae slot-style semantics.
    constexpr double kSortBufferGrowthFactor = 1.2;

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

        void set_stream(cudaStream_t stream) noexcept {
            stream_ = stream;
        }

        /// Ensure at least `size` bytes. Reuses existing storage when large enough;
        /// otherwise frees and reallocates with ×1.2 headroom (growth only).
        void ensure(size_t size) {
            if (size == 0) {
                return;
            }
            if (ptr_ && size_ >= size) {
                return;
            }
            const size_t grown = static_cast<size_t>(
                static_cast<double>(size) * kSortBufferGrowthFactor + 0.5);
            const size_t new_size = std::max(size, grown);
            allocate_exact(new_size);
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
                    lfs::core::CudaFailureDisposition::LogOnlyNoLatch);
            }
            ptr_ = nullptr;
            size_ = 0;
        }

        template <typename T>
        T* as() const noexcept {
            return static_cast<T*>(ptr_);
        }

        size_t size() const noexcept {
            return size_;
        }

        void* raw() const noexcept {
            return ptr_;
        }

    private:
        void allocate_exact(size_t size) {
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
            // Phase 0.1: count real driver allocs for gate G2 (sort-buffer churn).
            lfs::core::alloc_counter::record();
            ptr_ = ptr;
            size_ = size;
            lfs::diagnostics::VramProfiler::instance().recordAllocation(
                ptr_, size_,
                lfs::diagnostics::VramAllocationMethod::Async,
                label_ ? label_ : "rasterizer.fastgs.scratch");
        }

        void* ptr_ = nullptr;
        size_t size_ = 0;
        const char* label_ = "rasterizer.fastgs.scratch";
        cudaStream_t stream_ = nullptr;
    };

    // Thread-local persistent sort workspace (like FastRasterizerThreadLocalCaches).
    // Sorted primitive indices live here through backward; release is a no-op free.
    struct FastGSSortBufferCache {
        StreamOrderedDeviceBuffer keys_current{"rasterizer.fastgs.sort_keys"};
        StreamOrderedDeviceBuffer keys_alternate{"rasterizer.fastgs.sort_keys_alt"};
        StreamOrderedDeviceBuffer primitive_indices_current{"rasterizer.fastgs.sort_indices"};
        StreamOrderedDeviceBuffer primitive_indices_alternate{"rasterizer.fastgs.sort_indices_alt"};
        StreamOrderedDeviceBuffer cub_workspace{"rasterizer.fastgs.cub_workspace"};
        size_t cub_workspace_query_size = 0; // last successful CUB temp-bytes query
        int max_n_instances = 0;

        void bind_stream(cudaStream_t stream) {
            keys_current.set_stream(stream);
            keys_alternate.set_stream(stream);
            primitive_indices_current.set_stream(stream);
            primitive_indices_alternate.set_stream(stream);
            cub_workspace.set_stream(stream);
        }

        bool owns_sorted_indices(const void* ptr) const noexcept {
            return ptr != nullptr &&
                   (ptr == primitive_indices_current.raw() ||
                    ptr == primitive_indices_alternate.raw());
        }
    };

    FastGSSortBufferCache& sort_buffer_cache() {
        thread_local FastGSSortBufferCache cache;
        return cache;
    }

} // namespace

void fast_lfs::rasterization::release_sorted_primitive_indices(
    void* ptr,
    cudaStream_t /*stream*/) noexcept {
    // Phase 1.1: sorted indices live in the grow-only thread-local cache.
    // Ownership is never transferred to the caller — do not cudaFree.
    // Stream ordering with subsequent forwards on the same stream keeps the
    // buffer valid through backward without an explicit free.
    (void)ptr;
}

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
    LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.preprocess");
    check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    sync_fastgs_phase_if_requested("preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);

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
    LFS_FASTGS_PHASE_CHECK("cub::DeviceScan::InclusiveSum (Primitive Offsets)");
    sync_fastgs_phase_if_requested(
        "cub::DeviceScan::InclusiveSum (Primitive Offsets)",
        forward_status,
        "primitive offset scan",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);

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
    LFS_FASTGS_PHASE_CHECK("cudaMemcpy(n_instances)");
    const int n_instances = checked_fastgs_instance_count(n_instances_u64, static_cast<uint64_t>(n_primitives), n_tiles_u64);

    // Phase 1.1: grow-only thread-local sort buffers (keys×2, indices×2, CUB WS).
    // Sorted indices stay in-cache through backward; release_sorted is a no-op free.
    auto& sort_cache = sort_buffer_cache();
    sort_cache.bind_stream(stream);

    cub::DoubleBuffer<InstanceKey> keys;
    cub::DoubleBuffer<uint> primitive_indices;
    size_t cub_workspace_size = 0;
    size_t per_instance_sort_total_size = 0;
    uint* sorted_primitive_indices = nullptr;

    if (n_instances > 0) {
        const size_t n_instances_size = static_cast<size_t>(n_instances);
        sort_cache.keys_current.ensure(n_instances_size * sizeof(InstanceKey));
        sort_cache.keys_alternate.ensure(n_instances_size * sizeof(InstanceKey));
        sort_cache.primitive_indices_current.ensure(n_instances_size * sizeof(uint));
        sort_cache.primitive_indices_alternate.ensure(n_instances_size * sizeof(uint));
        sort_cache.max_n_instances = std::max(sort_cache.max_n_instances, n_instances);

        keys = cub::DoubleBuffer<InstanceKey>(
            sort_cache.keys_current.as<InstanceKey>(),
            sort_cache.keys_alternate.as<InstanceKey>());
        primitive_indices = cub::DoubleBuffer<uint>(
            sort_cache.primitive_indices_current.as<uint>(),
            sort_cache.primitive_indices_alternate.as<uint>());

        // Re-query CUB workspace only when the high-water instance count grows
        // past the size last queried (workspace needs scale with n_items).
        if (static_cast<size_t>(n_instances) > sort_cache.cub_workspace_query_size ||
            sort_cache.cub_workspace.size() == 0) {
            size_t query_bytes = 0;
            check_cuda_with_fastgs_status(
                cub::DeviceRadixSort::SortPairs(
                    nullptr,
                    query_bytes,
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
                query_bytes > 0,
                "FastGS CUB radix sort returned an empty workspace for nonempty instance input");
            sort_cache.cub_workspace.ensure(query_bytes);
            sort_cache.cub_workspace_query_size = static_cast<size_t>(n_instances);
            cub_workspace_size = query_bytes;
        } else {
            cub_workspace_size = sort_cache.cub_workspace.size();
        }
        LFS_ASSERT_MSG(sort_cache.cub_workspace.as<char>() != nullptr,
                       "FastGS CUB radix sort cannot execute with null workspace");

        per_instance_sort_total_size =
            sort_cache.keys_current.size() +
            sort_cache.keys_alternate.size() +
            sort_cache.primitive_indices_current.size() +
            sort_cache.primitive_indices_alternate.size() +
            sort_cache.cub_workspace.size();

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
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");
        check_cuda_with_fastgs_status(cudaGetLastError(), "create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);

        check_cuda_with_fastgs_status(
            cub::DeviceRadixSort::SortPairs(
                sort_cache.cub_workspace.as<char>(),
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
        LFS_FASTGS_PHASE_CHECK("cub::DeviceRadixSort::SortPairs (Tile/Depth)");
        sync_fastgs_phase_if_requested(
            "cub::DeviceRadixSort::SortPairs (Tile/Depth)",
            forward_status,
            "radix sort",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);

        sorted_primitive_indices = primitive_indices.Current();
        // Pointer must be one of the two persistent index buffers.
        if (!sort_cache.owns_sorted_indices(sorted_primitive_indices)) {
            throw std::runtime_error("FastGS radix sort returned an unexpected sorted index buffer");
        }
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
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.extract_instance_ranges");
        check_cuda_with_fastgs_status(cudaGetLastError(), "extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
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
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.blend");
    };
    if (normal != nullptr) {
        launch_blend.template operator()<true>();
    } else {
        launch_blend.template operator()<false>();
    }
    check_cuda_with_fastgs_status(cudaGetLastError(), "blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    sync_fastgs_phase_if_requested("blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);

    // Sorted indices remain owned by the thread-local cache (not released/freed).

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
