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
#include <atomic>
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
    // 1.5× headroom absorbs typical densify/view jumps without overflow re-runs.
    constexpr double kSortBufferGrowthFactor = 1.5;

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
            lfs::core::alloc_counter::record_site(lfs::core::alloc_counter::Site::FastgsSort);
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
        int capacity_n_instances = 0;        // host-side element capacity of sort buffers
        int last_n_instances = 0;            // most recent completed D2H count
        std::uint64_t* h_n_instances_pinned = nullptr;
        bool h_n_instances_is_pinned = false;
        bool force_sync_next = false; // testing: force mid-pipeline sync fallback
        cudaEvent_t n_instances_ready_event = nullptr;

        FastGSSortBufferCache() {
#if CUDART_VERSION >= 11020
            // Pinned host slot for async n_instances D2H (Phase 1.2).
            void* ptr = nullptr;
            if (cudaMallocHost(&ptr, sizeof(std::uint64_t)) == cudaSuccess) {
                h_n_instances_pinned = static_cast<std::uint64_t*>(ptr);
                h_n_instances_is_pinned = true;
                *h_n_instances_pinned = 0;
            }
#endif
            if (!h_n_instances_pinned) {
                // Fallback: pageable host memory (async D2H still works, may be slower).
                h_n_instances_pinned = new std::uint64_t(0);
                h_n_instances_is_pinned = false;
            }
            if (cudaEventCreateWithFlags(&n_instances_ready_event, cudaEventDisableTiming) !=
                cudaSuccess) {
                n_instances_ready_event = nullptr;
            }
        }

        ~FastGSSortBufferCache() {
            if (n_instances_ready_event) {
                (void)cudaEventDestroy(n_instances_ready_event);
                n_instances_ready_event = nullptr;
            }
            if (!h_n_instances_pinned) {
                return;
            }
            if (h_n_instances_is_pinned) {
                (void)cudaFreeHost(h_n_instances_pinned);
            } else {
                delete h_n_instances_pinned;
            }
            h_n_instances_pinned = nullptr;
        }

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

        /// Grow sort buffers to hold at least n_instances elements (×1.2 headroom).
        void ensure_instance_capacity(int n_instances) {
            if (n_instances <= 0) {
                return;
            }
            if (n_instances <= capacity_n_instances) {
                return;
            }
            const size_t n = static_cast<size_t>(n_instances);
            using Key = fast_lfs::rasterization::InstanceKey;
            keys_current.ensure(n * sizeof(Key));
            keys_alternate.ensure(n * sizeof(Key));
            primitive_indices_current.ensure(n * sizeof(uint));
            primitive_indices_alternate.ensure(n * sizeof(uint));
            // Capacity in elements is the smallest of the four (they grow together).
            const size_t cap_keys = keys_current.size() / sizeof(Key);
            capacity_n_instances = static_cast<int>(cap_keys);
            if (capacity_n_instances < n_instances) {
                capacity_n_instances = n_instances;
            }
        }
    };

    FastGSSortBufferCache& sort_buffer_cache() {
        thread_local FastGSSortBufferCache cache;
        return cache;
    }

    // Phase 1.2: count mid-pipeline n_instances hard-sync fallbacks (warmup/growth only).
    std::atomic<std::uint64_t> g_n_instances_fallback_syncs{0};
    std::atomic<bool> g_force_n_instances_sync{false};

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

void fast_lfs::rasterization::release_sort_workspace_buffers() noexcept {
    auto& cache = sort_buffer_cache();
    cache.keys_current.reset();
    cache.keys_alternate.reset();
    cache.primitive_indices_current.reset();
    cache.primitive_indices_alternate.reset();
    cache.cub_workspace.reset();
    cache.cub_workspace_query_size = 0;
    cache.capacity_n_instances = 0;
    cache.last_n_instances = 0;
}

std::uint64_t fast_lfs::rasterization::n_instances_fallback_sync_count() noexcept {
    return g_n_instances_fallback_syncs.load(std::memory_order_relaxed);
}

void fast_lfs::rasterization::reset_n_instances_fallback_sync_count() noexcept {
    g_n_instances_fallback_syncs.store(0, std::memory_order_relaxed);
}

void fast_lfs::rasterization::set_force_n_instances_sync_for_testing(bool force) noexcept {
    g_force_n_instances_sync.store(force, std::memory_order_relaxed);
}

void fast_lfs::rasterization::reset_sort_capacity_for_testing() noexcept {
    release_sort_workspace_buffers();
    auto& cache = sort_buffer_cache();
    cache.force_sync_next = true;
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
    const float2* sh_value_bounds,
    const uint sh_value_n_cells,
    const float4* w2c,
    const float3* cam_position,
    float* image,
    float* alpha,
    float* depth,
    float* normal,
    float3* primitive_normals,
    const float* bg_color,
    const float* bg_image,
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
        sh_value_bounds,
        sh_value_n_cells,
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

    // Phase 1.1/1.2: grow-only sort buffers + async n_instances (no mid-pipeline
    // hard sync in steady state). Device count always at offset[n_primitives-1].
    auto& sort_cache = sort_buffer_cache();
    sort_cache.bind_stream(stream);
    LFS_ASSERT_MSG(sort_cache.h_n_instances_pinned != nullptr,
                   "FastGS sort cache missing pinned n_instances slot");

    const std::uint64_t* d_n_instances = per_primitive_buffers.offset + n_primitives - 1;

    // Always kick async D2H (consumed after GPU work or on fallback sync).
    check_cuda_with_fastgs_status(
        cudaMemcpyAsync(
            sort_cache.h_n_instances_pinned, d_n_instances,
            sizeof(std::uint64_t), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(n_instances)",
        forward_status,
        "primitive offset scan",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);

    // Event marks D2H completion only — host can resolve n_instances without
    // waiting for create/sort/blend (keeps forward return overlapping blend).
    LFS_ASSERT_MSG(sort_cache.n_instances_ready_event != nullptr,
                   "FastGS sort cache missing n_instances ready event");
    check_cuda_with_fastgs_status(
        cudaEventRecord(sort_cache.n_instances_ready_event, stream),
        "cudaEventRecord(n_instances_ready)",
        forward_status,
        "primitive offset scan",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);

    const bool force_sync =
        g_force_n_instances_sync.load(std::memory_order_relaxed) ||
        sort_cache.force_sync_next ||
        sort_cache.capacity_n_instances == 0;

    auto ensure_cub_workspace = [&](int sort_n, cub::DoubleBuffer<InstanceKey>& keys_db,
                                    cub::DoubleBuffer<uint>& idx_db) -> size_t {
        size_t cub_workspace_size = 0;
        if (static_cast<size_t>(sort_n) > sort_cache.cub_workspace_query_size ||
            sort_cache.cub_workspace.size() == 0) {
            size_t query_bytes = 0;
            check_cuda_with_fastgs_status(
                cub::DeviceRadixSort::SortPairs(
                    nullptr,
                    query_bytes,
                    keys_db,
                    idx_db,
                    sort_n,
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
            sort_cache.cub_workspace_query_size = static_cast<size_t>(sort_n);
            cub_workspace_size = query_bytes;
        } else {
            cub_workspace_size = sort_cache.cub_workspace.size();
        }
        LFS_ASSERT_MSG(sort_cache.cub_workspace.as<char>() != nullptr,
                       "FastGS CUB radix sort cannot execute with null workspace");
        return cub_workspace_size;
    };

    // Run create → sort → extract for a known host sort count (exact or capacity).
    auto run_sort_path = [&](int sort_n, bool use_device_count_for_extract,
                             uint max_write_instances) {
        cub::DoubleBuffer<InstanceKey> keys(
            sort_cache.keys_current.as<InstanceKey>(),
            sort_cache.keys_alternate.as<InstanceKey>());
        cub::DoubleBuffer<uint> primitive_indices(
            sort_cache.primitive_indices_current.as<uint>(),
            sort_cache.primitive_indices_alternate.as<uint>());

        size_t cub_workspace_size = ensure_cub_workspace(sort_n, keys, primitive_indices);

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
            static_cast<uint>(n_primitives),
            max_write_instances);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");
        check_cuda_with_fastgs_status(cudaGetLastError(), "create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);

        check_cuda_with_fastgs_status(
            cub::DeviceRadixSort::SortPairs(
                sort_cache.cub_workspace.as<char>(),
                cub_workspace_size,
                keys,
                primitive_indices,
                sort_n, 0, key_end_bit,
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

        const uint* sorted_idx = primitive_indices.Current();
        if (!sort_cache.owns_sorted_indices(sorted_idx)) {
            throw std::runtime_error("FastGS radix sort returned an unexpected sorted index buffer");
        }

        kernels::forward::extract_instance_ranges_cu<<<div_round_up(sort_n, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
            keys.Current(),
            per_tile_buffers.instance_ranges,
            forward_status,
            depth_bits,
            n_tiles_u32,
            static_cast<uint>(sort_n),
            use_device_count_for_extract ? d_n_instances : nullptr);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.extract_instance_ranges");
        check_cuda_with_fastgs_status(cudaGetLastError(), "extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);

        return const_cast<uint*>(sorted_idx);
    };

    int n_instances = 0;
    uint* sorted_primitive_indices = nullptr;
    size_t per_instance_sort_total_size = 0;

    if (force_sync) {
        // First step / forced / empty capacity: mid-pipeline sync + exact sizes.
        g_n_instances_fallback_syncs.fetch_add(1, std::memory_order_relaxed);
        sort_cache.force_sync_next = false;
        check_cuda_with_fastgs_status(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(n_instances fallback)",
            forward_status,
            "primitive offset scan",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        LFS_FASTGS_PHASE_CHECK("cudaMemcpy(n_instances fallback)");
        n_instances = checked_fastgs_instance_count(
            *sort_cache.h_n_instances_pinned,
            static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sort_cache.last_n_instances = n_instances;
        if (n_instances > 0) {
            sort_cache.ensure_instance_capacity(n_instances);
            sorted_primitive_indices = run_sort_path(
                n_instances, /*use_device_count_for_extract=*/false,
                static_cast<uint>(n_instances));
        }
    } else {
        // Steady state: no mid-pipeline StreamSynchronize.
        // 1) Proactively grow capacity from last_n headroom (no sync).
        // 2) Launch create_instances clamped to capacity (overlaps D2H).
        // 3) Event-wait ONLY for the D2H (not create/sort/blend).
        // 4) Sort/extract/blend with the exact host count.
        // Overflow (n > capacity) is the rare growth fallback.
        if (sort_cache.last_n_instances > 0) {
            const int predicted = static_cast<int>(
                static_cast<double>(sort_cache.last_n_instances) * kSortBufferGrowthFactor + 0.5);
            sort_cache.ensure_instance_capacity(std::max(predicted, sort_cache.last_n_instances));
        }
        const int capacity = sort_cache.capacity_n_instances;
        cub::DoubleBuffer<InstanceKey> keys(
            sort_cache.keys_current.as<InstanceKey>(),
            sort_cache.keys_alternate.as<InstanceKey>());
        cub::DoubleBuffer<uint> primitive_indices(
            sort_cache.primitive_indices_current.as<uint>(),
            sort_cache.primitive_indices_alternate.as<uint>());

        // Optimistic create into capacity-sized buffers (no host count yet).
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
            static_cast<uint>(n_primitives),
            static_cast<uint>(capacity));
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");
        check_cuda_with_fastgs_status(cudaGetLastError(), "create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);

        // Resolve n_instances: wait only for D2H event (create still in flight).
        check_cuda_with_fastgs_status(
            cudaEventSynchronize(sort_cache.n_instances_ready_event),
            "cudaEventSynchronize(n_instances)",
            forward_status,
            "primitive offset scan",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        n_instances = checked_fastgs_instance_count(
            *sort_cache.h_n_instances_pinned,
            static_cast<uint64_t>(n_primitives), n_tiles_u64);

        if (n_instances > capacity) {
            // Growth fallback: drain, grow, re-run exact path.
            g_n_instances_fallback_syncs.fetch_add(1, std::memory_order_relaxed);
            check_cuda_with_fastgs_status(
                cudaStreamSynchronize(stream),
                "cudaStreamSynchronize(overflow re-run)",
                forward_status,
                "sort capacity overflow",
                static_cast<uint64_t>(n_primitives),
                n_tiles_u64);
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, stream),
                                 "cudaMemsetAsync(tile instance ranges re-run)");
            LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(forward_status, 0, sizeof(raster::FastGSForwardStatus), stream),
                                 "cudaMemsetAsync(FastGS forward status re-run)");
            sort_cache.ensure_instance_capacity(n_instances);
            sorted_primitive_indices = run_sort_path(
                n_instances, /*use_device_count_for_extract=*/false,
                static_cast<uint>(n_instances));
        } else if (n_instances > 0) {
            // Exact-size sort/extract — create already wrote [0, n_instances).
            size_t cub_workspace_size = ensure_cub_workspace(n_instances, keys, primitive_indices);
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
            if (!sort_cache.owns_sorted_indices(sorted_primitive_indices)) {
                throw std::runtime_error("FastGS radix sort returned an unexpected sorted index buffer");
            }

            kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
                keys.Current(),
                per_tile_buffers.instance_ranges,
                forward_status,
                depth_bits,
                n_tiles_u32,
                static_cast<uint>(n_instances),
                /*d_n_instances=*/nullptr);
            LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.extract_instance_ranges");
            check_cuda_with_fastgs_status(cudaGetLastError(), "extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
            sync_fastgs_phase_if_requested("extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        }

        sort_cache.last_n_instances = n_instances;
        if (n_instances > 0) {
            sort_cache.ensure_instance_capacity(n_instances);
        }

        // Fall through to shared blend path below.
    }

    // Fallback path still needs blend.
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
            bg_color,
            bg_image,
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

    per_instance_sort_total_size =
        sort_cache.keys_current.size() +
        sort_cache.keys_alternate.size() +
        sort_cache.primitive_indices_current.size() +
        sort_cache.primitive_indices_alternate.size() +
        sort_cache.cub_workspace.size();

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
