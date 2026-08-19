/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "core/crash_handler.hpp"
#include "core/memory_pressure.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include "forward.h"
#include "helper_math.h"
#include "kernels_forward.cuh"
#include "rasterization_config.h"
#include "utils.h"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
    namespace raster = fast_lfs::rasterization;
    static_assert(sizeof(raster::InstanceKey) == sizeof(uint));
    static_assert(sizeof(raster::InstanceKey) == 4);

    constexpr size_t kCubWorkspaceAlignment = 256;
    static_assert((kCubWorkspaceAlignment & (kCubWorkspaceAlignment - 1)) == 0);

    [[nodiscard]] size_t aligned_cub_workspace_offset(const size_t data_bytes) {
        constexpr size_t padding = kCubWorkspaceAlignment - 1;
        if (data_bytes > std::numeric_limits<size_t>::max() - padding) {
            throw std::overflow_error("FastGS CUB workspace alignment overflow");
        }
        return (data_bytes + padding) & ~padding;
    }

    class ExactAsyncDeviceBuffer {
    public:
        explicit ExactAsyncDeviceBuffer(const char* label)
            : label_(label),
              stream_(nullptr) {}

        ExactAsyncDeviceBuffer(const ExactAsyncDeviceBuffer&) = delete;
        ExactAsyncDeviceBuffer& operator=(const ExactAsyncDeviceBuffer&) = delete;

        ~ExactAsyncDeviceBuffer() {
            reset();
        }

        void bind_stream(cudaStream_t stream) {
            if (ptr_ && stream != stream_) {
                lfs::core::CudaMemoryPool::instance().record_stream(ptr_, stream);
            }
            stream_ = stream;
        }

        void allocate_exact(size_t size) {
            LFS_ASSERT_MSG(ptr_ == nullptr && size_ == 0,
                           "FastGS exact sort block must be empty before allocation");
            LFS_ASSERT_MSG(size > 0,
                           "FastGS exact sort block cannot allocate zero bytes");
            lfs::core::CudaMemoryPool::LabelGuard label_guard(label_);
            ptr_ = lfs::core::allocate_cuda_storage(
                size, stream_, lfs::core::CudaStorageMode::ExactAsync,
                label_, "fastgs.sort_workspace.allocate");
            LFS_ASSERT_MSG(ptr_ != nullptr,
                           "FastGS exact sort block allocation returned null");
            size_ = size;
        }

        void reset() noexcept {
            if (!ptr_) {
                return;
            }
            // Drop ownership first so re-entrant / double reset is a no-op.
            void* const p = ptr_;
            ptr_ = nullptr;
            size_ = 0;

            // after ordered process teardown the primary context may
            // already be unusable. Late TLS dtors must not touch the driver.
            if (lfs::core::gpu_process_teardown_started()) {
                return;
            }
            lfs::core::safe_cuda_pool_deallocate(p, stream_);
        }

        size_t size() const noexcept {
            return size_;
        }

        void* raw() const noexcept {
            return ptr_;
        }

    private:
        void* ptr_ = nullptr;
        size_t size_ = 0;
        const char* label_ = "rasterizer.fastgs.sort_workspace";
        cudaStream_t stream_ = nullptr;
    };

    // One thread-local exact block retained under EXACT-2. The four typed
    // instance buffers occupy the first 16 bytes per capacity element and the
    // exact CUB query occupies the tail. Sorted indices stay live through
    // backward; release_sorted_primitive_indices therefore remains a no-op.
    struct FastGSSortBufferCache {
        ExactAsyncDeviceBuffer workspace{"rasterizer.fastgs.sort_workspace"};
        size_t required_bytes = 0;
        size_t cub_workspace_bytes = 0;
        size_t cub_workspace_offset_bytes = 0;
        int capacity_n_instances = 0;
        std::uint64_t* h_n_instances_pinned = nullptr;
        bool h_n_instances_is_pinned = false;
        bool force_sync_next = false; // testing: force mid-pipeline sync fallback
        cudaEvent_t n_instances_ready_event = nullptr;

        FastGSSortBufferCache() {
#if CUDART_VERSION >= 11020
            // Pinned host slot for async n_instances D2H.
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
            // pre-shutdown hook already released device + host state.
            // After gpu_process_teardown_started(), never re-enter CUDA or free.
            if (lfs::core::gpu_process_teardown_started()) {
                n_instances_ready_event = nullptr;
                h_n_instances_pinned = nullptr;
                return;
            }
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
            workspace.bind_stream(stream);
        }

        [[nodiscard]] size_t per_buffer_bytes() const noexcept {
            return static_cast<size_t>(capacity_n_instances) *
                   sizeof(raster::InstanceKey);
        }

        [[nodiscard]] raster::InstanceKey* keys_current() const noexcept {
            return static_cast<raster::InstanceKey*>(workspace.raw());
        }

        [[nodiscard]] raster::InstanceKey* keys_alternate() const noexcept {
            auto* const base = static_cast<char*>(workspace.raw());
            return reinterpret_cast<raster::InstanceKey*>(
                base + per_buffer_bytes());
        }

        [[nodiscard]] uint* primitive_indices_current() const noexcept {
            auto* const base = static_cast<char*>(workspace.raw());
            return reinterpret_cast<uint*>(base + 2 * per_buffer_bytes());
        }

        [[nodiscard]] uint* primitive_indices_alternate() const noexcept {
            auto* const base = static_cast<char*>(workspace.raw());
            return reinterpret_cast<uint*>(base + 3 * per_buffer_bytes());
        }

        [[nodiscard]] void* cub_workspace() const noexcept {
            auto* const base = static_cast<char*>(workspace.raw());
            return base ? base + cub_workspace_offset_bytes : nullptr;
        }

        bool owns_sorted_indices(const void* ptr) const noexcept {
            return ptr != nullptr &&
                   (ptr == primitive_indices_current() ||
                    ptr == primitive_indices_alternate());
        }

        void bind_layout(int n_instances,
                         size_t cub_bytes,
                         size_t cub_offset_bytes,
                         size_t total_bytes) {
            LFS_ASSERT_MSG(workspace.size() == total_bytes,
                           "FastGS sort layout must cover the exact allocation");
            capacity_n_instances = n_instances;
            cub_workspace_bytes = cub_bytes;
            cub_workspace_offset_bytes = cub_offset_bytes;
            required_bytes = total_bytes;
            LFS_ASSERT_MSG(
                cub_workspace_offset_bytes % kCubWorkspaceAlignment == 0,
                "FastGS CUB workspace offset must be 256-byte aligned");
            LFS_ASSERT_MSG(
                reinterpret_cast<std::uintptr_t>(cub_workspace()) %
                        kCubWorkspaceAlignment ==
                    0,
                "FastGS CUB workspace address must be 256-byte aligned");
        }

        void replace_workspace(int n_instances,
                               size_t cub_bytes,
                               size_t cub_offset_bytes,
                               size_t total_bytes) {
            workspace.reset();
            required_bytes = 0;
            cub_workspace_bytes = 0;
            cub_workspace_offset_bytes = 0;
            capacity_n_instances = 0;
            workspace.allocate_exact(total_bytes);
            bind_layout(n_instances, cub_bytes, cub_offset_bytes, total_bytes);
        }

        void release_workspace() noexcept {
            workspace.reset();
            required_bytes = 0;
            cub_workspace_bytes = 0;
            cub_workspace_offset_bytes = 0;
            capacity_n_instances = 0;
        }
    };

    thread_local FastGSSortBufferCache* g_current_sort_buffer_cache = nullptr;

    void register_sort_workspace_pressure_client_once() {
        // The sort block is TLS-owned. Immediate pressure runs on the allocating
        // thread, so this client can reclaim only that thread's existing block
        // without constructing caches on unrelated threads. Run before the
        // tensor-pool trim so cudaFreeAsync is followed by a pool trim in the
        // same pressure episode.
        static const bool registered = [] {
            lfs::core::MemoryPressureCoordinator::instance().register_client(
                lfs::core::PressureClient{
                    .name = "fastgs-sort-workspace",
                    .priority = 5,
                    .domain = lfs::core::MemoryDomain::CudaDevice,
                    .affinity = lfs::core::PressureAffinity::ImmediateThreadSafe,
                    .estimate = [](const lfs::core::PressureRequest&) -> size_t {
                        return fast_lfs::rasterization::sort_workspace_allocated_bytes();
                    },
                    .shrink = [](const lfs::core::PressureRequest&) {
                        const size_t before =
                            fast_lfs::rasterization::sort_workspace_allocated_bytes();
                        fast_lfs::rasterization::release_sort_workspace_buffers();
                        return lfs::core::ReclaimResult{
                            .logical_bytes_released = before,
                        }; },
                });
            return true;
        }();
        (void)registered;
    }

    FastGSSortBufferCache& sort_buffer_cache() {
        register_sort_workspace_pressure_client_once();
        thread_local FastGSSortBufferCache cache;
        g_current_sort_buffer_cache = &cache;
        return cache;
    }

    // count mid-pipeline n_instances hard-sync fallbacks (warmup/growth only).
    std::atomic<std::uint64_t> g_n_instances_fallback_syncs{0};
    // test hooks (host-side; passed as kernel args each launch).
    std::atomic<int> g_warp_cull_mode{0};            // 0=on, 1=off, 2=wrong empty
    std::atomic<int> g_blend_batch_size_override{0}; // 0 = use config::blend_batch_size
    std::atomic<bool> g_force_n_instances_sync{false};

} // namespace

void fast_lfs::rasterization::release_sorted_primitive_indices(
    void* ptr,
    cudaStream_t /*stream*/) noexcept {
    // Sorted indices live in the persistent exact thread-local block.
    // Ownership is never transferred to the caller — do not cudaFree.
    // Stream ordering with subsequent forwards on the same stream keeps the
    // buffer valid through backward without an explicit free.
    (void)ptr;
}

void fast_lfs::rasterization::release_sort_workspace_buffers() noexcept {
    auto* const cache = g_current_sort_buffer_cache;
    // Device workspace only. Host pinned slot + n_instances event stay alive
    // for the TLS object's lifetime so mid-process release (training-thread
    // shutdown, VRAM tests) does not break a later FastGS forward on the same
    // thread. The TLS dtor frees host/event (or no-ops after process teardown).
    if (cache) {
        cache->release_workspace();
    }
}

std::uint64_t fast_lfs::rasterization::n_instances_fallback_sync_count() noexcept {
    return g_n_instances_fallback_syncs.load(std::memory_order_relaxed);
}

void fast_lfs::rasterization::reset_n_instances_fallback_sync_count() noexcept {
    g_n_instances_fallback_syncs.store(0, std::memory_order_relaxed);
}

void fast_lfs::rasterization::set_warp_cull_mode_for_testing(int mode) noexcept {
    g_warp_cull_mode.store(mode, std::memory_order_relaxed);
}

int fast_lfs::rasterization::warp_cull_mode_for_testing() noexcept {
    return g_warp_cull_mode.load(std::memory_order_relaxed);
}

void fast_lfs::rasterization::set_blend_batch_size_for_testing(int batch_size) noexcept {
    g_blend_batch_size_override.store(batch_size, std::memory_order_relaxed);
}

int fast_lfs::rasterization::blend_batch_size_for_testing() noexcept {
    return g_blend_batch_size_override.load(std::memory_order_relaxed);
}

void fast_lfs::rasterization::set_force_n_instances_sync_for_testing(bool force) noexcept {
    g_force_n_instances_sync.store(force, std::memory_order_relaxed);
}

void fast_lfs::rasterization::reset_sort_capacity_for_testing() noexcept {
    release_sort_workspace_buffers();
    auto& cache = sort_buffer_cache();
    cache.force_sync_next = true;
}

std::size_t fast_lfs::rasterization::sort_workspace_required_bytes() noexcept {
    const auto* const cache = g_current_sort_buffer_cache;
    return cache ? cache->required_bytes : 0;
}

std::size_t fast_lfs::rasterization::sort_workspace_allocated_bytes() noexcept {
    const auto* const cache = g_current_sort_buffer_cache;
    return cache ? cache->workspace.size() : 0;
}

int fast_lfs::rasterization::sort_workspace_capacity_n_instances() noexcept {
    const auto* const cache = g_current_sort_buffer_cache;
    return cache ? cache->capacity_n_instances : 0;
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
    const uint sh_value_bits,
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
        sh_value_bits,
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

    // grow-only sort buffers + async n_instances (no mid-pipeline
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

    auto ensure_sort_workspace = [&](int sort_n) {
        if (sort_n <= sort_cache.capacity_n_instances) {
            return;
        }

        cub::DoubleBuffer<InstanceKey> query_keys(
            static_cast<InstanceKey*>(nullptr),
            static_cast<InstanceKey*>(nullptr));
        cub::DoubleBuffer<uint> query_indices(
            static_cast<uint*>(nullptr),
            static_cast<uint*>(nullptr));
        size_t cub_bytes = 0;
        check_cuda_with_fastgs_status(
            cub::DeviceRadixSort::SortPairs(
                nullptr,
                cub_bytes,
                query_keys,
                query_indices,
                sort_n,
                0,
                key_end_bit,
                stream),
            "cub::DeviceRadixSort::SortPairs workspace query",
            forward_status,
            "radix sort workspace query",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        LFS_ASSERT_MSG(
            cub_bytes > 0,
            "FastGS CUB radix sort returned an empty workspace for nonempty instance input");

        constexpr size_t bytes_per_instance =
            2 * sizeof(InstanceKey) + 2 * sizeof(uint);
        const size_t n = static_cast<size_t>(sort_n);
        if (n > std::numeric_limits<size_t>::max() / bytes_per_instance) {
            throw std::overflow_error("FastGS exact sort workspace size overflow");
        }
        const size_t data_bytes = n * bytes_per_instance;
        const size_t cub_offset_bytes = aligned_cub_workspace_offset(data_bytes);
        if (cub_bytes > std::numeric_limits<size_t>::max() - cub_offset_bytes) {
            throw std::overflow_error("FastGS exact sort workspace size overflow");
        }
        const size_t required_bytes = cub_offset_bytes + cub_bytes;
        if (required_bytes == sort_cache.workspace.size()) {
            // Alignment padding may absorb a small instance-count increase.
            // Rebind the four slices without issuing a redundant allocation.
            sort_cache.bind_layout(
                sort_n, cub_bytes, cub_offset_bytes, required_bytes);
        } else {
            sort_cache.replace_workspace(
                sort_n, cub_bytes, cub_offset_bytes, required_bytes);
        }
    };

    // Run create → sort → extract for a known host sort count (exact or capacity).
    auto run_sort_path = [&](int sort_n, bool use_device_count_for_extract,
                             uint max_write_instances) {
        cub::DoubleBuffer<InstanceKey> keys(
            sort_cache.keys_current(),
            sort_cache.keys_alternate());
        cub::DoubleBuffer<uint> primitive_indices(
            sort_cache.primitive_indices_current(),
            sort_cache.primitive_indices_alternate());

        size_t cub_workspace_size = sort_cache.cub_workspace_bytes;
        LFS_ASSERT_MSG(sort_cache.cub_workspace() != nullptr && cub_workspace_size > 0,
                       "FastGS CUB radix sort cannot execute with an empty workspace");

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
                sort_cache.cub_workspace(),
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
        if (n_instances > 0) {
            ensure_sort_workspace(n_instances);
            sorted_primitive_indices = run_sort_path(
                n_instances, /*use_device_count_for_extract=*/false,
                static_cast<uint>(n_instances));
        }
    } else {
        // Create runs against the exact measured capacity already retained.
        // The D2H event is synchronized only after this launch. A new record
        // drains the stream, grows exactly, and re-runs the path below.
        const int capacity = sort_cache.capacity_n_instances;
        cub::DoubleBuffer<InstanceKey> keys(
            sort_cache.keys_current(),
            sort_cache.keys_alternate());
        cub::DoubleBuffer<uint> primitive_indices(
            sort_cache.primitive_indices_current(),
            sort_cache.primitive_indices_alternate());

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
            ensure_sort_workspace(n_instances);
            sorted_primitive_indices = run_sort_path(
                n_instances, /*use_device_count_for_extract=*/false,
                static_cast<uint>(n_instances));
        } else if (n_instances > 0) {
            // Exact-size sort/extract — create already wrote [0, n_instances).
            size_t cub_workspace_size = sort_cache.cub_workspace_bytes;
            LFS_ASSERT_MSG(sort_cache.cub_workspace() != nullptr && cub_workspace_size > 0,
                           "FastGS CUB radix sort cannot execute with an empty workspace");
            check_cuda_with_fastgs_status(
                cub::DeviceRadixSort::SortPairs(
                    sort_cache.cub_workspace(),
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

        // Fall through to shared blend path below.
    }

    // Production: warp cull ON (mode 0), blend_batch_size from config (or test hook).
    const int warp_cull_mode = g_warp_cull_mode.load(std::memory_order_relaxed);
    const int blend_batch_override = g_blend_batch_size_override.load(std::memory_order_relaxed);
    auto launch_blend = [&]<bool RENDER_NORMAL>() {
        kernels::forward::blend_cu<RENDER_NORMAL><<<grid, dim3(config::block_size_blend_forward), 0, stream>>>(
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
            grid.x,
            warp_cull_mode,
            blend_batch_override);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.blend");
    };
    if (normal != nullptr) {
        launch_blend.template operator()<true>();
    } else {
        launch_blend.template operator()<false>();
    }
    check_cuda_with_fastgs_status(cudaGetLastError(), "blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    sync_fastgs_phase_if_requested("blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);

    per_instance_sort_total_size = sort_cache.workspace.size();

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
