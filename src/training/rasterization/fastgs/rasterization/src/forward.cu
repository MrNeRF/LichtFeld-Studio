/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "core/crash_handler.hpp"
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
    static_assert(sizeof(raster::InstanceKey) == sizeof(uint64_t));
    static_assert(sizeof(raster::InstanceKey) == 8);

    constexpr size_t kCubWorkspaceAlignment = 256;
    static_assert((kCubWorkspaceAlignment & (kCubWorkspaceAlignment - 1)) == 0);

    [[nodiscard]] size_t aligned_cub_workspace_offset(const size_t data_bytes) {
        constexpr size_t padding = kCubWorkspaceAlignment - 1;
        if (data_bytes > std::numeric_limits<size_t>::max() - padding) {
            throw std::overflow_error("FastGS CUB workspace alignment overflow");
        }
        return (data_bytes + padding) & ~padding;
    }

    // FastGS sort storage is one exact phase allocation. One values buffer is
    // copied into the retained prefix before phase B; the keys and the other
    // values buffer are dead after forward rasterization.
    template <typename Key>
    struct FastGSSortWorkspace {
        char* base = nullptr;
        size_t total_bytes = 0;
        size_t cub_workspace_bytes = 0;
        size_t cub_workspace_offset_bytes = 0;
        int n_instances = 0;

        [[nodiscard]] size_t per_key_buffer_bytes() const noexcept {
            return static_cast<size_t>(n_instances) *
                   sizeof(Key);
        }

        [[nodiscard]] size_t per_index_buffer_bytes() const noexcept {
            return static_cast<size_t>(n_instances) * sizeof(uint);
        }

        [[nodiscard]] Key* keys_current() const noexcept {
            return reinterpret_cast<Key*>(base);
        }

        [[nodiscard]] Key* keys_alternate() const noexcept {
            return reinterpret_cast<Key*>(
                base + per_key_buffer_bytes());
        }

        [[nodiscard]] uint* primitive_indices_current() const noexcept {
            // Both possible sort outputs must be disjoint from the retained phase prefix.
            return reinterpret_cast<uint*>(base + 2 * per_key_buffer_bytes());
        }

        [[nodiscard]] uint* primitive_indices_alternate() const noexcept {
            return reinterpret_cast<uint*>(
                base + 2 * per_key_buffer_bytes() + per_index_buffer_bytes());
        }

        [[nodiscard]] void* cub_workspace() const noexcept {
            return base ? base + cub_workspace_offset_bytes : nullptr;
        }

        void bind_layout(char* allocation,
                         int instance_count,
                         size_t cub_bytes,
                         size_t cub_offset_bytes,
                         size_t allocation_bytes) {
            LFS_ASSERT_MSG(allocation != nullptr && allocation_bytes > 0,
                           "FastGS sort layout requires a nonempty arena allocation");
            base = allocation;
            n_instances = instance_count;
            cub_workspace_bytes = cub_bytes;
            cub_workspace_offset_bytes = cub_offset_bytes;
            total_bytes = allocation_bytes;
            LFS_ASSERT_MSG(
                cub_workspace_offset_bytes % kCubWorkspaceAlignment == 0,
                "FastGS CUB workspace offset must be 256-byte aligned");
            LFS_ASSERT_MSG(
                reinterpret_cast<std::uintptr_t>(cub_workspace()) %
                        kCubWorkspaceAlignment ==
                    0,
                "FastGS CUB workspace address must be 256-byte aligned");
        }
    };

    // test hooks (host-side; passed as kernel args each launch).
    std::atomic<int> g_sort_mode{2};
    std::atomic<bool> g_touched_gating{true};
    std::atomic<int> g_warp_cull_mode{0};            // 0=on, 1=off, 2=wrong empty
    std::atomic<int> g_blend_batch_size_override{0}; // 0 = use config::blend_batch_size

} // namespace

void fast_lfs::rasterization::set_touched_gating_for_testing(const bool enabled) noexcept {
    g_touched_gating.store(enabled, std::memory_order_relaxed);
}

bool fast_lfs::rasterization::touched_gating_for_testing() noexcept {
    return g_touched_gating.load(std::memory_order_relaxed);
}

void fast_lfs::rasterization::set_sort_mode_for_testing(const int mode) noexcept {
    g_sort_mode.store(mode >= 0 && mode <= 2 ? mode : 2, std::memory_order_relaxed);
}

int fast_lfs::rasterization::sort_mode_for_testing() noexcept {
    return g_sort_mode.load(std::memory_order_relaxed);
}

void fast_lfs::rasterization::release_sorted_primitive_indices(
    void* ptr,
    cudaStream_t /*stream*/) noexcept {
    // Sorted indices are part of the owning arena frame. The frame release
    // returns the whole bump allocation after backward has finished.
    (void)ptr;
}

void fast_lfs::rasterization::release_sort_workspace_buffers() noexcept {
    // Kept as a source-compatible no-op for callers that used to release the
    // removed thread-local sort cache. Arena frames own this storage now.
}

std::uint64_t fast_lfs::rasterization::n_instances_fallback_sync_count() noexcept {
    return 0;
}

void fast_lfs::rasterization::reset_n_instances_fallback_sync_count() noexcept {
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
    (void)force;
}

void fast_lfs::rasterization::reset_sort_capacity_for_testing() noexcept {
}

std::size_t fast_lfs::rasterization::sort_workspace_required_bytes() noexcept {
    return 0;
}

std::size_t fast_lfs::rasterization::sort_workspace_allocated_bytes() noexcept {
    return 0;
}

int fast_lfs::rasterization::sort_workspace_capacity_n_instances() noexcept {
    return 0;
}

fast_lfs::rasterization::ForwardResult fast_lfs::rasterization::forward(
    std::function<char*(size_t)> per_primitive_buffers_func,
    std::function<void(size_t)> begin_phase_func,
    std::function<char*(size_t)> phase_buffers_func,
    std::function<char*(const void*, size_t)> retain_phase_prefix_func,
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
    cudaStream_t stream,
    float* max_screen_share,
    float dilation_scale,
    int visible_capacity_hint) {

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const uint64_t n_tiles_u64 = static_cast<uint64_t>(grid.x) * static_cast<uint64_t>(grid.y);
    const int n_tiles = checked_to_int(n_tiles_u64, "n_tiles exceeds int range");
    const uint n_tiles_u32 = static_cast<uint>(n_tiles);
    const int key_end_bit = packed_instance_key_end_bit(n_tiles_u32);
    const uint sh_layout_slots = kernels::shSlotsForBases(static_cast<uint>(sh_layout_bases));

    // Allocate per-tile buffers through arena
    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, stream),
                         "cudaMemsetAsync(tile instance ranges)");

    const float w_f = static_cast<float>(width);
    const float h_f = static_cast<float>(height);
    float clip_left, clip_right, clip_top, clip_bottom;
    ewa_clip_bounds(w_f, h_f, fx, fy, cx, cy, clip_left, clip_right, clip_top, clip_bottom);

    const int visible_cap =
        (visible_capacity_hint > 0 && visible_capacity_hint < n_primitives)
            ? visible_capacity_hint
            : n_primitives;
    const int alloc_visible_cap = std::max(visible_cap, 1);

    const uint n_blocks = div_round_up(static_cast<uint>(n_primitives), static_cast<uint>(config::block_size_preprocess));

    uint* primitive_work_indices = reinterpret_cast<uint*>(
        per_primitive_buffers_func(static_cast<size_t>(n_primitives) * sizeof(uint)));
    if (!primitive_work_indices)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS primitive work indices");

    std::uint8_t* active_block_flag = reinterpret_cast<std::uint8_t*>(
        per_primitive_buffers_func(static_cast<size_t>(n_blocks) * sizeof(std::uint8_t)));
    if (!active_block_flag)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS active block flags");

    uint* active_block_ids = reinterpret_cast<uint*>(
        per_primitive_buffers_func(static_cast<size_t>(n_blocks) * sizeof(uint)));
    if (!active_block_ids)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS active block ids");

    uint* visible_indices = reinterpret_cast<uint*>(
        per_primitive_buffers_func(static_cast<size_t>(alloc_visible_cap) * sizeof(uint)));
    if (!visible_indices)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS visible indices");

    char* per_primitive_buffers_base =
        per_primitive_buffers_func(PerPrimitiveBuffers::required_persistent(alloc_visible_cap));
    if (!per_primitive_buffers_base)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS primitive buffers");
    char* per_primitive_buffers_blob = per_primitive_buffers_base;
    PerPrimitiveBuffers per_primitive_buffers =
        PerPrimitiveBuffers::from_persistent_blob(per_primitive_buffers_blob, alloc_visible_cap);

    float3* primitive_normals = nullptr;
    if (normal != nullptr) {
        primitive_normals = reinterpret_cast<float3*>(
            per_primitive_buffers_func(static_cast<size_t>(alloc_visible_cap) * sizeof(float3)));
        if (!primitive_normals)
            throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS primitive normal buffer");
    }

    begin_phase_func(PerPrimitiveBuffers::required_phase_single_pass(n_primitives, alloc_visible_cap));
    PerPrimitiveBuffers phase_buffers =
        PerPrimitiveBuffers::from_phase_allocator_single_pass(phase_buffers_func, n_primitives, alloc_visible_cap);
    per_primitive_buffers.scan_header = phase_buffers.scan_header;
    per_primitive_buffers.scan_tiles = phase_buffers.scan_tiles;
    per_primitive_buffers.depth_keys = phase_buffers.depth_keys;
    per_primitive_buffers.n_touched_tiles = phase_buffers.n_touched_tiles;
    per_primitive_buffers.offset = phase_buffers.offset;
    per_primitive_buffers.screen_bounds = phase_buffers.screen_bounds;

    auto* forward_status = per_primitive_buffers.forward_status;
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(forward_status, 0, sizeof(raster::FastGSForwardStatus), stream),
                         "cudaMemsetAsync(FastGS forward status)");
    const size_t scan_bytes = sizeof(ScanHeader) + static_cast<size_t>(n_blocks) * sizeof(std::uint64_t);
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(per_primitive_buffers.scan_header, 0, scan_bytes, stream),
                         "cudaMemsetAsync(FastGS single-pass scan state)");
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(active_block_flag, 0, static_cast<size_t>(n_blocks) * sizeof(std::uint8_t), stream),
                         "cudaMemsetAsync(FastGS active block flags)");

    const bool sh_is_q16 = (sh_value_bounds != nullptr && sh_value_n_cells > 0u);
    auto launch_preprocess = [&]<int ACTIVE_SH_BASES, bool SH_Q16>() {
        kernels::forward::preprocess_single_pass_cu<ACTIVE_SH_BASES, SH_Q16><<<n_blocks, config::block_size_preprocess, 0, stream>>>(
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
            per_primitive_buffers.scan_header,
            per_primitive_buffers.scan_tiles,
            active_block_flag,
            active_block_ids,
            primitive_work_indices,
            visible_indices,
            per_primitive_buffers.depth_keys,
            per_primitive_buffers.depths,
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.projected,
            primitive_normals,
            static_cast<uint>(n_primitives),
            static_cast<uint>(visible_cap),
            n_blocks,
            grid.x,
            grid.y,
            sh_layout_slots,
            fx,
            fy,
            cx,
            cy,
            clip_left,
            clip_right,
            clip_top,
            clip_bottom,
            near_,
            far_,
            mip_filter,
            dilation_scale,
            max_screen_share);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.preprocess_single_pass");
    };
    auto launch_preprocess_for_bases = [&]<int ACTIVE_SH_BASES>() {
        if (sh_is_q16)
            launch_preprocess.template operator()<ACTIVE_SH_BASES, true>();
        else
            launch_preprocess.template operator()<ACTIVE_SH_BASES, false>();
    };
    if (active_sh_bases <= 1)
        launch_preprocess.template operator()<1, false>();
    else if (active_sh_bases <= 4)
        launch_preprocess_for_bases.template operator()<4>();
    else if (active_sh_bases <= 9)
        launch_preprocess_for_bases.template operator()<9>();
    else
        launch_preprocess_for_bases.template operator()<16>();
    check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    sync_fastgs_phase_if_requested("preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);

    check_cuda_with_fastgs_status(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize(FastGS single-pass scan)",
        forward_status,
        "preprocess",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);
    LFS_FASTGS_PHASE_CHECK("cudaStreamSynchronize(FastGS single-pass scan)");
    ScanHeader h_scan{};
    check_cuda_with_fastgs_status(
        cudaMemcpy(&h_scan, per_primitive_buffers.scan_header, sizeof(h_scan), cudaMemcpyDeviceToHost),
        "cudaMemcpy(FastGS single-pass counts)",
        forward_status,
        "preprocess",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);
    const int n_visible = checked_fastgs_visible_count(h_scan.n_visible, n_primitives);
    if (n_visible > visible_cap) {
        ForwardResult overflow;
        overflow.n_visible = n_visible;
        overflow.visible_capacity_overflow = true;
        return overflow;
    }

    int n_instances = 0;
    uint* sorted_primitive_indices = nullptr;
    size_t per_instance_sort_total_size = 0;
    size_t visible_sort_scratch_size = 0;
    unsigned tile_sort_key_bytes = 0;

    if (n_visible > 0) {
        n_instances = checked_fastgs_instance_count(
            h_scan.n_instances, static_cast<uint64_t>(n_primitives), n_tiles_u64);

        if (n_instances > 0) {
            auto sort_instances = [&]<typename Key, bool DepthOrdered>(const uint* depth_work_order, const int sort_end_bit) {
                FastGSSortWorkspace<Key> sort_workspace;
                cub::DoubleBuffer<Key> query_keys(
                    static_cast<Key*>(nullptr),
                    static_cast<Key*>(nullptr));
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
                        n_instances,
                        0,
                        sort_end_bit,
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
                    2 * sizeof(Key) + 2 * sizeof(uint);
                const size_t n = static_cast<size_t>(n_instances);
                if (n > std::numeric_limits<size_t>::max() / bytes_per_instance) {
                    throw std::overflow_error("FastGS exact sort workspace size overflow");
                }
                const size_t data_bytes = n * bytes_per_instance;
                const size_t cub_offset_bytes = aligned_cub_workspace_offset(data_bytes);
                if (cub_bytes > std::numeric_limits<size_t>::max() - cub_offset_bytes) {
                    throw std::overflow_error("FastGS exact sort workspace size overflow");
                }
                const size_t total_bytes = cub_offset_bytes + cub_bytes;
                char* const sort_blob = phase_buffers_func(total_bytes);
                if (!sort_blob) {
                    throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS sort buffers from arena");
                }
                sort_workspace.bind_layout(
                    sort_blob, n_instances, cub_bytes, cub_offset_bytes, total_bytes);
                per_instance_sort_total_size = total_bytes;
                tile_sort_key_bytes = sizeof(Key);

                cub::DoubleBuffer<Key> keys(
                    sort_workspace.keys_current(),
                    sort_workspace.keys_alternate());
                cub::DoubleBuffer<uint> primitive_indices(
                    sort_workspace.primitive_indices_current(),
                    sort_workspace.primitive_indices_alternate());

                kernels::forward::create_instances_cu<Key, DepthOrdered><<<div_round_up(n_visible, config::block_size_create_instances), config::block_size_create_instances, 0, stream>>>(
                    per_primitive_buffers.n_touched_tiles,
                    per_primitive_buffers.offset,
                    per_primitive_buffers.depth_keys,
                    per_primitive_buffers.screen_bounds,
                    per_primitive_buffers.projected,
                    visible_indices,
                    depth_work_order,
                    keys.Current(),
                    primitive_indices.Current(),
                    forward_status,
                    grid.x,
                    static_cast<uint>(n_visible),
                    static_cast<uint>(n_instances));
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.create_instances");
                check_cuda_with_fastgs_status(cudaGetLastError(), "create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);
                sync_fastgs_phase_if_requested("create_instances", forward_status, "create_instances", static_cast<uint64_t>(n_primitives), n_tiles_u64);

                check_cuda_with_fastgs_status(
                    cub::DeviceRadixSort::SortPairs(
                        sort_workspace.cub_workspace(),
                        sort_workspace.cub_workspace_bytes,
                        keys,
                        primitive_indices,
                        n_instances, 0, sort_end_bit,
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

                const uint* sorted_source = primitive_indices.Current();
                if (sorted_source != sort_workspace.primitive_indices_current() &&
                    sorted_source != sort_workspace.primitive_indices_alternate()) {
                    throw std::runtime_error("FastGS radix sort returned an unexpected sorted index buffer");
                }

                kernels::forward::extract_instance_ranges_cu<Key><<<div_round_up(n_instances, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges, 0, stream>>>(
                    keys.Current(),
                    per_tile_buffers.instance_ranges,
                    forward_status,
                    n_tiles_u32,
                    static_cast<uint>(n_instances),
                    /*d_n_instances=*/nullptr);
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.extract_instance_ranges");
                check_cuda_with_fastgs_status(cudaGetLastError(), "extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);
                sync_fastgs_phase_if_requested("extract_instance_ranges", forward_status, "extract_instance_ranges", static_cast<uint64_t>(n_primitives), n_tiles_u64);

                // Retain one compact sorted values copy at the phase base.  The
                // sort keys and ping-pong scratch can then be overwritten by the
                // backward helpers, while backward still sees stable ordering.
                sorted_primitive_indices = reinterpret_cast<uint*>(
                    retain_phase_prefix_func(sorted_source,
                                             static_cast<size_t>(n_instances) * sizeof(uint)));
            };
            const int sort_mode = sort_mode_for_testing();
            // Keep small frames on the packed-key path; once instances are large enough,
            // depth-sorting visible splats once and running a narrow tile-ID sort moves
            // much less memory than 64-bit packed-key radix sort and stays bit-identical.
            const bool depth_ordered =
                sort_mode == 1 ||
                (sort_mode == 2 && n_instances >= 65536 &&
                 (n_tiles_u32 <= 65536u ||
                  static_cast<uint64_t>(n_instances) >= 2ull * static_cast<uint64_t>(n_visible)));
            if (depth_ordered) {
                size_t depth_cub_bytes = 0;
                LFS_CUDA_CHECK_MSG(cub::DeviceRadixSort::SortPairs(
                                       nullptr, depth_cub_bytes,
                                       static_cast<const uint*>(nullptr), static_cast<uint*>(nullptr),
                                       static_cast<const uint*>(nullptr), static_cast<uint*>(nullptr),
                                       n_visible, 0, 32, stream),
                                   "FastGS visible-depth sort query");
                size_t scan_cub_bytes = 0;
                LFS_CUDA_CHECK_MSG(cub::DeviceScan::InclusiveSum(
                                       nullptr, scan_cub_bytes,
                                       static_cast<const uint*>(nullptr),
                                       static_cast<uint*>(nullptr),
                                       n_visible, stream),
                                   "FastGS depth-ordered offset scan query");
                const size_t key_bytes = static_cast<size_t>(n_visible) * sizeof(uint);
                auto allocate = [&](size_t bytes) {
                    const size_t aligned_bytes = aligned_cub_workspace_offset(bytes);
                    if (aligned_bytes > std::numeric_limits<size_t>::max() - visible_sort_scratch_size)
                        throw std::overflow_error("FastGS visible sort storage overflow");
                    char* memory = phase_buffers_func(bytes);
                    if (!memory)
                        throw std::runtime_error("OUT_OF_MEMORY: FastGS visible-depth sort storage");
                    visible_sort_scratch_size += aligned_bytes;
                    return memory;
                };
                auto* work_in = reinterpret_cast<uint*>(allocate(key_bytes));
                auto* sorted_depth = reinterpret_cast<uint*>(allocate(key_bytes));
                auto* depth_work_order = reinterpret_cast<uint*>(allocate(key_bytes));
                auto* ordered_counts = reinterpret_cast<uint*>(allocate(key_bytes));
                auto* depth_scratch = allocate(depth_cub_bytes);
                auto* scan_scratch = allocate(scan_cub_bytes);
                kernels::forward::iota_u32_cu<<<div_round_up(static_cast<uint>(n_visible), 256u), 256, 0, stream>>>(
                    work_in, static_cast<uint>(n_visible));
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.depth_work_iota");
                LFS_CUDA_CHECK_MSG(cub::DeviceRadixSort::SortPairs(
                                       depth_scratch, depth_cub_bytes, per_primitive_buffers.depth_keys, sorted_depth,
                                       work_in, depth_work_order, n_visible, 0, 32, stream),
                                   "FastGS stable visible-depth sort");
                kernels::forward::gather_depth_ordered_counts_cu<<<div_round_up(static_cast<uint>(n_visible), 256u), 256, 0, stream>>>(
                    depth_work_order, per_primitive_buffers.n_touched_tiles,
                    ordered_counts, static_cast<uint>(n_visible));
                LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.depth_ordered_counts");
                LFS_CUDA_CHECK_MSG(cub::DeviceScan::InclusiveSum(
                                       scan_scratch, scan_cub_bytes,
                                       ordered_counts, per_primitive_buffers.offset, n_visible, stream),
                                   "FastGS depth-ordered instance offsets");
                const int tile_bits = std::max(1, key_end_bit - kInstanceDepthBits);
                if (n_tiles_u32 <= 65536u)
                    sort_instances.template operator()<std::uint16_t, true>(depth_work_order, tile_bits);
                else
                    sort_instances.template operator()<uint, true>(depth_work_order, tile_bits);
            } else {
                size_t scan_cub_bytes = 0;
                LFS_CUDA_CHECK_MSG(cub::DeviceScan::InclusiveSum(
                                       nullptr, scan_cub_bytes,
                                       static_cast<const uint*>(nullptr),
                                       static_cast<uint*>(nullptr),
                                       n_visible, stream),
                                   "FastGS packed-key offset scan query");
                char* scan_scratch = phase_buffers_func(scan_cub_bytes);
                if (!scan_scratch)
                    throw std::runtime_error("OUT_OF_MEMORY: FastGS packed-key offset scan storage");
                LFS_CUDA_CHECK_MSG(cub::DeviceScan::InclusiveSum(
                                       scan_scratch, scan_cub_bytes,
                                       per_primitive_buffers.n_touched_tiles,
                                       per_primitive_buffers.offset, n_visible, stream),
                                   "FastGS packed-key instance offsets");
                sort_instances.template operator()<InstanceKey, false>(nullptr, key_end_bit);
            }
        }
    }

    // Production: warp cull ON (mode 0), blend_batch_size from config (or test hook).
    const int warp_cull_mode = g_warp_cull_mode.load(std::memory_order_relaxed);
    const int blend_batch_override = g_blend_batch_size_override.load(std::memory_order_relaxed);
    // Plain RGB renders take the quadrant blend: the instance lists stay uncompacted,
    // so no per-instance position array is needed.
    const bool quadrant_blend =
        normal == nullptr && depth == nullptr && warp_cull_mode == 0 && blend_batch_override == 0;

    unsigned char* contributor_masks = nullptr;
    uint* contributor_positions = nullptr;
    std::uint8_t* primitive_touched = nullptr;
    size_t contributor_storage_size = 0;
    if (n_instances > 0) {
        const size_t masks_bytes = aligned_cub_workspace_offset(static_cast<size_t>(n_instances));
        const size_t positions_bytes =
            quadrant_blend ? 0 : aligned_cub_workspace_offset(static_cast<size_t>(n_instances) * sizeof(uint));
        if (masks_bytes > std::numeric_limits<size_t>::max() - positions_bytes)
            throw std::overflow_error("FastGS contributor storage overflow");
        const size_t touched_bytes = g_touched_gating.load(std::memory_order_relaxed)
                                         ? aligned_cub_workspace_offset(static_cast<size_t>(n_visible))
                                         : 0;
        contributor_storage_size = masks_bytes + positions_bytes;
        if (touched_bytes > std::numeric_limits<size_t>::max() - contributor_storage_size)
            throw std::overflow_error("FastGS touched storage overflow");
        contributor_storage_size += touched_bytes;
        char* memory = phase_buffers_func(contributor_storage_size);
        if (!memory)
            throw std::runtime_error("OUT_OF_MEMORY: FastGS contributor storage");
        contributor_masks = reinterpret_cast<unsigned char*>(memory);
        if (quadrant_blend) {
            LFS_CUDA_CHECK_MSG(cudaMemsetAsync(contributor_masks, 0, masks_bytes, stream), "FastGS contributor mask reset");
        } else {
            contributor_positions = reinterpret_cast<uint*>(memory + masks_bytes);
        }
        if (touched_bytes) {
            primitive_touched = reinterpret_cast<std::uint8_t*>(memory + masks_bytes + positions_bytes);
            LFS_CUDA_CHECK_MSG(cudaMemsetAsync(primitive_touched, 0, touched_bytes, stream), "FastGS touched reset");
        }
        const size_t indices_bytes = aligned_cub_workspace_offset(static_cast<size_t>(n_instances) * sizeof(uint));
        if (contributor_storage_size > std::numeric_limits<size_t>::max() - indices_bytes)
            throw std::overflow_error("FastGS contributor prefix overflow");
        retain_phase_prefix_func(sorted_primitive_indices, indices_bytes + contributor_storage_size);
    }
    // Schedule heaviest tiles first (LPT) so long-running tiles start at t=0
    // and avoid tail-wave SM idling.
    const bool use_tile_lpt = (n_instances > 0 && n_tiles > 1);
    if (use_tile_lpt) {
        kernels::forward::prepare_tile_lpt_keys_cu<<<div_round_up(n_tiles_u32, 256u), 256u, 0, stream>>>(
            per_tile_buffers.instance_ranges,
            per_tile_buffers.tile_counts_in,
            per_tile_buffers.tile_ids_in,
            n_tiles_u32);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.prepare_tile_lpt_keys");
        const int count_bits = std::max(1, extract_end_bit(static_cast<uint>(n_instances)));
        size_t cub_bytes = per_tile_buffers.tile_sort_cub_bytes;
        LFS_CUDA_CHECK_MSG(
            cub::DeviceRadixSort::SortPairsDescending(
                per_tile_buffers.tile_sort_cub, cub_bytes,
                per_tile_buffers.tile_counts_in, per_tile_buffers.tile_counts_out,
                per_tile_buffers.tile_ids_in, per_tile_buffers.tile_order,
                n_tiles, 0, count_bits, stream),
            "FastGS forward tile LPT sort");
    }

    auto launch_blend = [&]<bool RENDER_NORMAL, bool RENDER_DEPTH>() {
        static const bool carved = []() {
            LFS_FASTGS_CUDA_CALL(
                cudaFuncSetAttribute(
                    kernels::forward::blend_cu<RENDER_NORMAL, RENDER_DEPTH>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 50),
                "cudaFuncSetAttribute(blend_cu)");
            return true;
        }();
        (void)carved;
        kernels::forward::blend_cu<RENDER_NORMAL, RENDER_DEPTH><<<n_tiles_u32, config::block_size_blend_forward, 0, stream>>>(
            per_tile_buffers.instance_ranges,
            use_tile_lpt ? per_tile_buffers.tile_order : nullptr,
            sorted_primitive_indices,
            contributor_masks,
            contributor_positions,
            primitive_work_indices,
            per_primitive_buffers.projected,
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
    if (quadrant_blend) {
        static const bool quadrant_carved = []() {
            LFS_FASTGS_CUDA_CALL(
                cudaFuncSetAttribute(
                    kernels::forward::blend_quadrant_cu,
                    cudaFuncAttributePreferredSharedMemoryCarveout, 50),
                "cudaFuncSetAttribute(blend_quadrant_cu)");
            return true;
        }();
        (void)quadrant_carved;
        kernels::forward::blend_quadrant_cu<<<n_tiles_u32 * 4u, 64u, 0, stream>>>(
            per_tile_buffers.instance_ranges,
            use_tile_lpt ? per_tile_buffers.tile_order : nullptr,
            sorted_primitive_indices,
            contributor_masks,
            primitive_work_indices,
            per_primitive_buffers.projected,
            image,
            alpha,
            per_tile_buffers.n_contributions,
            per_tile_buffers.final_transmittance,
            bg_color,
            bg_image,
            width,
            height,
            grid.x);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.blend_quadrant");
    } else if (normal != nullptr) {
        if (depth != nullptr)
            launch_blend.template operator()<true, true>();
        else
            launch_blend.template operator()<true, false>();
    } else {
        if (depth != nullptr)
            launch_blend.template operator()<false, true>();
        else
            launch_blend.template operator()<false, false>();
    }
    check_cuda_with_fastgs_status(cudaGetLastError(), "blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    sync_fastgs_phase_if_requested("blend", forward_status, "blend", static_cast<uint64_t>(n_primitives), n_tiles_u64);

    ForwardResult result;
    result.n_instances = n_instances;
    result.n_visible = n_visible;
    result.visible_capacity = alloc_visible_cap;
    result.per_primitive_buffers = per_primitive_buffers_base;
    result.per_primitive_buffers_size = PerPrimitiveBuffers::required_persistent(alloc_visible_cap);
    result.primitive_work_indices = primitive_work_indices;
    result.visible_indices = visible_indices;
    result.active_block_flag = active_block_flag;
    result.active_block_ids = active_block_ids;
    result.n_active_blocks = static_cast<int>(h_scan.n_active_blocks);
    result.primitive_normals = primitive_normals;
    result.sorted_primitive_indices = sorted_primitive_indices;
    result.contributor_masks = contributor_masks;
    result.contributor_positions = contributor_positions;
    result.primitive_touched = primitive_touched;
    result.contributor_storage_size = contributor_storage_size;
    result.sorted_primitive_indices_size = static_cast<size_t>(std::max(n_instances, 0)) * sizeof(uint);
    result.per_instance_sort_total_size = per_instance_sort_total_size;
    result.visible_sort_scratch_size = visible_sort_scratch_size;
    result.tile_sort_key_bytes = tile_sort_key_bytes;
    result.per_instance_sort_scratch_size = per_instance_sort_total_size > result.sorted_primitive_indices_size
                                                ? per_instance_sort_total_size - result.sorted_primitive_indices_size
                                                : 0;
    return result;
}
