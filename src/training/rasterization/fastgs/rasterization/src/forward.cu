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
#include "visibility.cuh"
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

    // FastGS sort storage is one exact phase allocation. One values buffer is
    // copied into the retained prefix before phase B; the keys and the other
    // values buffer are dead after forward rasterization.
    struct FastGSSortWorkspace {
        char* base = nullptr;
        size_t total_bytes = 0;
        size_t cub_workspace_bytes = 0;
        size_t cub_workspace_offset_bytes = 0;
        int n_instances = 0;

        [[nodiscard]] size_t per_buffer_bytes() const noexcept {
            return static_cast<size_t>(n_instances) *
                   sizeof(raster::InstanceKey);
        }

        [[nodiscard]] raster::InstanceKey* keys_current() const noexcept {
            return reinterpret_cast<raster::InstanceKey*>(base + per_buffer_bytes());
        }

        [[nodiscard]] raster::InstanceKey* keys_alternate() const noexcept {
            return reinterpret_cast<raster::InstanceKey*>(
                base + 2 * per_buffer_bytes());
        }

        [[nodiscard]] uint* primitive_indices_current() const noexcept {
            return reinterpret_cast<uint*>(base);
        }

        [[nodiscard]] uint* primitive_indices_alternate() const noexcept {
            return reinterpret_cast<uint*>(base + 3 * per_buffer_bytes());
        }

        [[nodiscard]] uint* retained_indices() const noexcept {
            return reinterpret_cast<uint*>(base);
        }

        [[nodiscard]] void* cub_workspace() const noexcept {
            return base ? base + cub_workspace_offset_bytes : nullptr;
        }

        bool owns_sorted_indices(const void* ptr) const noexcept {
            return ptr != nullptr && ptr == retained_indices();
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
    std::atomic<int> g_warp_cull_mode{0};            // 0=on, 1=off, 2=wrong empty
    std::atomic<int> g_blend_batch_size_override{0}; // 0 = use config::blend_batch_size

} // namespace

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
    float* max_screen_share) {

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

    // First pass computes only visibility.  The retained primitive arrays are
    // allocated after this stable compaction, so their footprint follows the
    // visible work rather than N.
    const float w_f = static_cast<float>(width);
    const float h_f = static_cast<float>(height);
    float clip_left, clip_right, clip_top, clip_bottom;
    ewa_clip_bounds(w_f, h_f, fx, fy, cx, cy, clip_left, clip_right, clip_top, clip_bottom);

    char* visibility_blob = per_primitive_buffers_func(required<VisibilityBuffers>(n_primitives));
    if (!visibility_blob)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS visibility buffers");
    VisibilityBuffers visibility_buffers = VisibilityBuffers::from_blob(visibility_blob, n_primitives);

    auto launch_preprocess = [&](const uint* primitive_indices,
                                 uint* visibility_mask,
                                 uint* depth_keys,
                                 float* depths,
                                 std::uint64_t* n_touched_tiles,
                                 ushort4* screen_bounds,
                                 PackedMeanBBox* mean2d,
                                 float4* conic_opacity,
                                 float4* color,
                                 float3* normals,
                                 const uint n_work_items,
                                 float* screen_share) {
        kernels::forward::preprocess_cu<<<div_round_up(n_work_items, static_cast<uint>(config::block_size_preprocess)), config::block_size_preprocess, 0, stream>>>(
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
            primitive_indices,
            visibility_mask,
            depth_keys,
            depths,
            n_touched_tiles,
            screen_bounds,
            mean2d,
            conic_opacity,
            color,
            normals,
            n_work_items,
            n_primitives,
            grid.x,
            grid.y,
            active_sh_bases,
            sh_layout_slots,
            w_f,
            h_f,
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
            depth_bits,
            mip_filter,
            screen_share);
        LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.preprocess");
    };

    const size_t visibility_mask_bytes =
        ((static_cast<size_t>(n_primitives) + 31u) / 32u) * sizeof(uint);
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(visibility_buffers.visibility_mask, 0,
                                         visibility_mask_bytes, stream),
                         "cudaMemsetAsync(FastGS visibility mask)");

    launch_preprocess(nullptr,
                      visibility_buffers.visibility_mask,
                      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                      static_cast<uint>(n_primitives), max_screen_share);
    LFS_FASTGS_PHASE_CHECK("fastgs.forward.preprocess.visibility");

    const uint n_visibility_blocks = static_cast<uint>(
        (static_cast<size_t>(n_primitives) + config::visibility_block_size - 1) /
        config::visibility_block_size);
    lfs::rasterization::visibility::count_blocks<<<n_visibility_blocks,
                                                   lfs::rasterization::visibility::kBlockSize, 0, stream>>>(
        visibility_buffers.visibility_mask,
        visibility_buffers.block_counts,
        static_cast<uint>(n_primitives));
    LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.count_visible_blocks");

    check_cuda_with_fastgs_status(
        cub::DeviceScan::InclusiveSum(
            visibility_buffers.cub_workspace,
            visibility_buffers.cub_workspace_size,
            visibility_buffers.block_counts,
            visibility_buffers.block_offsets,
            static_cast<int>(n_visibility_blocks),
            stream),
        "cub::DeviceScan::InclusiveSum (FastGS visibility)",
        nullptr,
        "visibility scan",
        static_cast<uint64_t>(n_primitives),
        n_tiles_u64);
    LFS_FASTGS_PHASE_CHECK("cub::DeviceScan::InclusiveSum (FastGS visibility)");

    lfs::rasterization::visibility::compact_indices<<<div_round_up(n_primitives, config::block_size_preprocess), config::block_size_preprocess, 0, stream>>>(
        visibility_buffers.visibility_mask,
        visibility_buffers.block_offsets,
        visibility_buffers.visible_indices,
        visibility_buffers.primitive_work_indices,
        static_cast<uint>(n_primitives));
    LFS_CUDA_LAUNCH_CHECK(stream, "fastgs.forward.compact_visible");

    uint h_n_visible = 0;
    LFS_CUDA_CHECK_MSG(
        cudaMemcpy(&h_n_visible, visibility_buffers.block_offsets + n_visibility_blocks - 1,
                   sizeof(h_n_visible), cudaMemcpyDeviceToHost),
        "cudaMemcpy(FastGS visible count)");
    const int n_visible = checked_to_int(h_n_visible, "visible primitive count exceeds int range");

    char* per_primitive_buffers_base =
        per_primitive_buffers_func(PerPrimitiveBuffers::required_persistent(n_visible));
    if (!per_primitive_buffers_base)
        throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS primitive buffers");
    char* per_primitive_buffers_blob = per_primitive_buffers_base;
    PerPrimitiveBuffers per_primitive_buffers =
        PerPrimitiveBuffers::from_persistent_blob(per_primitive_buffers_blob, n_visible);

    float3* primitive_normals = nullptr;
    if (normal != nullptr && n_visible > 0) {
        primitive_normals = reinterpret_cast<float3*>(
            per_primitive_buffers_func(static_cast<size_t>(n_visible) * sizeof(float3)));
        if (!primitive_normals)
            throw std::runtime_error("OUT_OF_MEMORY: Failed to allocate FastGS primitive normal buffer");
    }

    begin_phase_func(PerPrimitiveBuffers::required_phase(n_visible));
    PerPrimitiveBuffers phase_buffers =
        PerPrimitiveBuffers::from_phase_allocator(phase_buffers_func, n_visible);
    per_primitive_buffers.depth_keys = phase_buffers.depth_keys;
    per_primitive_buffers.n_touched_tiles = phase_buffers.n_touched_tiles;
    per_primitive_buffers.offset = phase_buffers.offset;
    per_primitive_buffers.screen_bounds = phase_buffers.screen_bounds;
    per_primitive_buffers.cub_workspace = phase_buffers.cub_workspace;
    per_primitive_buffers.cub_workspace_size = phase_buffers.cub_workspace_size;

    auto* forward_status = per_primitive_buffers.forward_status;
    LFS_FASTGS_CUDA_CALL(cudaMemsetAsync(forward_status, 0, sizeof(raster::FastGSForwardStatus), stream),
                         "cudaMemsetAsync(FastGS forward status)");

    if (n_visible > 0) {
        launch_preprocess(visibility_buffers.visible_indices,
                          nullptr,
                          per_primitive_buffers.depth_keys,
                          per_primitive_buffers.depths,
                          per_primitive_buffers.n_touched_tiles,
                          per_primitive_buffers.screen_bounds,
                          per_primitive_buffers.mean2d,
                          per_primitive_buffers.conic_opacity,
                          per_primitive_buffers.color,
                          primitive_normals,
                          static_cast<uint>(n_visible), nullptr);
        check_cuda_with_fastgs_status(cudaGetLastError(), "preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
        sync_fastgs_phase_if_requested("preprocess", forward_status, "preprocess", static_cast<uint64_t>(n_primitives), n_tiles_u64);
    }

    check_cuda_with_fastgs_status(
        cub::DeviceScan::InclusiveSum(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            n_visible,
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

    // Resolve the compact workset count before allocating sort storage. The
    // arena is bump/frame based, so the query must precede the one exact
    // allocation containing both ping-pong data buffers and CUB scratch.
    const std::uint64_t* d_n_instances = n_visible > 0
                                             ? per_primitive_buffers.offset + n_visible - 1
                                             : nullptr;
    int n_instances = 0;
    uint* sorted_primitive_indices = nullptr;
    size_t per_instance_sort_total_size = 0;
    FastGSSortWorkspace sort_workspace;

    if (n_visible > 0) {
        check_cuda_with_fastgs_status(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(n_instances)",
            forward_status,
            "primitive offset scan",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        LFS_FASTGS_PHASE_CHECK("cudaStreamSynchronize(n_instances)");
        std::uint64_t h_n_instances = 0;
        check_cuda_with_fastgs_status(
            cudaMemcpy(&h_n_instances, d_n_instances,
                       sizeof(h_n_instances), cudaMemcpyDeviceToHost),
            "cudaMemcpy(n_instances)",
            forward_status,
            "primitive offset scan",
            static_cast<uint64_t>(n_primitives),
            n_tiles_u64);
        n_instances = checked_fastgs_instance_count(
            h_n_instances, static_cast<uint64_t>(n_primitives), n_tiles_u64);

        if (n_instances > 0) {
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
                    n_instances,
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

            cub::DoubleBuffer<InstanceKey> keys(
                sort_workspace.keys_current(),
                sort_workspace.keys_alternate());
            cub::DoubleBuffer<uint> primitive_indices(
                sort_workspace.primitive_indices_current(),
                sort_workspace.primitive_indices_alternate());

            kernels::forward::create_instances_cu<<<div_round_up(n_visible, config::block_size_create_instances), config::block_size_create_instances, 0, stream>>>(
                per_primitive_buffers.n_touched_tiles,
                per_primitive_buffers.offset,
                per_primitive_buffers.depth_keys,
                per_primitive_buffers.screen_bounds,
                per_primitive_buffers.mean2d,
                per_primitive_buffers.conic_opacity,
                visibility_buffers.visible_indices,
                keys.Current(),
                primitive_indices.Current(),
                forward_status,
                grid.x,
                depth_bits,
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

            const uint* sorted_source = primitive_indices.Current();
            if (sorted_source != sort_workspace.primitive_indices_current() &&
                sorted_source != sort_workspace.primitive_indices_alternate()) {
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

            // Retain one compact sorted values copy at the phase base.  The
            // sort keys and ping-pong scratch can then be overwritten by the
            // backward helpers, while backward still sees stable ordering.
            sorted_primitive_indices = reinterpret_cast<uint*>(
                retain_phase_prefix_func(sorted_source,
                                         static_cast<size_t>(n_instances) * sizeof(uint)));
        }
    }

    // Production: warp cull ON (mode 0), blend_batch_size from config (or test hook).
    const int warp_cull_mode = g_warp_cull_mode.load(std::memory_order_relaxed);
    const int blend_batch_override = g_blend_batch_size_override.load(std::memory_order_relaxed);
    auto launch_blend = [&]<bool RENDER_NORMAL>() {
        kernels::forward::blend_cu<RENDER_NORMAL><<<grid, dim3(config::block_size_blend_forward), 0, stream>>>(
            per_tile_buffers.instance_ranges,
            sorted_primitive_indices,
            visibility_buffers.primitive_work_indices,
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

    ForwardResult result;
    result.n_instances = n_instances;
    result.n_visible = n_visible;
    result.per_primitive_buffers = per_primitive_buffers_base;
    result.per_primitive_buffers_size = PerPrimitiveBuffers::required_persistent(n_visible);
    result.primitive_work_indices = visibility_buffers.primitive_work_indices;
    result.primitive_normals = primitive_normals;
    result.sorted_primitive_indices = sorted_primitive_indices;
    result.sorted_primitive_indices_size = static_cast<size_t>(std::max(n_instances, 0)) * sizeof(uint);
    result.per_instance_sort_total_size = per_instance_sort_total_size;
    result.per_instance_sort_scratch_size = per_instance_sort_total_size > result.sorted_primitive_indices_size
                                                ? per_instance_sort_total_size - result.sorted_primitive_indices_size
                                                : 0;
    return result;
}
