/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "buffer_utils.h"
#include "core/igs_failure_diagnostics.hpp"
#include "core/logger.hpp"
#include "forward.h"
#include "helper_math.h"
#include "rasterization_config.h"
#include "utils.h"
#include <array>
#include <cub/cub.cuh>
#include <functional>
#include <stdexcept>
#include <vector>

namespace {

    struct PrimitiveFailureOffender {
        uint primitive_idx = 0;
        uint global_idx = 0;
        uint n_touched_tiles = 0;
        ushort4 screen_bounds = make_ushort4(0, 0, 0, 0);
        float2 mean2d = make_float2(0.0f, 0.0f);
        float4 conic_opacity = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    };

    template <typename T>
    bool copy_device_value(const T* device_ptr, T& host_value) {
        return cudaMemcpy(&host_value, device_ptr, sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    void log_last_igs_plus_failure_snapshot() {
        lfs::core::IGSPlusFailureSnapshot snapshot;
        if (!lfs::core::try_get_igs_plus_failure_snapshot(snapshot)) {
            return;
        }

        LOG_ERROR(
            "Last igs+ refine before failure: iter={}, size_before={}, active_before={}, free_before={}, budget={}, budget_for_alloc={}, candidate_budget={}, selectable={}, selected={}, filled={}, appended={}, active_after={}, free_after={}, sampled_scale_p95={}, sampled_scale_max={}, sampled_scale_exp_max={}",
            snapshot.iter,
            snapshot.size_before,
            snapshot.active_before,
            snapshot.free_before,
            snapshot.budget,
            snapshot.budget_for_alloc,
            snapshot.candidate_budget,
            snapshot.selectable,
            snapshot.selected,
            snapshot.num_filled,
            snapshot.num_appended,
            snapshot.active_after,
            snapshot.free_after,
            snapshot.sampled_scale_p95,
            snapshot.sampled_scale_max,
            snapshot.sampled_scale_exp_max);
    }

    void log_instance_failure_offenders(
        const lfs::rendering::PerPrimitiveBuffers& per_primitive_buffers,
        int n_primitives,
        uint n_visible_primitives,
        uint n_instances) {
        std::vector<uint> n_touched_tiles(static_cast<size_t>(n_primitives));
        const auto copy_error = cudaMemcpy(
            n_touched_tiles.data(),
            per_primitive_buffers.n_touched_tiles,
            sizeof(uint) * static_cast<size_t>(n_primitives),
            cudaMemcpyDeviceToHost);
        if (copy_error != cudaSuccess) {
            LOG_ERROR(
                "Failed to copy n_touched_tiles for raster failure diagnostics: {}",
                cudaGetErrorString(copy_error));
            return;
        }

        std::array<PrimitiveFailureOffender, 8> top_offenders{};
        for (int primitive_idx = 0; primitive_idx < n_primitives; ++primitive_idx) {
            const uint touched_tiles = n_touched_tiles[static_cast<size_t>(primitive_idx)];
            if (touched_tiles == 0) {
                continue;
            }

            for (size_t slot = 0; slot < top_offenders.size(); ++slot) {
                if (touched_tiles <= top_offenders[slot].n_touched_tiles) {
                    continue;
                }

                for (size_t shift = top_offenders.size() - 1; shift > slot; --shift) {
                    top_offenders[shift] = top_offenders[shift - 1];
                }
                top_offenders[slot].primitive_idx = static_cast<uint>(primitive_idx);
                top_offenders[slot].n_touched_tiles = touched_tiles;
                break;
            }
        }

        LOG_ERROR(
            "Rasterization failure summary: n_primitives={}, n_visible_primitives={}, n_instances={}, avg_instances_per_visible={:.2f}",
            n_primitives,
            n_visible_primitives,
            n_instances,
            n_visible_primitives > 0 ? static_cast<double>(n_instances) / static_cast<double>(n_visible_primitives) : 0.0);

        for (size_t rank = 0; rank < top_offenders.size(); ++rank) {
            auto& offender = top_offenders[rank];
            if (offender.n_touched_tiles == 0) {
                break;
            }

            copy_device_value(per_primitive_buffers.global_idx + offender.primitive_idx, offender.global_idx);
            copy_device_value(per_primitive_buffers.screen_bounds + offender.primitive_idx, offender.screen_bounds);
            copy_device_value(per_primitive_buffers.mean2d + offender.primitive_idx, offender.mean2d);
            copy_device_value(per_primitive_buffers.conic_opacity + offender.primitive_idx, offender.conic_opacity);

            const uint bounds_width = static_cast<uint>(offender.screen_bounds.y - offender.screen_bounds.x);
            const uint bounds_height = static_cast<uint>(offender.screen_bounds.w - offender.screen_bounds.z);

            LOG_ERROR(
                "Rasterization offender[{}]: primitive_idx={}, global_idx={}, touched_tiles={}, bounds=({}, {})-({}, {}), size={}x{}, mean2d=({}, {}), opacity={}",
                rank,
                offender.primitive_idx,
                offender.global_idx,
                offender.n_touched_tiles,
                offender.screen_bounds.x,
                offender.screen_bounds.z,
                offender.screen_bounds.y,
                offender.screen_bounds.w,
                bounds_width,
                bounds_height,
                offender.mean2d.x,
                offender.mean2d.y,
                offender.conic_opacity.w);
        }
    }

    void log_rasterization_failure_diagnostics(
        const lfs::rendering::PerPrimitiveBuffers& per_primitive_buffers,
        int n_primitives,
        uint n_visible_primitives,
        uint n_instances) {
        log_last_igs_plus_failure_snapshot();
        log_instance_failure_offenders(per_primitive_buffers, n_primitives, n_visible_primitives, n_instances);
    }

} // namespace

// Selection colors (__constant__ must be defined before kernels_forward.cuh)
namespace lfs::rendering::config {
    __constant__ float3 SELECTION_GROUP_COLORS[MAX_SELECTION_GROUPS] = {
        {0.0f, 0.604f, 0.733f}, // 0: center marker (cyan)
        {1.0f, 0.3f, 0.3f},     // 1: red
        {0.3f, 1.0f, 0.3f},     // 2: green
        {0.3f, 0.5f, 1.0f},     // 3: blue
        {1.0f, 1.0f, 0.3f},     // 4: yellow
        {1.0f, 0.5f, 0.0f},     // 5: orange
        {0.8f, 0.3f, 1.0f},     // 6: purple
        {0.3f, 1.0f, 1.0f},     // 7: cyan
        {1.0f, 0.5f, 0.8f},     // 8: pink
    };
    __constant__ float3 SELECTION_COLOR_PREVIEW = {0.0f, 0.871f, 0.298f};

    void setSelectionGroupColor(const int group_id, const float3 color) {
        if (group_id >= 0 && group_id < MAX_SELECTION_GROUPS) {
            cudaMemcpyToSymbol(SELECTION_GROUP_COLORS, &color, sizeof(float3),
                               static_cast<size_t>(group_id) * sizeof(float3));
        }
    }

    void setSelectionPreviewColor(const float3 color) {
        cudaMemcpyToSymbol(SELECTION_COLOR_PREVIEW, &color, sizeof(float3));
    }
} // namespace lfs::rendering::config

#include "kernels_forward.cuh"

// Initialize mean2d buffer with invalid marker values
__global__ void init_mean2d_kernel(float2* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = make_float2(-10000.0f, -10000.0f);
    }
}

// Invalidate screen positions for gaussians outside crop box (uses precomputed flag)
__global__ void invalidate_outside_crop_kernel(
    float2* __restrict__ screen_positions,
    const bool* __restrict__ outside_crop,
    const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;
    if (outside_crop[idx]) {
        screen_positions[idx] = make_float2(-10000.0f, -10000.0f);
    }
}

// Copy mean2d to screen positions, flipping Y to match window coordinates
// The rasterizer's mean2d has Y increasing upward (OpenGL convention),
// but window coordinates have Y increasing downward
__global__ void copy_screen_positions_kernel(
    const float2* __restrict__ mean2d,
    float2* __restrict__ screen_positions_out,
    float height,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    float2 pos = mean2d[idx];
    // Flip Y: window_y = height - rasterizer_y
    // Keep invalid markers as-is (they have large negative values)
    if (pos.y > -1000.0f) {
        pos.y = height - pos.y;
    }
    screen_positions_out[idx] = pos;
}

// Simple kernel to select Gaussians within brush radius
__global__ void brush_select_kernel(
    const float2* __restrict__ screen_positions,
    float mouse_x,
    float mouse_y,
    float radius_sq,
    uint8_t* __restrict__ selection_out,
    int n_primitives) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_primitives)
        return;

    float2 pos = screen_positions[idx];

    // Skip invalid/off-screen positions (marked with large negative values)
    if (pos.x < -1000.0f || pos.y < -1000.0f)
        return;

    float dx = pos.x - mouse_x;
    float dy = pos.y - mouse_y;
    float dist_sq = dx * dx + dy * dy;

    if (dist_sq <= radius_sq) {
        selection_out[idx] = 1;
    }
}

void lfs::rendering::brush_select(
    const float2* screen_positions,
    float mouse_x,
    float mouse_y,
    float radius,
    uint8_t* selection_out,
    int n_primitives) {

    if (n_primitives <= 0)
        return;

    constexpr int block_size = 256;
    int grid_size = (n_primitives + block_size - 1) / block_size;

    brush_select_kernel<<<grid_size, block_size>>>(
        screen_positions,
        mouse_x,
        mouse_y,
        radius * radius, // Pass squared radius to avoid sqrt in kernel
        selection_out,
        n_primitives);
}

// Ray casting point-in-polygon test
__device__ __forceinline__ bool point_in_polygon(
    const float px, const float py,
    const float2* __restrict__ poly,
    const int n) {
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const float yi = poly[i].y, yj = poly[j].y;
        if ((yi > py) != (yj > py)) {
            const float xi = poly[i].x, xj = poly[j].x;
            if (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
                inside = !inside;
        }
    }
    return inside;
}

__global__ void polygon_select_kernel(
    const float2* __restrict__ positions,
    const float2* __restrict__ polygon,
    const int num_verts,
    bool* __restrict__ selection,
    const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float2 pos = positions[idx];
    if (pos.x < -1000.0f)
        return; // Invalid position marker

    if (point_in_polygon(pos.x, pos.y, polygon, num_verts))
        selection[idx] = true;
}

__global__ void polygon_select_mode_kernel(
    const float2* __restrict__ positions,
    const float2* __restrict__ polygon,
    const int num_verts,
    bool* __restrict__ selection,
    const int n,
    const bool add_mode) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float2 pos = positions[idx];
    if (pos.x < -1000.0f)
        return;

    if (point_in_polygon(pos.x, pos.y, polygon, num_verts))
        selection[idx] = add_mode;
}

__global__ void rect_select_kernel(
    const float2* __restrict__ positions,
    const float x0, const float y0, const float x1, const float y1,
    bool* __restrict__ selection,
    const int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float2 pos = positions[idx];
    if (pos.x < -1000.0f)
        return;

    if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1)
        selection[idx] = true;
}

__global__ void rect_select_mode_kernel(
    const float2* __restrict__ positions,
    const float x0, const float y0, const float x1, const float y1,
    bool* __restrict__ selection,
    const int n,
    const bool add_mode) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    const float2 pos = positions[idx];
    if (pos.x < -1000.0f)
        return;

    if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1)
        selection[idx] = add_mode;
}

void lfs::rendering::rect_select(
    const float2* positions,
    const float x0, const float y0, const float x1, const float y1,
    bool* selection,
    const int n_primitives) {
    if (n_primitives <= 0)
        return;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_primitives + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rect_select_kernel<<<grid, BLOCK_SIZE>>>(positions, x0, y0, x1, y1, selection, n_primitives);
}

void lfs::rendering::rect_select_mode(
    const float2* positions,
    const float x0, const float y0, const float x1, const float y1,
    bool* selection,
    const int n_primitives,
    const bool add_mode) {
    if (n_primitives <= 0)
        return;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_primitives + BLOCK_SIZE - 1) / BLOCK_SIZE;
    rect_select_mode_kernel<<<grid, BLOCK_SIZE>>>(positions, x0, y0, x1, y1, selection, n_primitives, add_mode);
}

void lfs::rendering::set_selection_element(bool* selection, const int index, const bool value) {
    cudaMemcpy(selection + index, &value, sizeof(bool), cudaMemcpyHostToDevice);
}

void lfs::rendering::polygon_select(
    const float2* positions,
    const float2* polygon,
    const int num_vertices,
    bool* selection,
    const int n_primitives) {
    if (n_primitives <= 0 || num_vertices < 3)
        return;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_primitives + BLOCK_SIZE - 1) / BLOCK_SIZE;
    polygon_select_kernel<<<grid, BLOCK_SIZE>>>(positions, polygon, num_vertices, selection, n_primitives);
}

void lfs::rendering::polygon_select_mode(
    const float2* positions,
    const float2* polygon,
    const int num_vertices,
    bool* selection,
    const int n_primitives,
    const bool add_mode) {
    if (n_primitives <= 0 || num_vertices < 3)
        return;

    constexpr int BLOCK_SIZE = 256;
    const int grid = (n_primitives + BLOCK_SIZE - 1) / BLOCK_SIZE;
    polygon_select_mode_kernel<<<grid, BLOCK_SIZE>>>(positions, polygon, num_vertices, selection, n_primitives, add_mode);
}

// sorting is done separately for depth and tile as proposed in https://github.com/m-schuetz/Splatshop
void lfs::rendering::forward(
    std::function<char*(size_t)> per_primitive_buffers_func,
    std::function<char*(size_t)> per_tile_buffers_func,
    std::function<char*(size_t)> per_instance_buffers_func,
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
    float* depth,
    const int n_primitives,
    const int active_sh_bases,
    const int total_bases_sh_rest,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_, // near and far are macros in windowns
    const float far_,
    const bool show_rings,
    const float ring_width,
    const float* model_transforms,
    const int* transform_indices,
    const int num_transforms,
    const uint8_t* selection_mask,
    float2* screen_positions_out,
    bool cursor_active,
    float cursor_x,
    float cursor_y,
    float cursor_radius,
    bool preview_selection_add_mode,
    bool* preview_selection_out,
    bool cursor_saturation_preview,
    float cursor_saturation_amount,
    bool show_center_markers,
    const float* crop_box_transform,
    const float3* crop_box_min,
    const float3* crop_box_max,
    bool crop_inverse,
    bool crop_desaturate,
    int crop_parent_node_index,
    const float* ellipsoid_transform,
    const float3* ellipsoid_radii,
    bool ellipsoid_inverse,
    bool ellipsoid_desaturate,
    int ellipsoid_parent_node_index,
    const float* view_volume_transform,
    const float3* view_volume_min,
    const float3* view_volume_max,
    bool view_volume_cull,
    const bool* deleted_mask,
    unsigned long long* hovered_depth_id,
    int focused_gaussian_id,
    const bool* emphasized_node_mask,
    int num_selected_nodes,
    bool dim_non_emphasized,
    const bool* node_visibility_mask,
    int num_visibility_nodes,
    float emphasis_flash_intensity,
    bool orthographic,
    float ortho_scale,
    bool mip_filter,
    const int* visible_indices,
    int visible_count) {

    const dim3 grid(div_round_up(width, config::tile_width), div_round_up(height, config::tile_height), 1);
    const dim3 block(config::tile_width, config::tile_height, 1);
    const int n_tiles = grid.x * grid.y;

    char* per_tile_buffers_blob = per_tile_buffers_func(required<PerTileBuffers>(n_tiles));
    PerTileBuffers per_tile_buffers = PerTileBuffers::from_blob(per_tile_buffers_blob, n_tiles);

    static cudaStream_t memset_stream = 0;
    if constexpr (!config::debug) {
        static bool memset_stream_initialized = false;
        if (!memset_stream_initialized) {
            cudaStreamCreate(&memset_stream);
            memset_stream_initialized = true;
        }
        cudaMemsetAsync(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles, memset_stream);
    } else
        cudaMemset(per_tile_buffers.instance_ranges, 0, sizeof(uint2) * n_tiles);

    // Use visible_count for buffer allocation if visibility filtering is active
    const int buffer_n_primitives = (visible_count > 0 && visible_indices != nullptr)
                                        ? visible_count
                                        : n_primitives;

    char* per_primitive_buffers_blob = per_primitive_buffers_func(required<PerPrimitiveBuffers>(buffer_n_primitives));
    PerPrimitiveBuffers per_primitive_buffers = PerPrimitiveBuffers::from_blob(per_primitive_buffers_blob, buffer_n_primitives);

    cudaMemset(per_primitive_buffers.n_visible_primitives, 0, sizeof(uint));
    cudaMemset(per_primitive_buffers.n_instances, 0, sizeof(uint));

    // Initialize mean2d with invalid marker values for brush selection
    // Only visible Gaussians will have their mean2d updated by preprocess kernel
    if (screen_positions_out != nullptr) {
        constexpr int init_block = 256;
        int init_grid = (buffer_n_primitives + init_block - 1) / init_block;
        init_mean2d_kernel<<<init_grid, init_block>>>(per_primitive_buffers.mean2d, buffer_n_primitives);
    }

    kernels::forward::preprocess_cu<<<div_round_up(buffer_n_primitives, config::block_size_preprocess), config::block_size_preprocess>>>(
        means,
        scales_raw,
        rotations_raw,
        opacities_raw,
        sh_coefficients_0,
        sh_coefficients_rest,
        w2c,
        cam_position,
        per_primitive_buffers.depth_keys.Current(),
        per_primitive_buffers.primitive_indices.Current(),
        per_primitive_buffers.n_touched_tiles,
        per_primitive_buffers.screen_bounds,
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        per_primitive_buffers.depth,
        per_primitive_buffers.outside_crop,
        per_primitive_buffers.selection_status,
        per_primitive_buffers.global_idx,
        per_primitive_buffers.n_visible_primitives,
        per_primitive_buffers.n_instances,
        buffer_n_primitives,
        visible_indices,
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
        model_transforms,
        transform_indices,
        num_transforms,
        selection_mask,
        cursor_active,
        cursor_x,
        cursor_y,
        cursor_radius * cursor_radius, // Pass squared radius for efficient comparison
        preview_selection_add_mode,
        preview_selection_out,
        cursor_saturation_preview,
        cursor_saturation_amount,
        crop_box_transform,
        crop_box_min,
        crop_box_max,
        crop_inverse,
        crop_desaturate,
        crop_parent_node_index,
        ellipsoid_transform,
        ellipsoid_radii,
        ellipsoid_inverse,
        ellipsoid_desaturate,
        ellipsoid_parent_node_index,
        view_volume_transform,
        view_volume_min,
        view_volume_max,
        view_volume_cull,
        deleted_mask,
        focused_gaussian_id,
        hovered_depth_id,
        emphasized_node_mask,
        num_selected_nodes,
        dim_non_emphasized,
        node_visibility_mask,
        num_visibility_nodes,
        orthographic,
        ortho_scale,
        mip_filter);
    CHECK_CUDA(config::debug, "preprocess")

    // Copy screen positions if requested (for interactive overlay queries)
    // Note: When visibility filtering is active, screen positions are written directly
    // in the kernel using global_idx, so this copy is only needed without filtering
    if (screen_positions_out != nullptr && visible_indices == nullptr) {
        cudaMemcpy(screen_positions_out, per_primitive_buffers.mean2d,
                   sizeof(float2) * n_primitives, cudaMemcpyDeviceToDevice);

        // In desaturate mode, invalidate screen positions for outside gaussians
        // Check crop box desaturate, ellipsoid desaturate, and depth filter
        const bool has_view_volume = (view_volume_transform != nullptr);
        if (crop_desaturate || ellipsoid_desaturate || has_view_volume) {
            constexpr int BLOCK = 256;
            const int grid_size = (n_primitives + BLOCK - 1) / BLOCK;
            invalidate_outside_crop_kernel<<<grid_size, BLOCK>>>(
                screen_positions_out, per_primitive_buffers.outside_crop, n_primitives);
        }
    }

    uint n_visible_primitives;
    cudaMemcpy(&n_visible_primitives, per_primitive_buffers.n_visible_primitives, sizeof(uint), cudaMemcpyDeviceToHost);
    uint n_instances;
    cudaMemcpy(&n_instances, per_primitive_buffers.n_instances, sizeof(uint), cudaMemcpyDeviceToHost);

    if (n_visible_primitives > 0x7fffffffu) {
        log_rasterization_failure_diagnostics(per_primitive_buffers, n_primitives, n_visible_primitives, n_instances);
        LOG_ERROR("Rasterization suspicious visible primitive count: {}", n_visible_primitives);
        throw std::runtime_error("Rasterization failed: visible primitive count exceeds 32-bit launch range");
    }
    if (n_instances > 0x7fffffffu) {
        log_rasterization_failure_diagnostics(per_primitive_buffers, n_primitives, n_visible_primitives, n_instances);
        LOG_ERROR(
            "Rasterization suspicious instance count: {} instances across {} visible primitives",
            n_instances,
            n_visible_primitives);
        throw std::runtime_error("Rasterization failed: instance count exceeds 32-bit allocation range");
    }

    const int n_visible_primitives_i = static_cast<int>(n_visible_primitives);
    const int n_instances_i = static_cast<int>(n_instances);
    const int alloc_instances = std::max(static_cast<int>(n_instances), 1);
    const size_t per_instance_bytes = required<PerInstanceBuffers>(alloc_instances);
    constexpr size_t hard_alloc_bytes = size_t{128} << 30;
    if (per_instance_bytes > hard_alloc_bytes) {
        log_rasterization_failure_diagnostics(per_primitive_buffers, n_primitives, n_visible_primitives, n_instances);
        LOG_ERROR(
            "Rasterization rejecting suspicious instance allocation: {} instances across {} visible primitives would request {:.2f} GiB",
            n_instances,
            n_visible_primitives,
            static_cast<double>(per_instance_bytes) / (1024.0 * 1024.0 * 1024.0));
        throw std::runtime_error("Rasterization failed: suspicious per-instance allocation request");
    }

    char* per_instance_buffers_blob = per_instance_buffers_func(per_instance_bytes);
    PerInstanceBuffers per_instance_buffers = PerInstanceBuffers::from_blob(per_instance_buffers_blob, alloc_instances);

    if (n_visible_primitives > 0) {
        cub::DeviceRadixSort::SortPairs(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.depth_keys,
            per_primitive_buffers.primitive_indices,
            n_visible_primitives_i);
        CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Depth)")

        kernels::forward::apply_depth_ordering_cu<<<div_round_up(n_visible_primitives_i, config::block_size_apply_depth_ordering), config::block_size_apply_depth_ordering>>>(
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.n_touched_tiles,
            per_primitive_buffers.offset,
            n_visible_primitives_i);
        CHECK_CUDA(config::debug, "apply_depth_ordering")

        cub::DeviceScan::ExclusiveSum(
            per_primitive_buffers.cub_workspace,
            per_primitive_buffers.cub_workspace_size,
            per_primitive_buffers.offset,
            per_primitive_buffers.offset,
            n_visible_primitives_i);
        CHECK_CUDA(config::debug, "cub::DeviceScan::ExclusiveSum (Primitive Offsets)")

        kernels::forward::create_instances_cu<<<div_round_up(n_visible_primitives_i, config::block_size_create_instances), config::block_size_create_instances>>>(
            per_primitive_buffers.primitive_indices.Current(),
            per_primitive_buffers.offset,
            per_primitive_buffers.screen_bounds,
            per_primitive_buffers.mean2d,
            per_primitive_buffers.conic_opacity,
            per_instance_buffers.keys.Current(),
            per_instance_buffers.primitive_indices.Current(),
            grid.x,
            n_visible_primitives_i);
        CHECK_CUDA(config::debug, "create_instances")

        if (n_instances > 0) {
            cub::DeviceRadixSort::SortPairs(
                per_instance_buffers.cub_workspace,
                per_instance_buffers.cub_workspace_size,
                per_instance_buffers.keys,
                per_instance_buffers.primitive_indices,
                n_instances_i);
            CHECK_CUDA(config::debug, "cub::DeviceRadixSort::SortPairs (Tile)")
        }
    }

    if constexpr (!config::debug)
        cudaStreamSynchronize(memset_stream);

    if (n_instances > 0) {
        kernels::forward::extract_instance_ranges_cu<<<div_round_up(n_instances_i, config::block_size_extract_instance_ranges), config::block_size_extract_instance_ranges>>>(
            per_instance_buffers.keys.Current(),
            per_tile_buffers.instance_ranges,
            n_instances_i);
        CHECK_CUDA(config::debug, "extract_instance_ranges")
    }

    kernels::forward::blend_cu<<<grid, block>>>(
        per_tile_buffers.instance_ranges,
        per_instance_buffers.primitive_indices.Current(),
        per_primitive_buffers.mean2d,
        per_primitive_buffers.conic_opacity,
        per_primitive_buffers.color,
        per_primitive_buffers.depth,
        per_primitive_buffers.outside_crop,
        per_primitive_buffers.selection_status,
        per_primitive_buffers.global_idx,
        image,
        alpha,
        depth,
        width,
        height,
        grid.x,
        show_rings,
        ring_width,
        show_center_markers,
        emphasis_flash_intensity,
        transform_indices,
        emphasized_node_mask,
        num_selected_nodes);
    CHECK_CUDA(config::debug, "blend")
}
