/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include "rasterization_config.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>

namespace fast_lfs::rasterization {

    struct ForwardResult {
        int n_instances = 0;
        int n_visible = 0;
        char* per_primitive_buffers = nullptr;
        size_t per_primitive_buffers_size = 0;
        uint* primitive_work_indices = nullptr;
        float3* primitive_normals = nullptr;
        uint* sorted_primitive_indices = nullptr;
        size_t sorted_primitive_indices_size = 0;
        size_t per_instance_sort_scratch_size = 0;
        size_t per_instance_sort_total_size = 0;
    };

    /// Sorted indices live in the owning rasterizer arena frame. The frame
    /// release returns the whole allocation after backward has finished.
    void release_sorted_primitive_indices(void* ptr, cudaStream_t stream) noexcept;

    /// Source-compatible no-op retained for callers of the removed TLS cache.
    /// Sort storage is released with its rasterizer arena frame.
    void release_sort_workspace_buffers() noexcept;

    /// Legacy compatibility counter; exact arena sizing has no fallback path.
    [[nodiscard]] std::uint64_t n_instances_fallback_sync_count() noexcept;
    void reset_n_instances_fallback_sync_count() noexcept;
    /// Compatibility hook; exact sizing always resolves n_instances first.
    void set_force_n_instances_sync_for_testing(bool force) noexcept;
    /// Compatibility hook; there is no retained sort capacity to drop.
    void reset_sort_capacity_for_testing() noexcept;

    /// Legacy TLS telemetry accessors. Sort storage is now reported through
    /// the rasterizer arena and these return zero.
    [[nodiscard]] std::size_t sort_workspace_required_bytes() noexcept;
    [[nodiscard]] std::size_t sort_workspace_allocated_bytes() noexcept;
    [[nodiscard]] int sort_workspace_capacity_n_instances() noexcept;

    // Warp-cull mode for blend_cu:
    /// 0 = enabled (production), 1 = disabled (all-1s mask, reference),
    /// 2 = wrong empty mask (deliberately incorrect).
    void set_warp_cull_mode_for_testing(int mode) noexcept;
    [[nodiscard]] int warp_cull_mode_for_testing() noexcept;
    /// Override forward blend fetch batch size (multiple of 32, 32..256). 0 = config default.
    void set_blend_batch_size_for_testing(int batch_size) noexcept;
    [[nodiscard]] int blend_batch_size_for_testing() noexcept;

    ForwardResult forward(
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
        const float2* sh_value_bounds, // null = fp32 or IEEE f16
        unsigned int sh_value_n_cells,
        unsigned int sh_value_bits, // 0=fp32, 16=q16 (with bounds) or IEEE f16 (no bounds)
        const float4* w2c,
        const float3* cam_position,
        float* image,
        float* alpha,
        float* depth,
        float* normal,         // [3*H*W] or nullptr
        const float* bg_color, // [3] device solid bg, or nullptr
        const float* bg_image, // [3*H*W] CHW per-pixel bg, or nullptr (wins over bg_color)
        const int n_primitives,
        const int active_sh_bases,
        const int sh_layout_bases,
        const int width,
        const int height,
        const float fx,
        const float fy,
        const float cx,
        const float cy,
        const float near,
        const float far,
        bool mip_filter,
        cudaStream_t stream,
        float* max_screen_share = nullptr,
        FastGSCameraKind camera_kind = FastGSCameraKind::PINHOLE,
        float fisheye_k1 = 0.0f,
        float fisheye_k2 = 0.0f,
        float fisheye_k3 = 0.0f,
        float fisheye_k4 = 0.0f,
        float fisheye_theta_max = 0.0f);

} // namespace fast_lfs::rasterization
