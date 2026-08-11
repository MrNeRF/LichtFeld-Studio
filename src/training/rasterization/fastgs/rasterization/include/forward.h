/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "helper_math.h"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>

namespace fast_lfs::rasterization {

    struct ForwardResult {
        int n_instances = 0;
        uint* sorted_primitive_indices = nullptr;
        size_t sorted_primitive_indices_size = 0;
        size_t per_instance_sort_scratch_size = 0;
        size_t per_instance_sort_total_size = 0;
    };

    /// Sorted indices live in the persistent exact thread-local block — no free.
    void release_sorted_primitive_indices(void* ptr, cudaStream_t stream) noexcept;

    /// Release the consolidated FastGS sort workspace on the calling thread
    /// (keys×2, indices×2, CUB WS; pinned n_instances slot kept). Call at an
    /// established workspace boundary so VRAM is returned before worker join.
    void release_sort_workspace_buffers() noexcept;

    /// mid-pipeline n_instances hard-sync fallback count (warmup/growth).
    [[nodiscard]] std::uint64_t n_instances_fallback_sync_count() noexcept;
    void reset_n_instances_fallback_sync_count() noexcept;
    /// Testing: force the next forward(s) onto the mid-pipeline sync path.
    void set_force_n_instances_sync_for_testing(bool force) noexcept;
    /// Testing: drop retained capacity so the next forward must grow+sync.
    void reset_sort_capacity_for_testing() noexcept;

    /// Exact retained sort TLS residency on the calling thread
    /// (keys×2 + indices×2 + alignment padding + CUB WS).
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
        float* normal,             // [3*H*W] or nullptr
        float3* primitive_normals, // [N] scratch, required when normal != nullptr
        const float* bg_color,     // [3] device solid bg, or nullptr
        const float* bg_image,     // [3*H*W] CHW per-pixel bg, or nullptr (wins over bg_color)
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
        cudaStream_t stream);

} // namespace fast_lfs::rasterization
