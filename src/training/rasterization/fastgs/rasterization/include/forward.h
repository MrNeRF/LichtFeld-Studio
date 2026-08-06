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

    /// Phase 1.1: sorted indices live in grow-only thread-local cache — no free.
    void release_sorted_primitive_indices(void* ptr, cudaStream_t stream) noexcept;

    /// Phase 1.2: mid-pipeline n_instances hard-sync fallback count (warmup/growth).
    [[nodiscard]] std::uint64_t n_instances_fallback_sync_count() noexcept;
    void reset_n_instances_fallback_sync_count() noexcept;
    /// Testing: force the next forward(s) onto the mid-pipeline sync path.
    void set_force_n_instances_sync_for_testing(bool force) noexcept;
    /// Testing: drop high-water capacity so the next forward must grow+sync.
    void reset_sort_capacity_for_testing() noexcept;

    ForwardResult forward(
        std::function<char*(size_t)> per_primitive_buffers_func,
        std::function<char*(size_t)> per_tile_buffers_func,
        const float3* means,
        const float3* scales_raw,
        const float4* rotations_raw,
        const float* opacities_raw,
        const float3* sh_coefficients_0,
        const float4* sh_coefficients_rest, // compact float4-packed swizzled layout
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
