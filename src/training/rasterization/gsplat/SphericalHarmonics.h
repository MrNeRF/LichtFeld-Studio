/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    void launch_spherical_harmonics_swizzled_fwd_kernel(
        uint32_t degrees_to_use,
        uint32_t layout_degree,
        uint32_t color_stride,
        const float* dirs,             // [..., 3]
        const float* sh0,              // [N, 1, 3] / [N, 3]
        const float* sh_rest_swizzled, // FP32 swizzle, or q16 codes when bounds are supplied
        const bool* masks,             // [...] optional
        int64_t n_elements,
        float* colors, // [..., 3]
        cudaStream_t stream = nullptr,
        const float* sh_value_bounds = nullptr);

    void launch_spherical_harmonics_swizzled_bwd_kernel(
        uint32_t degrees_to_use,
        uint32_t layout_degree,
        uint32_t color_stride,
        const float* dirs,             // [..., 3]
        const float* sh0,              // [N, 1, 3] / [N, 3]
        const float* sh_rest_swizzled, // FP32 swizzle, or q16 codes when bounds are supplied
        const bool* masks,             // [...] optional
        const float* v_colors,         // [..., 3]
        int64_t n_elements,
        int32_t K,
        bool compute_v_dirs,
        float* v_coeffs, // [..., K, 3] canonical output, unless grad_sh0 is set
        float* v_dirs,   // [..., 3] optional
        cudaStream_t stream = nullptr,
        const float* sh_value_bounds = nullptr,
        float* grad_sh0 = nullptr,    // [N, 1, 3]: adds the gradient here instead
        float* grad_shN = nullptr,    // swizzled FP32 shN gradient, optional
        uint32_t grad_shN_slots = 0); // float4 slots per primitive of grad_shN

} // namespace gsplat_lfs
