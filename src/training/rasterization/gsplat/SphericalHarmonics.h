/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_lfs {

    void launch_spherical_harmonics_swizzled_fwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,             // [..., 3]
        const float* sh0,              // [N, 1, 3] / [N, 3]
        const float* sh_rest_swizzled, // vksplat swizzled SH-rest storage
        const bool* masks,             // [...] optional
        int64_t n_elements,
        float* colors, // [..., 3]
        cudaStream_t stream = nullptr);

    void launch_spherical_harmonics_swizzled_bwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,             // [..., 3]
        const float* sh0,              // [N, 1, 3] / [N, 3]
        const float* sh_rest_swizzled, // vksplat swizzled SH-rest storage
        const bool* masks,             // [...] optional
        const float* v_colors,         // [..., 3]
        int64_t n_elements,
        int32_t K,
        bool compute_v_dirs,
        float* v_coeffs, // [..., K, 3] canonical output
        float* v_dirs,   // [..., 3] optional
        cudaStream_t stream = nullptr);

} // namespace gsplat_lfs
