/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace gsplat_fwd {

    void launch_spherical_harmonics_fwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,              // [M, 3]
        const float* coeffs,            // [N_total, K, 3] (N-sized when using visible_indices)
        const bool* masks,              // [M] optional
        const int32_t* visible_indices, // [M] maps elem_id -> global_idx, nullptr = direct
        int64_t n_elements,             // M (visible gaussians)
        int32_t K,
        float* colors, // [M, 3]
        cudaStream_t stream = nullptr);

    void launch_spherical_harmonics_bwd_kernel(
        uint32_t degrees_to_use,
        const float* dirs,     // [..., 3]
        const float* coeffs,   // [..., K, 3]
        const bool* masks,     // [...] optional
        const float* v_colors, // [..., 3]
        int64_t n_elements,
        int32_t K,
        bool compute_v_dirs,
        float* v_coeffs, // [..., K, 3]
        float* v_dirs,   // [..., 3] optional
        cudaStream_t stream = nullptr);

} // namespace gsplat_fwd
