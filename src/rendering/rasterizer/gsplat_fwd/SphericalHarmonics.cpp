/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "SphericalHarmonics.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_fwd {

    void spherical_harmonics_fwd(
        uint32_t degrees_to_use,
        const float* dirs,
        const float* coeffs,
        const bool* masks,
        const int32_t* visible_indices,
        int64_t total_elements,
        int32_t K,
        float* colors,
        cudaStream_t stream) {
        if (total_elements == 0) {
            return;
        }

        launch_spherical_harmonics_fwd_kernel(
            degrees_to_use,
            dirs, coeffs, masks, visible_indices,
            total_elements, K,
            colors, stream);
    }

    void spherical_harmonics_bwd(
        uint32_t K,
        uint32_t degrees_to_use,
        const float* dirs,
        const float* coeffs,
        const bool* masks,
        const float* v_colors,
        int64_t total_elements,
        bool compute_v_dirs,
        float* v_coeffs,
        float* v_dirs,
        cudaStream_t stream) {
        if (total_elements == 0) {
            return;
        }

        launch_spherical_harmonics_bwd_kernel(
            degrees_to_use,
            dirs, coeffs, masks, v_colors,
            total_elements, K,
            compute_v_dirs,
            v_coeffs, v_dirs, stream);
    }

} // namespace gsplat_fwd
