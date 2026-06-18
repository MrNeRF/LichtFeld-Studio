/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "SphericalHarmonics.h"
#include "Common.h"
#include "Ops.h"

#include <cuda_runtime.h>

namespace gsplat_lfs {

    void spherical_harmonics_swizzled_fwd(
        uint32_t degrees_to_use,
        const float* dirs,
        const float* sh0,
        const float* sh_rest_swizzled,
        const bool* masks,
        int64_t total_elements,
        float* colors,
        cudaStream_t stream) {
        if (total_elements == 0) {
            return;
        }

        launch_spherical_harmonics_swizzled_fwd_kernel(
            degrees_to_use,
            dirs, sh0, sh_rest_swizzled, masks,
            total_elements,
            colors, stream);
    }

    void spherical_harmonics_swizzled_bwd(
        uint32_t K,
        uint32_t degrees_to_use,
        const float* dirs,
        const float* sh0,
        const float* sh_rest_swizzled,
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

        launch_spherical_harmonics_swizzled_bwd_kernel(
            degrees_to_use,
            dirs, sh0, sh_rest_swizzled, masks, v_colors,
            total_elements, K,
            compute_v_dirs,
            v_coeffs, v_dirs, stream);
    }

} // namespace gsplat_lfs
